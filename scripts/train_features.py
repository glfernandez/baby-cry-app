"""
Train a dense classifier on the regenerated summary-feature dataset with optional
augmentation and CRNN distillation support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from audio_summary_features import SUMMARY_COLUMNS  # type: ignore  # pylint: disable=import-error
else:
    from .audio_summary_features import SUMMARY_COLUMNS  # type: ignore

LABEL_MAP = {
    0: "hungry",
    1: "needs_burping",
    2: "belly_pain",
    3: "discomfort",
    4: "tired",
    5: "lonely",
    6: "cold_hot",
    7: "scared",
    8: "dirty_diaper",
}


class CombinedHistory:
    def __init__(self) -> None:
        self.history: Dict[str, list[float]] = {}

    def update(self, new_history: keras.callbacks.History) -> None:
        for key, values in new_history.history.items():
            self.history.setdefault(key, []).extend(float(v) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Data/babycry_features_v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/features_aug"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tflite-friendly", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Pre-train using teacher probabilities when available.",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=30,
        help="Number of epochs for the teacher-matching phase.",
    )
    return parser.parse_args()


def load_dataset(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray], Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)

    if all(col in df.columns for col in SUMMARY_COLUMNS):
        feature_columns = list(SUMMARY_COLUMNS)
    else:
        excluded = {"Cry_Audio_File", "Cry_Reason", "Variant"}
        excluded.update(col for col in df.columns if col.startswith("Teacher_"))
        feature_columns = [col for col in df.columns if col not in excluded]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["Cry_Reason"].astype(int).to_numpy()
    sample_ids = df["Cry_Audio_File"].astype(str).tolist()
    variants = df["Variant"].astype(str).to_numpy() if "Variant" in df.columns else None

    teacher_columns = [col for col in df.columns if col.startswith("Teacher_")]
    if teacher_columns:
        teacher_df = df[teacher_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        teacher_probs = teacher_df.to_numpy(dtype=np.float32)
    else:
        teacher_probs = None

    return X, y, sample_ids, teacher_probs, variants


def build_model(input_dim: int, num_classes: int, use_batch_norm: bool) -> keras.Model:
    layers_list: list[keras.Layer] = [keras.Input(shape=(input_dim,))]
    if use_batch_norm:
        layers_list.append(layers.BatchNormalization())
    layers_list.extend(
        [
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return keras.Sequential(layers_list)


def save_metadata(
    output_dir: Path,
    *,
    command: str,
    dataset: Path,
    history: CombinedHistory,
    test_size: float,
    epochs: int,
    patience: int,
    seed: int,
    class_weight: bool,
    distill: bool,
    distill_epochs: int,
    metrics: Dict[str, float],
) -> None:
    metadata = {
        "run_name": output_dir.name,
        "command": command,
        "dataset": str(dataset),
        "label_map": LABEL_MAP,
        "test_size": test_size,
        "epochs_requested": epochs,
        "early_stopping_patience": patience,
        "seed": seed,
        "class_weight": class_weight,
        "distill": distill,
        "distill_epochs": distill_epochs,
        "metrics": history.history,
        "evaluation": metrics,
        "artifacts": {
            "model": "feature_model.keras",
            "scaler": "feature_scaler.pkl",
            "labels": "label_map.json",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
    from collections import Counter

    counts = Counter(labels.tolist())
    total = float(sum(counts.values()))
    return {label: total / (len(counts) * count) for label, count in counts.items()}


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    X, y, sample_ids, teacher_probs, variants = load_dataset(args.dataset)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    teacher_train = teacher_probs[train_idx] if teacher_probs is not None else None
    teacher_val = teacher_probs[val_idx] if teacher_probs is not None else None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(LABEL_MAP),
        use_batch_norm=not args.tflite_friendly,
    )

    history_tracker = CombinedHistory()

    if args.distill:
        if teacher_train is None:
            raise ValueError("Distillation requested but teacher probabilities are missing from dataset.")
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=[keras.metrics.KLDivergence(name="kl_div")],
        )
        distill_callbacks = []
        if args.patience > 0:
            distill_callbacks.append(
                EarlyStopping(monitor="val_loss", patience=max(1, args.patience // 2), restore_best_weights=True)
            )
        history_distill = model.fit(
            X_train,
            teacher_train,
            validation_data=(X_val, teacher_val),
            epochs=args.distill_epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=distill_callbacks,
        )
        history_tracker.update(history_distill)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
        )

    class_weight = compute_class_weight(y_train) if args.class_weight else None

    history_fine_tune = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    history_tracker.update(history_fine_tune)

    model_path = output_dir / "feature_model.keras"
    scaler_path = output_dir / "feature_scaler.pkl"
    labels_path = output_dir / "label_map.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    labels_path.write_text(json.dumps(LABEL_MAP, indent=2))

    predictions = model.predict(X_val, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(y_pred == y_val))

    report = classification_report(
        y_val,
        y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
    )
    cm = confusion_matrix(y_val, y_pred).tolist()
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    if variants is not None:
        val_variants = variants[val_idx]
        orig_mask = val_variants == "orig"
        if np.any(orig_mask):
            orig_acc = float(np.mean(y_pred[orig_mask] == y_val[orig_mask]))
            metrics["accuracy_original_only"] = orig_acc

    (output_dir / "classification_report.txt").write_text(report)
    (output_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))

    command_segments = [
        "python",
        "scripts/train_features.py",
        f"--dataset {args.dataset}",
        f"--output-dir {args.output_dir}",
        f"--test-size {args.test_size}",
        f"--epochs {args.epochs}",
        f"--batch-size {args.batch_size}",
        f"--patience {args.patience}",
        f"--seed {args.seed}",
    ]
    if args.tflite_friendly:
        command_segments.append("--tflite-friendly")
    if args.class_weight:
        command_segments.append("--class-weight")
    if args.distill:
        command_segments.extend(["--distill", f"--distill-epochs {args.distill_epochs}"])

    save_metadata(
        output_dir=output_dir,
        command=" ".join(command_segments),
        dataset=args.dataset.resolve(),
        history=history_tracker,
        test_size=args.test_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        class_weight=bool(class_weight),
        distill=args.distill,
        distill_epochs=args.distill_epochs,
        metrics=metrics,
    )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved label map to {labels_path}")
    print(f"Validation accuracy: {accuracy:.3f}")
    if "accuracy_original_only" in metrics:
        print(f"Original clips accuracy: {metrics['accuracy_original_only']:.3f}")


if __name__ == "__main__":
    main()
