"""
Perform a lightweight sanity-check training run for the babycry model.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.audio_features import FeatureSpec, extract_feature_matrix, to_model_input
else:
    from .audio_features import FeatureSpec, extract_feature_matrix, to_model_input

LABELS: Dict[str, int] = {
    "hu": 0,
    "bu": 1,
    "bp": 2,
    "dc": 3,
    "ti": 4,
    "lo": 5,
    "ch": 6,
    "sc": 7,
    "dk": 8,
}


def discover_wavs(dataset_root: Path) -> List[Path]:
    wavs = sorted(dataset_root.rglob("*.wav"))
    return [p for p in wavs if p.is_file()]


def parse_label(path: Path) -> str:
    stem = path.stem
    return stem.split("-")[-1]


def select_subset(paths: Iterable[Path], max_per_label: int) -> List[Path]:
    buckets: Dict[str, List[Path]] = {key: [] for key in LABELS}
    for path in paths:
        label = parse_label(path)
        if label not in buckets:
            continue
        if len(buckets[label]) < max_per_label:
            buckets[label].append(path)
    for label, collected in buckets.items():
        if collected:
            random.shuffle(collected)
    subset: List[Path] = []
    for label_paths in buckets.values():
        subset.extend(label_paths)
    random.shuffle(subset)
    return subset


def build_dataset(paths: List[Path], spec: FeatureSpec) -> Tuple[np.ndarray, np.ndarray, StandardScaler, int]:
    scaler = StandardScaler()
    features_list: List[np.ndarray] = []
    labels: List[int] = []
    channels = 0

    for path in paths:
        label_code = parse_label(path)
        if label_code not in LABELS:
            continue
        feature_matrix, channel_count = extract_feature_matrix(path, spec)
        scaler.partial_fit(feature_matrix)
        features_list.append(feature_matrix)
        labels.append(LABELS[label_code])
        channels = max(channels, channel_count)

    if not features_list:
        raise RuntimeError("No features extracted. Check dataset path.")

    scaled_features = [scaler.transform(fm) for fm in features_list]
    tensors = np.concatenate([to_model_input(fm, channels, spec) for fm in scaled_features], axis=0)

    return tensors, np.array(labels), scaler, channels


def build_model(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    keras.backend.set_image_data_format("channels_first")
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu", data_format="channels_first")(inputs)
    x = layers.MaxPool2D(pool_size=(1, 3), data_format="channels_first")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((input_shape[1], -1))(x)
    x = layers.LSTM(32, return_sequences=True, dropout=0.25)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_sanity(
    dataset_root: Path,
    output_dir: Path,
    max_per_label: int,
    epochs: int,
    batch_size: int,
    seed: int,
    enforce_stereo: bool,
    patience: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    spec = FeatureSpec(enforce_stereo=enforce_stereo)

    candidates = discover_wavs(dataset_root)
    subset = select_subset(candidates, max_per_label=max_per_label)
    if len(subset) < len(LABELS):
        raise RuntimeError("Not enough labeled examples collected for sanity check.")

    print(f"Selected {len(subset)} audio files for training/validation")
    per_label_counts: Dict[str, int] = {label: 0 for label in LABELS}
    for path in subset:
        per_label_counts[parse_label(path)] += 1
    for label, idx in sorted(LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {per_label_counts[label]} clips")

    tensors, labels, scaler, channels = build_dataset(subset, spec)
    X_train, X_val, y_train, y_val = train_test_split(
        tensors, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(LABELS))
    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "babycry_sanity.h5"
    scaler_path = output_dir / "sanity_scaler.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved sanity model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print("History:")
    for key, values in history.history.items():
        print(f"  {key}: {[round(v, 4) for v in values]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("../Data/WAV_Audio_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("../models/sanity"))
    parser.add_argument("--max-per-label", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 disables)")
    parser.add_argument("--mono", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sanity(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        max_per_label=args.max_per_label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        enforce_stereo=not args.mono,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
