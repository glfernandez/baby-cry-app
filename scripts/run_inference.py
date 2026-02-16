"""
Run the pretrained babycry classifier on custom audio recordings.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.audio_features import FeatureSpec, extract_feature_matrix, to_model_input
else:
    from .audio_features import FeatureSpec, extract_feature_matrix, to_model_input

LABEL_MAP: Dict[int, str] = {
    0: "hungry",
    1: "needs burping",
    2: "belly pain",
    3: "discomfort",
    4: "tired",
    5: "lonely",
    6: "cold or hot",
    7: "scared",
    8: "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Audio files or directories containing WAV recordings.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../Work_in_progress/best_accuracy/babycry_model_wip3_model.h5"),
        help="Path to the pretrained Keras model.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("../models/feature_scaler.pkl"),
        help="Path to the feature StandardScaler produced by prepare_scaler.py.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Expect single-channel input instead of duplicating mono recordings.",
    )
    return parser.parse_args()


def expand_sources(sources: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for src in sources:
        src = src.resolve()
        if src.is_dir():
            resolved.extend(sorted(src.rglob("*.wav")))
        else:
            resolved.append(src)
    return resolved


def load_model(model_path: Path) -> keras.Model:
    keras.backend.set_image_data_format("channels_first")
    return keras.models.load_model(model_path)


def load_scaler(scaler_path: Path) -> StandardScaler:
    scaler = joblib.load(scaler_path)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Scaler at {scaler_path} is not a sklearn.preprocessing.StandardScaler instance.")
    return scaler


def majority_vote(predictions: np.ndarray) -> Tuple[int, float]:
    if predictions.ndim == 1:
        best_label = int(np.argmax(predictions))
        confidence = float(predictions[best_label])
        return best_label, confidence
    frame_indices = np.argmax(predictions, axis=-1)
    counts = Counter(frame_indices)
    best_label, _ = counts.most_common(1)[0]
    confidence = float(np.mean(predictions[:, best_label]))
    return best_label, confidence


def classify(
    model: keras.Model,
    scaler: StandardScaler,
    wav_path: Path,
    spec: FeatureSpec,
) -> Tuple[str, float, np.ndarray]:
    features, channels = extract_feature_matrix(wav_path, spec)
    scaled = scaler.transform(features)
    batch = to_model_input(scaled, channels, spec)
    predictions = model.predict(batch, verbose=0)[0]
    label_idx, confidence = majority_vote(predictions)
    return LABEL_MAP[label_idx], confidence, predictions


def main() -> None:
    args = parse_args()
    sources = expand_sources(args.sources)
    if not sources:
        raise RuntimeError("No audio sources supplied.")

    model = load_model(args.model_path.resolve())
    scaler = load_scaler(args.scaler_path.resolve())
    spec = FeatureSpec(enforce_stereo=not args.mono)

    for wav_path in sources:
        if not wav_path.exists():
            print(f"[warn] Skipping missing file {wav_path}")
            continue
        label, confidence, frame_preds = classify(model, scaler, wav_path, spec)
        probs = frame_preds if frame_preds.ndim == 1 else np.mean(frame_preds, axis=0)
        probability_str = ", ".join(f"{LABEL_MAP[idx]}={prob:.2f}" for idx, prob in enumerate(probs))
        print(f"{wav_path.name}: {label} (confidence {confidence:.2f})")
        print(f"  frame-mean probabilities: {probability_str}")


if __name__ == "__main__":
    main()
