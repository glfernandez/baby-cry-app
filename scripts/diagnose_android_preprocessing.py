"""
Diagnostic script to compare Android preprocessing with Python preprocessing.

This script helps identify discrepancies between Android and Python preprocessing
by comparing logged Android values with Python-computed values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.audio_features import FeatureSpec, extract_feature_matrix, to_model_input
else:
    from .audio_features import FeatureSpec, extract_feature_matrix, to_model_input

LABEL_MAP = {
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


def parse_logcat_mel_sample(log_line: str) -> list[float] | None:
    """Extract mel sample values from a logcat line like 'Raw mel sample=[...]'."""
    if "Raw mel sample=" not in log_line:
        return None
    try:
        start = log_line.index("[") + 1
        end = log_line.index("]")
        values_str = log_line[start:end]
        return [float(v.strip()) for v in values_str.split(",")]
    except (ValueError, IndexError):
        return None


def parse_logcat_stats(log_line: str) -> dict[str, float] | None:
    """Extract stats from a logcat line like 'Raw mel stats: mean=... std=...'."""
    if "Raw mel stats:" not in log_line:
        return None
    stats = {}
    for part in log_line.split("Raw mel stats:")[1].strip().split():
        if "=" in part:
            key, value = part.split("=", 1)
            try:
                stats[key] = float(value)
            except ValueError:
                pass
    return stats if stats else None


def compare_mel_features(
    wav_path: Path,
    android_mel_sample: list[float] | None = None,
    android_stats: dict[str, float] | None = None,
) -> None:
    """Compare Android preprocessing with Python preprocessing for a WAV file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {wav_path.name}")
    print(f"{'='*60}")

    # Load audio
    audio, sr = librosa.load(wav_path, sr=44100, mono=False, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    spec = FeatureSpec(enforce_stereo=True)
    features, channels = extract_feature_matrix(wav_path, spec)

    # Extract first frame of first channel
    python_first_frame = features[0, :40]  # First 40 values (channel 0)
    python_mean = python_first_frame.mean()
    python_std = python_first_frame.std()
    python_min = python_first_frame.min()
    python_max = python_first_frame.max()

    print(f"\nPython preprocessing (first frame, channel 0):")
    print(f"  Sample (first 40): {python_first_frame[:40].tolist()}")
    print(f"  Mean: {python_mean:.4f}")
    print(f"  Std: {python_std:.4f}")
    print(f"  Min: {python_min:.4f}")
    print(f"  Max: {python_max:.4f}")

    if android_mel_sample:
        android_array = np.array(android_mel_sample)
        android_mean = android_array.mean()
        android_std = android_array.std()
        android_min = android_array.min()
        android_max = android_array.max()

        print(f"\nAndroid preprocessing (from logcat):")
        print(f"  Sample (first 40): {android_mel_sample}")
        print(f"  Mean: {android_mean:.4f}")
        print(f"  Std: {android_std:.4f}")
        print(f"  Min: {android_min:.4f}")
        print(f"  Max: {android_max:.4f}")

        print(f"\nDifferences:")
        print(f"  Mean diff: {abs(python_mean - android_mean):.4f}")
        print(f"  Std diff: {abs(python_std - android_std):.4f}")
        print(f"  Min diff: {abs(python_min - android_min):.4f}")
        print(f"  Max diff: {abs(python_max - android_max):.4f}")

        if len(android_mel_sample) == len(python_first_frame):
            mse = np.mean((python_first_frame - android_array) ** 2)
            print(f"  MSE: {mse:.6f}")
            max_diff = np.max(np.abs(python_first_frame - android_array))
            print(f"  Max absolute diff: {max_diff:.4f}")

    if android_stats:
        print(f"\nAndroid stats from logcat:")
        for key, value in android_stats.items():
            print(f"  {key}: {value:.4f}")


def test_inference(
    wav_path: Path,
    model_path: Path,
    scaler_path: Path,
) -> None:
    """Run inference and show predictions."""
    print(f"\n{'='*60}")
    print("Running inference...")
    print(f"{'='*60}")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    spec = FeatureSpec(enforce_stereo=True)
    features, channels = extract_feature_matrix(wav_path, spec)
    scaled = scaler.transform(features)
    batch = to_model_input(scaled, channels, spec)

    predictions = model.predict(batch, verbose=0)[0]
    if predictions.ndim == 2:
        # Average over frames
        probs = np.mean(predictions, axis=0)
    else:
        probs = predictions

    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    print(f"\nPredictions:")
    for idx, label in LABEL_MAP.items():
        prob = probs[idx]
        marker = " <--" if idx == best_idx else ""
        print(f"  {label}: {prob:.4f}{marker}")
    print(f"\nBest: {LABEL_MAP[best_idx]} (confidence: {confidence:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="Path to WAV file to analyze")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/long_run/babycry_sanity.keras"),
        help="Path to Keras model",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("models/long_run/sanity_scaler.pkl"),
        help="Path to scaler pickle file",
    )
    parser.add_argument(
        "--android-mel",
        type=str,
        help="Android mel sample from logcat (comma-separated or JSON array)",
    )
    parser.add_argument(
        "--android-stats",
        type=str,
        help="Android stats from logcat as JSON dict",
    )
    parser.add_argument(
        "--logcat-file",
        type=Path,
        help="Path to logcat file to parse",
    )
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    android_mel_sample = None
    android_stats = None

    if args.android_mel:
        try:
            android_mel_sample = json.loads(args.android_mel)
        except json.JSONDecodeError:
            android_mel_sample = [float(x.strip()) for x in args.android_mel.split(",")]
    elif args.logcat_file:
        # Parse logcat file
        with open(args.logcat_file) as f:
            for line in f:
                if mel := parse_logcat_mel_sample(line):
                    android_mel_sample = mel
                if stats := parse_logcat_stats(line):
                    android_stats = stats

    if args.android_stats:
        android_stats = json.loads(args.android_stats)

    compare_mel_features(wav_path, android_mel_sample, android_stats)

    if args.model_path.exists() and args.scaler_path.exists():
        test_inference(wav_path, args.model_path, args.scaler_path)
    else:
        print(f"\nSkipping inference (model or scaler not found)")


if __name__ == "__main__":
    main()

