"""
Compare predictions between the original Keras models and their TFLite
counterparts for a set of WAV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import tensorflow as tf

from audio_features import FeatureSpec, extract_feature_matrix, to_model_input
from audio_features_csv import wavs_to_dataframe
from export_tflite import (FEATURE_INPUT_DIM, build_crnn_model,
                           build_feature_fused_model,
                           load_feature_weights_npz)


def load_flex_delegate() -> List[tf.lite.experimental.load_delegate]:
    try:
        lib_path = Path(tf.sysconfig.get_lib()) / "libtensorflowlite_flex.dylib"
        if lib_path.exists():
            delegate = tf.lite.experimental.load_delegate(str(lib_path))
            return [delegate]
    except Exception as exc:  # pragma: no cover - delegate optional
        print(f"[warn] Failed to load Flex delegate: {exc}")
    return []


def run_crnn_parity(args: argparse.Namespace, wav_paths: Iterable[Path]) -> None:
    print("== CRNN parity check ==")
    keras_model = build_crnn_model()
    keras_model.load_weights(args.crnn_model)
    scaler = joblib.load(args.crnn_scaler)

    interpreter = tf.lite.Interpreter(
        model_path=str(args.crnn_tflite),
        experimental_delegates=load_flex_delegate(),
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"CRNN TFLite input shape: {input_details['shape']} dtype: {input_details['dtype']}")

    spec = FeatureSpec(enforce_stereo=True)

    for wav in wav_paths:
        features, channels = extract_feature_matrix(wav, spec)
        scaled = scaler.transform(features)
        batch = to_model_input(scaled, channels, spec).astype(np.float32)

        keras_pred = keras_model.predict(batch, verbose=0)[0]

        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details["index"])[0]

        diff = np.max(np.abs(keras_pred - tflite_pred))
        print(f"{wav.name}: Keras={keras_pred.argmax()} TFLite={tflite_pred.argmax()} max|Δ|={diff:.6f}")
        print("  keras :", np.round(keras_pred, 4))
        print("  tflite:", np.round(tflite_pred, 4))


def load_scaler_json(path: Path) -> dict[str, list[float]]:
    return json.loads(path.read_text())


def run_feature_parity(args: argparse.Namespace, wav_paths: Iterable[Path]) -> None:
    print("\n== Feature model parity check ==")
    weights = load_feature_weights_npz(args.feature_weights)
    gamma, beta, mean, var, W1, b1, W2, b2, W3, b3 = weights
    scale = gamma / np.sqrt(var + 1e-3)
    offset = beta - mean * scale
    W1_prime = W1 * scale[:, np.newaxis]
    b1_prime = offset @ W1 + b1

    keras_model = build_feature_fused_model(weights)

    interpreter = tf.lite.Interpreter(
        model_path=str(args.feature_tflite),
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"Feature TFLite input shape: {input_details['shape']} dtype: {input_details['dtype']}")

    scaler = joblib.load(args.feature_scaler)

    df = wavs_to_dataframe(wav_paths)
    feature_columns = [col for col in df.columns if col not in {"Cry_Audio_File", "Cry_Reason"}]
    X = df[feature_columns].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    keras_preds = keras_model.predict(X_scaled, verbose=0)

    diffs = []
    manual_diffs = []
    for idx, wav in enumerate(wav_paths):
        sample = X_scaled[idx : idx + 1]
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details["index"])[0]

        # Manual inference using fused weights
        z1 = np.maximum(0.0, sample @ W1_prime + b1_prime)
        z2 = np.maximum(0.0, z1 @ W2 + b2)
        logits = z2 @ W3 + b3
        probs = np.exp(logits - logits.max())
        manual_pred = (probs / probs.sum()).flatten()

        diff = np.max(np.abs(keras_preds[idx] - tflite_pred))
        manual_diff = np.max(np.abs(keras_preds[idx] - manual_pred))
        diffs.append(diff)
        manual_diffs.append(manual_diff)

        print(f"{wav.name}: Keras={keras_preds[idx].argmax()} TFLite={tflite_pred.argmax()} max|Δ|={diff:.6f}")
        print("  keras :", np.round(keras_preds[idx], 4))
        print("  tflite:", np.round(tflite_pred, 4))
        print("  manual:", np.round(manual_pred, 4), f"(max|Δ| vs Keras = {manual_diff:.6f})")

    print(f"\nFeature model max absolute difference across samples: {np.max(diffs):.6f}")
    print(f"Manual inference max|Δ| vs Keras: {np.max(manual_diffs):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("samples"),
        help="Directory containing WAV files to evaluate.",
    )
    parser.add_argument(
        "--crnn-model",
        type=Path,
        default=Path("models/long_run/babycry_sanity.h5"),
        help="Path to CRNN Keras model.",
    )
    parser.add_argument(
        "--crnn-tflite",
        type=Path,
        default=Path("models/long_run/babycry_sanity.tflite"),
        help="Path to CRNN TFLite model.",
    )
    parser.add_argument(
        "--crnn-scaler",
        type=Path,
        default=Path("models/long_run/sanity_scaler.pkl"),
        help="Path to CRNN StandardScaler pickle.",
    )
    parser.add_argument(
        "--feature-weights",
        type=Path,
        default=Path("models/features_run/feature_model_weights.npz"),
        help="Path to NumPy archive of feature model weights.",
    )
    parser.add_argument(
        "--feature-tflite",
        type=Path,
        default=Path("models/features_run/feature_model.tflite"),
        help="Path to feature dense TFLite model.",
    )
    parser.add_argument(
        "--feature-scaler",
        type=Path,
        default=Path("models/features_run/feature_scaler.pkl"),
        help="Path to feature StandardScaler pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_paths = sorted(args.wav_dir.glob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No WAV files found in {args.wav_dir}")

    run_crnn_parity(args, wav_paths)
    run_feature_parity(args, wav_paths)


if __name__ == "__main__":
    main()


