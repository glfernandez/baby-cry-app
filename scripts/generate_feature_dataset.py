"""
Regenerate the summary-feature CSV from the curated WAV library with optional
augmentation and CRNN teacher probabilities for distillation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import tensorflow as tf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from audio_summary_features import (  # type: ignore  # pylint: disable=import-error
        SUMMARY_COLUMNS,
        SummaryFeatureConfig,
        feature_vector_from_audio,
        load_mono_audio,
    )
    from audio_features import FeatureSpec  # type: ignore  # pylint: disable=import-error
else:
    from .audio_summary_features import SUMMARY_COLUMNS, SummaryFeatureConfig, feature_vector_from_audio, load_mono_audio
    from .audio_features import FeatureSpec

LABEL_MAP = {
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

AUGMENTATIONS = (
    ("noise", lambda y, sr: add_noise(y, 0.006)),
    ("pitch_up", lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)),
    ("pitch_down", lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.5)),
    ("stretch_up", lambda y, sr: match_length(librosa.effects.time_stretch(y, rate=1.05), len(y))),
    ("stretch_down", lambda y, sr: match_length(librosa.effects.time_stretch(y, rate=0.95), len(y))),
)


def discover_wavs(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def label_from_filename(path: Path) -> Optional[str]:
    parts = path.stem.split("-")
    return parts[-1] if parts else None


def iter_labeled(paths: Iterable[Path]) -> Iterator[Tuple[Path, int]]:
    for path in paths:
        code = label_from_filename(path)
        if code is None or code not in LABEL_MAP:
            continue
        yield path, LABEL_MAP[code]


def add_noise(samples: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, std, samples.shape).astype(np.float32)
    augmented = samples + noise
    return np.clip(augmented, -1.0, 1.0)


def match_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    if len(samples) == target_len:
        return samples
    if len(samples) > target_len:
        return samples[:target_len]
    padding = target_len - len(samples)
    return np.pad(samples, (0, padding), mode="constant")


def collect_variants(
    base_samples: np.ndarray,
    sample_rate: int,
    *,
    include_augmentations: bool,
) -> List[Tuple[str, np.ndarray]]:
    variants = [("orig", base_samples)]
    if not include_augmentations:
        return variants
    for name, transform in AUGMENTATIONS:
        augmented = transform(base_samples, sample_rate)
        variants.append((name, np.clip(augmented, -1.0, 1.0)))
    return variants


def prepare_teacher_context(
    model_path: Optional[Path],
    scaler_path: Optional[Path],
    sample_rate: int,
) -> Optional[Tuple[tf.lite.Interpreter, joblib, FeatureSpec]]:
    if model_path is None or scaler_path is None:
        return None
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    scaler = joblib.load(scaler_path)
    spec = FeatureSpec(sample_rate=sample_rate)
    return interpreter, scaler, spec


def teacher_probabilities(
    context: Optional[Tuple[tf.lite.Interpreter, joblib, FeatureSpec]],
    samples: np.ndarray,
    sample_rate: int,
) -> Optional[np.ndarray]:
    if context is None:
        return None
    interpreter, scaler, spec = context
    target_rate = spec.sample_rate
    if sample_rate != target_rate:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_rate)
    stereo = np.vstack([samples, samples])
    mel_frames = []
    for channel_audio in stereo:
        mel = librosa.feature.melspectrogram(
            y=channel_audio,
            sr=target_rate,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            n_mels=spec.n_mels,
            power=1.0,
        )
        log_mel = np.log(np.maximum(mel, 1e-10)).T
        mel_frames.append(fit_to_frames(log_mel, spec.target_frames))
    feature_matrix = np.concatenate(mel_frames, axis=1)
    scaled = scaler.transform(feature_matrix)
    tensor = scaled.reshape(spec.target_frames, stereo.shape[0], spec.n_mels).transpose(1, 0, 2)
    tensor = tensor[np.newaxis, ...].astype(np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, tensor.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)[0]


def fit_to_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    current = features.shape[0]
    if current == target_frames:
        return features
    if current > target_frames:
        return features[:target_frames]
    pad = np.zeros((target_frames - current, features.shape[1]), dtype=features.dtype)
    return np.vstack([features, pad])


def format_row(
    identifier: str,
    label_id: int,
    features: np.ndarray,
    variant: str,
    teacher_probs: Optional[np.ndarray],
    teacher_columns: List[str],
) -> List[str]:
    row = [identifier, str(label_id), variant]
    row.extend(f"{float(value):.9f}" for value in features)
    if teacher_probs is not None:
        row.extend(f"{float(p):.9f}" for p in teacher_probs)
    else:
        row.extend("" for _ in teacher_columns)
    return row


def write_dataset(
    rows: List[List[str]],
    output: Path,
    teacher_columns: List[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Cry_Audio_File", "Cry_Reason", "Variant", *SUMMARY_COLUMNS]
    headers.extend(teacher_columns)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("Data/files"))
    parser.add_argument("--output", type=Path, default=Path("Data/babycry_features_v3.csv"))
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--augment", action="store_true", help="Generate augmented variants per clip.")
    parser.add_argument(
        "--teacher-model",
        type=Path,
        help="Path to CRNN TFLite model for distillation logits.",
    )
    parser.add_argument(
        "--teacher-scaler",
        type=Path,
        help="Path to CRNN scaler pickle used alongside the teacher model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} not found.")

    wavs = discover_wavs(dataset_root)
    labeled = list(iter_labeled(wavs))
    if not labeled:
        raise RuntimeError("No labeled WAV files discovered.")

    config = SummaryFeatureConfig(sample_rate=args.sample_rate)
    teacher_ctx = prepare_teacher_context(args.teacher_model, args.teacher_scaler, args.sample_rate)
    teacher_columns = [f"Teacher_{i}" for i in range(9)] if teacher_ctx else []

    rows: List[List[str]] = []
    label_counts: Dict[int, int] = {}

    for index, (wav_path, label_id) in enumerate(labeled, start=1):
        base_samples = load_mono_audio(wav_path, config=config)
        variants = collect_variants(base_samples, config.sample_rate, include_augmentations=args.augment)
        for variant_name, samples in variants:
            identifier = str(wav_path.resolve()) if variant_name == "orig" else f"{wav_path.resolve()}#aug_{variant_name}"
            features = feature_vector_from_audio(samples, config.sample_rate, config=config)
            teacher_probs = teacher_probabilities(teacher_ctx, samples, config.sample_rate)
            rows.append(
                format_row(
                    identifier,
                    label_id,
                    features,
                    variant_name,
                    teacher_probs,
                    teacher_columns,
                )
            )
            label_counts[label_id] = label_counts.get(label_id, 0) + 1
        if index % 25 == 0 or index == len(labeled):
            print(f"[{index}/{len(labeled)}] processed {wav_path.name}")

    write_dataset(rows, args.output.resolve(), teacher_columns)

    print(f"Wrote {len(rows)} samples to {args.output}")
    for code, idx in LABEL_MAP.items():
        count = label_counts.get(idx, 0)
        print(f"  {code} ({idx}): {count}")


if __name__ == "__main__":
    main()
