"""
Compute and persist the feature scaler used by the babycry inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.audio_features import FeatureSpec, extract_feature_matrix
else:
    from .audio_features import FeatureSpec, extract_feature_matrix


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.wav")):
        if path.is_file():
            yield path


def accumulate_statistics(
    dataset_root: Path,
    spec: FeatureSpec,
) -> Tuple[StandardScaler, int]:
    scaler = StandardScaler()
    file_count = 0

    for wav_path in iter_audio_files(dataset_root):
        features, _ = extract_feature_matrix(wav_path, spec)
        scaler.partial_fit(features)
        file_count += 1

    if file_count == 0:
        raise RuntimeError(f"No WAV files found under {dataset_root}")

    return scaler, file_count


def save_metadata(metadata_path: Path, *, file_count: int, spec: FeatureSpec) -> None:
    metadata = {
        "files": file_count,
        "sample_rate": spec.sample_rate,
        "n_fft": spec.n_fft,
        "hop_length": spec.hop_length,
        "n_mels": spec.n_mels,
        "target_frames": spec.target_frames,
        "enforce_stereo": spec.enforce_stereo,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../Data/WAV_Audio_files"),
        help="Folder containing the curated babycry WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../models"),
        help="Directory where the scaler and metadata will be stored.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Do not duplicate mono signals; keep a single channel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    spec = FeatureSpec(enforce_stereo=not args.mono)
    scaler, file_count = accumulate_statistics(dataset_root, spec)

    scaler_path = output_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    save_metadata(output_dir / "feature_scaler.json", file_count=file_count, spec=spec)

    print(f"Saved scaler to {scaler_path}")
    print(f"Processed {file_count} audio files from {dataset_root}")


if __name__ == "__main__":
    main()
