"""
Run the summary-feature classifier on WAV files or precomputed CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from audio_summary_features import (  # type: ignore  # pylint: disable=import-error
        SUMMARY_COLUMNS,
        SummaryFeatureConfig,
        extract_summary_features,
    )
else:
    from .audio_summary_features import SUMMARY_COLUMNS, SummaryFeatureConfig, extract_summary_features

DEFAULT_LABEL_MAP: dict[int, str] = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained Keras feature model.",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        required=True,
        help="Pickled StandardScaler fitted on the regenerated dataset.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional JSON label map (index -> class name).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV of precomputed features. When omitted, WAV inputs are required.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44_100,
        help="Resample WAVs to this rate before feature extraction.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="WAV files or directories to analyse when --csv is not supplied.",
    )
    return parser.parse_args()


def load_model(path: Path) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def load_scaler(path: Path) -> StandardScaler:
    scaler = joblib.load(path)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"{path} is not a sklearn.preprocessing.StandardScaler instance.")
    return scaler


def load_label_map(path: Path | None) -> dict[int, str]:
    if path is None:
        return DEFAULT_LABEL_MAP
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}


def expand_sources(sources: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for src in sources:
        src = src.resolve()
        if src.is_dir():
            expanded.extend(sorted(src.rglob("*.wav")))
        else:
            expanded.append(src)
    return expanded


def features_from_wavs(paths: Sequence[Path], config: SummaryFeatureConfig) -> tuple[np.ndarray, list[str]]:
    feature_rows: list[np.ndarray] = []
    names: list[str] = []
    for path in paths:
        if not path.exists():
            print(f"[warn] missing file {path}, skipping")
            continue
        vector = extract_summary_features(path, config=config)
        feature_rows.append(vector)
        names.append(path.name)
    if not feature_rows:
        raise RuntimeError("No valid WAV files processed.")
    return np.stack(feature_rows, axis=0), names


def features_from_csv(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    for column in SUMMARY_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"CSV missing required column {column}")
    features = df[list(SUMMARY_COLUMNS)].to_numpy(dtype=np.float32)
    names = (
        df["Cry_Audio_File"].astype(str).tolist()
        if "Cry_Audio_File" in df.columns
        else [f"row_{i}" for i in range(len(df))]
    )
    return features, names


def main() -> None:
    args = parse_args()

    model = load_model(args.model.resolve())
    scaler = load_scaler(args.scaler.resolve())
    label_map = load_label_map(args.labels.resolve() if args.labels else None)

    if args.csv:
        features, names = features_from_csv(args.csv.resolve())
    else:
        sources = expand_sources(args.sources)
        if not sources:
            raise RuntimeError("No WAV files supplied.")
        config = SummaryFeatureConfig(sample_rate=args.sample_rate)
        features, names = features_from_wavs(sources, config)

    scaled = scaler.transform(features)
    predictions = model.predict(scaled, verbose=0)

    for name, probs in zip(names, predictions, strict=False):
        label_idx = int(np.argmax(probs))
        label = label_map.get(label_idx, f"class_{label_idx}")
        sorted_indices = np.argsort(probs)[::-1]
        top_entries = [f"{label_map.get(int(idx), idx)}={probs[int(idx)]:.2f}" for idx in sorted_indices[:5]]
        print(f"{name}: {label} ({probs[label_idx]:.2f})")
        print(f"  top probabilities: {', '.join(top_entries)}")


if __name__ == "__main__":
    main()


