"""
Export sklearn StandardScaler parameters to JSON for mobile inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaler",
        type=Path,
        required=True,
        help="Path to the pickled StandardScaler.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file. Defaults to <scaler>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaler_path = args.scaler.resolve()
    output_path = args.output.resolve() if args.output else scaler_path.with_suffix(".json")

    scaler = joblib.load(scaler_path)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"{scaler_path} is not a sklearn.preprocessing.StandardScaler.")

    data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    output_path.write_text(json.dumps(data))
    print(f"Exported scaler parameters to {output_path}")


if __name__ == "__main__":
    main()


