# Quickstart

## 1. Environment
Use Python 3.10+.

Create and activate a virtual environment, then install dependencies used by scripts in this repo.

## 2. Prepare Data (Local Only)
Keep raw WAV files outside git-tracked paths or under ignored folders.

Expected label format in filename (suffix before extension):
- `*-hu.wav`, `*-bu.wav`, `*-bp.wav`, `*-dc.wav`, `*-ti.wav`, `*-lo.wav`, `*-ch.wav`, `*-sc.wav`, `*-dk.wav`

## 3. Generate Feature Dataset
Example:
```bash
python scripts/generate_feature_dataset.py \
  --dataset-root /path/to/local/wavs \
  --output /path/to/local/features.csv \
  --augment
```

## 4. Train Classifier
Example:
```bash
python scripts/train_features.py \
  --dataset /path/to/local/features.csv \
  --output-dir /path/to/local/model_out \
  --epochs 80
```

## 5. Run Inference
Example:
```bash
python scripts/run_inference_features.py \
  --model /path/to/local/model_out/feature_model.keras \
  --scaler /path/to/local/model_out/feature_scaler.pkl \
  /path/to/test_audio.wav
```

## 6. Optional TFLite Export
```bash
python scripts/export_tflite.py
```

## Notes
- Paths above are local examples only.
- Keep datasets and model binaries out of public commits unless intentionally open-sourced.
