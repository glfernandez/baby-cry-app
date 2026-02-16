# Baby Cry App

Agentic audio prototype for infant distress signal classification, built as a privacy-first engineering project.

## Project Scope
- Detect and classify short cry segments into likely intent labels.
- Run an explicit multi-stage decision flow (detector -> classifier -> validator).
- Support model training/export workflows and Android app integration for on-device testing.

## Label Set
- `hungry`
- `needs_burping`
- `belly_pain`
- `discomfort`
- `tired`
- `lonely`
- `cold_hot`
- `scared`
- `dirty_diaper`

## Agentic Decision Flow
1. `Detector`: selects candidate audio windows.
2. `Classifier`: extracts features and predicts class probabilities.
3. `Validator`: applies confidence/top-k checks before surfacing final output.

This architecture is designed to reduce false positives in non-stationary home audio.

## Repository Structure
- `scripts/`: dataset generation, training, inference, scaler/model export.
- `android/`: Android app module for mobile testing.
- `docs/QUICKSTART.md`: local setup and command examples.
- `docs/SECURITY_CHECKLIST.md`: pre-push public-safety checklist.
- `.env.example`: environment variable template.

## Quickstart
Use the full guide:
- `docs/QUICKSTART.md`

Core training/inference path:
```bash
python scripts/generate_feature_dataset.py --dataset-root /path/to/wavs --output /path/to/features.csv --augment
python scripts/train_features.py --dataset /path/to/features.csv --output-dir /path/to/model_out --epochs 80
python scripts/run_inference_features.py --model /path/to/model_out/feature_model.keras --scaler /path/to/model_out/feature_scaler.pkl /path/to/test_audio.wav
```

## Android Build
```bash
cd android
./gradlew assembleDebug
```

## Public Security Posture
- No raw family audio or private datasets are committed.
- No hardcoded API credentials should be committed.
- `.gitignore` blocks local artifacts and sensitive runtime files.

## Preview
![Preview 1](docs/images/preview-01.jpeg)
![Preview 2](docs/images/preview-02.jpeg)

## Disclaimer
This is a technical prototype for engineering demonstration. It is not a medical device.
