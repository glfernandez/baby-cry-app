# Baby Cry App

A public proof-of-concept for infant distress detection using an agentic audio pipeline.

## What This Project Does
The system processes short audio windows and predicts likely cry intent classes.

Current class map:
- hungry
- needs_burping
- belly_pain
- discomfort
- tired
- lonely
- cold_hot
- scared
- dirty_diaper

## Agentic Pipeline (Practical)
The implementation is structured as three decision stages:
- Detector stage: selects candidate audio segments.
- Classifier stage: runs feature extraction + model inference.
- Validator stage: checks confidence/top-k behavior before result usage.

This separation is meant to reduce false alerts in noisy environments.

## Repository Structure
- `scripts/`: feature extraction, dataset generation, training, export, inference.
- `docs/`: quickstart and public security checklist.
- `.env.example`: environment variable template.

## Key Scripts
- `scripts/generate_feature_dataset.py`: builds summary-feature CSV from labeled WAV files.
- `scripts/train_features.py`: trains the dense classifier and saves artifacts.
- `scripts/run_inference_features.py`: runs inference on WAV or feature CSV input.
- `scripts/export_tflite.py`: exports trained model to TFLite.
- `scripts/fit_feature_scaler.py`, `scripts/prepare_scaler.py`, `scripts/export_scaler.py`: scaler workflows.

## Quickstart
See `docs/QUICKSTART.md` for setup and command examples.

## Disclaimer
This is a technical prototype and not a medical device.
