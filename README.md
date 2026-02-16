# agentic-audio-monitor (Infant Distress Detection)

AI-native audio monitoring backend for infant distress detection and cry-category inference.

## Specification
Build a privacy-conscious, near-real-time system that:
- captures short local audio windows,
- extracts robust acoustic features,
- runs on-device/edge inference,
- returns ranked cry categories with confidence,
- supports downstream automation and alerting.

The core target is robust operation under noisy home environments with minimal latency.

## Agentic Layer
The system is designed as an agentic pipeline rather than a single classifier call.

- Detector Agent: identifies candidate cry segments from continuous audio.
- Classifier Agent: predicts likely distress classes from extracted features.
- Validator Agent: rejects low-confidence or ambient-noise false positives and applies simple policy checks before surfacing a result.

This architecture improves reliability by separating signal detection, classification, and decision validation.

## Technical Stack
- Python
- Librosa / NumPy for audio feature extraction
- TensorFlow/Keras model export and inference tooling
- Android integration scaffolding for mobile capture/inference workflows

## Repository Layout
- `scripts/`: feature extraction, training, scaling, export, and inference scripts
- `android/`: app integration and deployment scaffolding
- `docs/`: implementation notes and planning docs

## Security and Privacy
This public repository is sanitized for open sharing.

- No API keys or tokens are stored in source.
- Environment variables are documented in `.env.example`.
- Raw audio, generated datasets, local machine paths, and model artifacts are excluded via `.gitignore`.

## Development Notes
- This project follows specification-driven development for rapid prototyping in a sensitive-data context.
- For production use, add formal evaluation datasets, calibrated thresholds, and monitoring/rollback controls.

## Disclaimer
This is a technical proof-of-concept and not a medical device.
