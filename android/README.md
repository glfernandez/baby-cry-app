# Pixel Cry Analyzer – Android Client

This module contains the private APK project that will run both baby-cry models locally on a Pixel 8 device. The UI emphasises a Shazam-like one-tap capture flow, presenting the raw-audio CRNN and feature model predictions with confidence bars and alternative probabilities.

## Project Layout

- `app/` – Jetpack Compose UI, microphone capture plumbing, and TensorFlow Lite inference scaffolding.
- `app/src/main/assets/` – place model assets here:
  - `babycry_sanity.tflite`
  - `sanity_scaler.json`
  - `feature_model.tflite`
  - `feature_scaler.json`
- `app/src/main/java/com/aiyana/cry/ml/` – on-device feature extraction + dual-model interpreter (currently stubbed; will mirror the Python parity pipeline).
- `app/src/main/java/com/aiyana/cry/audio/` – rolling audio buffer & capture utilities (to be implemented next pass).

## Getting Started

1. Install the Android SDK (API 34) and Java 17 (Android Studio Hedgehog ships both).
2. From this directory, generate a Gradle wrapper: `./gradlew` is not checked in because the host machine lacks Gradle. Run `gradle wrapper --gradle-version 8.6` once.
3. Open the project in Android Studio, select the `internalRelease` build variant for sideloadable builds.
4. Copy the four model/scaler artifacts listed above into `app/src/main/assets`.
5. Build & run on a Pixel 8 device (Android 14). The app requests microphone + media read permissions on first launch.

## Next Steps

- Implement the audio capture ring buffer for 6–8 s snippets with auto-stop when RMS stabilises. ✅ (uses `AudioCaptureManager`)
- Port the Python DSP feature extraction code to Kotlin so both TFLite models receive parity inputs. (raw CRNN path complete; feature-vector pipeline TBD)
- Replace the placeholder inference stub with real TensorFlow Lite interpreters and surface the top results in the UI. ✅ (raw model wired via `CryAnalyzerEngine`)
- Wire up optional encrypted local history and on-device parity smoke tests using sample WAV fixtures.

Refer to `docs/android_app_plan.md` for the full architecture, UX guidelines, and performance targets.

