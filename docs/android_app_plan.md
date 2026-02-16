## Pixel 8 Cry Analyzer – Architecture Sketch

### Goals
- Capture a cry clip directly on-device, process it locally, and display results from both models (CRNN and feature-dense) with confidence bars and per-class probabilities.
- Preserve privacy: no network calls, optional local history only.
- Provide waveform preview and quick share/export of the prediction summary if needed.

### Functional Flow
1. **Record / Import**
   - Built-in recorder with start/stop controls (16 kHz mono, WAV).
   - Optionally pick an existing `.wav`/`.m4a` clip from storage; convert to WAV internally.
   - Display waveform preview (MPAndroidChart or custom Canvas).
2. **Pre-processing**
   - CRNN branch:
     - Convert stereo if needed, resample to 44.1 kHz.
     - Generate log-mel features (2 × 381 × 40).
     - Apply StandardScaler (`sanity_scaler.json`), reshape to `(1, 2, 381, 40)`.
   - Feature branch:
     - Compute donateacry summary statistics (Amplitude envelope mean, RMS, ZCR, MFCC stats, etc.).
     - Apply scaler (`feature_scaler.json`) to produce 25-d vector.
3. **Inference**
   - CRNN: run `babycry_sanity.tflite` via TensorFlow Lite interpreter with Flex delegate.
   - Feature model: run `feature_model.tflite`.
   - Collect softmax outputs, max class per model, confidence, and class rankings.
4. **Presentation**
   - Unified result screen showing:
     - Waveform thumbnail.
     - Model A (CRNN): top label + confidence + horizontal bar chart for all 9 classes.
     - Model B (Feature): same layout for 5 classes.
     - Optional tabular comparison or “agreement/disagreement” badge.
   - Save entry (path, timestamp, predictions) to encrypted Room DB if user enables history.

### Implementation Notes
- **Project structure**
  - `app/` (Kotlin, Jetpack Compose or XML + ViewModel).
  - `ml/` assets folder containing `babycry_sanity.tflite`, `feature_model.tflite`, scaler JSONs, label maps, and donateacry NPZ weights if post-processing needed.
  - `native/` packaging for `libtensorflowlite_flex.so` (Flex delegate).
- **ML Utilities**
  - Kotlin `AudioFeatureExtractor` replicating Python logic (FFT, mel filters, MFCC, summary stats).
  - `Scaler` utility that applies mean/scale from JSON.
  - `CryAnalyzer` service orchestrating both models on a background coroutine.
- **UI**
- `ListenScreen` (single floating mic button + pulse animation) → `ProcessingScreen` (progress ring) → `ResultsScreen`.
- Material 3 components with high contrast text, color-coded confidence bars, and haptic feedback for key actions.
- Settings toggle for history + storage permissions.
- Surface optional waveform toggle (off by default) to keep first-run UX minimal.

### UX & Interaction Details
- **Quick capture**: primary action is a large circular mic button centered on screen; tap starts a rolling 6 s capture buffer (Shazam-style) that auto-stops when confidence stabilizes or 8 s elapses.
- **Live status**: show animated concentric waves and “Listening…” copy; allow swipe down to cancel.
- **Processing transition**: upon capture completion, lock UI, show compact waveform thumbnail and spinner (< 400 ms target).
- **Result presentation**:
  - Stack two cards (“Raw Audio Model”, “Feature Model”) with top label + probability, secondary line showing agreement badge.
  - Include expandable “Other possibilities” list sorted by confidence descending, limited to top 4 for readability.
  - Provide single “New recording” button at bottom; history icon in app bar if feature enabled.
- **Accessibility**: announce transitions via TalkBack strings, ensure color bars also include numeric percentages.

### Performance Considerations
- Pre-allocate audio buffer for rolling capture to avoid allocation spikes.
- Warm up both interpreters at app start (lazy background coroutine) to minimize first inference latency.
- Aim for end-to-end capture + inference under 2.5 s on Pixel 8 (CPU path), with burst mode disabled during processing.
- Current build implements the CRNN raw-audio inference path with Kotlin DSP + TFLite; feature-vector pipeline remains TODO.

### Pending Engineering Tasks
- Implement Kotlin DSP utilities mirroring Python feature specs with unit parity tests.
- Integrate Flex delegate (`libtensorflowlite_flex.so`) packaging and runtime checks.
- Wire up Compose navigation skeleton with coroutine-based recording service and ViewModel state machine.
- Add instrumentation tests that pump canned WAV fixtures through the full stack to confirm UI + inference wiring.
- Add Storage Access Framework import flow to analyze existing recordings alongside live capture.

### Testing & Verification
- Include comparison harness using `scripts/check_parity.py` outputs to validate mobile results during QA.
- Instrumented tests with canned WAV fixtures to ensure deterministic predictions.
- Monitor runtime (goal < 1 s per clip) and memory (< 32 MB extra).

### Deployment
- Build flavors: `internalRelease` (sideload APK signed with local key).
- Documentation with steps to install via `adb` or Pixel file manager.
- Optional crash/analytics excluded by default to maintain privacy.


