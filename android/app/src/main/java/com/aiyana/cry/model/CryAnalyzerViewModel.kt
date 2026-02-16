package com.aiyana.cry.model

import android.app.Application
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aiyana.cry.audio.AudioCaptureManager
import com.aiyana.cry.audio.readWavData
import com.aiyana.cry.ml.CryAnalyzerEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class CryAnalyzerViewModel(application: Application) : AndroidViewModel(application) {

    private val tag = "CryAnalyzer"
    private val app = application
    private val _uiState = MutableStateFlow<CryAnalyzerUiState>(CryAnalyzerUiState.PermissionRequired)
    val uiState = _uiState.asStateFlow()

    private val audioCapture = AudioCaptureManager()
    private val analyzer by lazy { CryAnalyzerEngine(app) }

    private val isListening = AtomicBoolean(false)
    private var startTimestamp: Long = 0L
    private var lastCapturedDuration: Long = 0L

    fun onPermissionUpdated(granted: Boolean) {
        _uiState.value = if (granted) {
            (_uiState.value as? CryAnalyzerUiState.Idle)
                ?.copy(ready = true)
                ?: CryAnalyzerUiState.Idle()
        } else {
            CryAnalyzerUiState.PermissionRequired
        }
    }

    fun startListening() {
        if (!isListening.compareAndSet(false, true)) return
        if (_uiState.value is CryAnalyzerUiState.PermissionRequired) {
            isListening.set(false)
            return
        }
        startTimestamp = SystemClock.elapsedRealtime()
        _uiState.value = CryAnalyzerUiState.Listening(elapsedMillis = 0, averageLevel = 0f)
        Log.d(tag, "startListening()")

        try {
            audioCapture.start { level ->
                if (!isListening.get()) return@start
                val elapsed = SystemClock.elapsedRealtime() - startTimestamp
                _uiState.value = CryAnalyzerUiState.Listening(
                    elapsedMillis = elapsed,
                    averageLevel = level.coerceIn(0f, 1f)
                )
                if (elapsed >= AUTO_STOP_MS) {
                    Log.d(tag, "Auto-stop triggered at $elapsed ms")
                    stopListening(autoTriggered = true)
                }
            }
        } catch (t: Throwable) {
            isListening.set(false)
            _uiState.value = CryAnalyzerUiState.Error("Recorder error: ${t.localizedMessage ?: "unknown"}")
        }
    }

    @Suppress("UNUSED_PARAMETER")
    fun stopListening(autoTriggered: Boolean = false) {
        if (!isListening.compareAndSet(true, false)) return

        viewModelScope.launch {
            val samples = audioCapture.stop()
            val elapsed = SystemClock.elapsedRealtime() - startTimestamp
            lastCapturedDuration = max(elapsed, 1_000L)
            Log.d(tag, "Captured ${samples.size} samples in ${elapsed}ms (rms=${audioCapture.rmsLevel})")
            if (samples.isEmpty()) {
                _uiState.value = CryAnalyzerUiState.Error("No audio captured. Check microphone input.")
                return@launch
            }
            _uiState.value = CryAnalyzerUiState.Processing

            val engineResult = analyzer.analyze(
                monoPcmFloat = samples,
                sampleRateHz = audioCapture.sampleRateHz
            )

            val predictions = engineResult.toModelPredictions()
            Log.d(tag, "Inference produced ${predictions.size} predictions (raw=${engineResult.rawAudio != null})")
            if (predictions.isEmpty()) {
                _uiState.value = CryAnalyzerUiState.Error("Unable to infer cry type. Please retry.")
            } else {
                _uiState.value = CryAnalyzerUiState.Completed(
                    predictions = predictions,
                    capturedDurationMillis = lastCapturedDuration
                )
            }
        }
    }

    fun reset() {
        isListening.set(false)
        val currentSample = (_uiState.value as? CryAnalyzerUiState.Idle)?.selectedSample
        _uiState.value = CryAnalyzerUiState.Idle(selectedSample = currentSample)
        Log.d(tag, "UI reset to idle")
    }

    fun selectSample(sampleIndex: Int?) {
        val currentState = _uiState.value
        if (currentState is CryAnalyzerUiState.Idle) {
            _uiState.value = currentState.copy(selectedSample = sampleIndex)
        } else {
            _uiState.value = CryAnalyzerUiState.Idle(selectedSample = sampleIndex)
        }
    }

    fun analyzeImportedAudio(uri: Uri) {
        viewModelScope.launch {
            _uiState.value = CryAnalyzerUiState.Processing
            try {
                val wav = withContext(Dispatchers.IO) {
                    app.contentResolver.readWavData(uri)
                }
                Log.d(tag, "Imported WAV ${wav.samples.size} samples @${wav.sampleRate}Hz (${wav.durationMillis}ms)")
                val engineResult = analyzer.analyze(
                    monoPcmFloat = wav.samples,
                    sampleRateHz = wav.sampleRate
                )
                val predictions = engineResult.toModelPredictions()
                Log.d(tag, "Imported inference produced ${predictions.size} predictions")
                if (predictions.isEmpty()) {
                    _uiState.value = CryAnalyzerUiState.Error("Unable to infer cry type from file.")
                } else {
                    _uiState.value = CryAnalyzerUiState.Completed(
                        predictions = predictions,
                        capturedDurationMillis = max(wav.durationMillis, 1_000L)
                    )
                }
            } catch (t: Throwable) {
                _uiState.value = CryAnalyzerUiState.Error(
                    "Failed to analyze recording: ${t.localizedMessage ?: "unknown error"}"
                )
            }
        }
    }

    fun analyzeEmbeddedSample(sampleIndex: Int) {
        viewModelScope.launch {
            if (sampleIndex !in CrySamples.samples.indices) {
                _uiState.value = CryAnalyzerUiState.Error("Sample unavailable.")
                return@launch
            }
            _uiState.value = CryAnalyzerUiState.Processing
            try {
                val resId = CrySamples.samples[sampleIndex].resId
                val wav = withContext(Dispatchers.IO) {
                    app.resources.openRawResource(resId).use { stream ->
                        stream.readWavData()
                    }
                }
                val engineResult = analyzer.analyze(
                    monoPcmFloat = wav.samples,
                    sampleRateHz = wav.sampleRate
                )
                val predictions = engineResult.toModelPredictions()
                if (predictions.isEmpty()) {
                    _uiState.value = CryAnalyzerUiState.Error("Unable to infer cry type from sample.")
                } else {
                    _uiState.value = CryAnalyzerUiState.Completed(
                        predictions = predictions,
                        capturedDurationMillis = max(wav.durationMillis, 1_000L)
                    )
                }
            } catch (t: Throwable) {
                _uiState.value = CryAnalyzerUiState.Error("Failed to analyze sample: ${t.localizedMessage ?: "unknown error"}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            audioCapture.shutdown()
            analyzer.close()
        }
    }

    private fun CryAnalyzerEngine.EngineResult.toModelPredictions(): List<ModelPrediction> {
        val outputs = buildList {
            rawAudio?.let { add(it.toPrediction(ModelId.RawAudio)) }
            feature?.let { add(it.toPrediction(ModelId.Feature)) }
        }
        return outputs
    }

    private fun CryAnalyzerEngine.EngineResult.ModelOutput.toPrediction(
        modelId: ModelId
    ): ModelPrediction {
        val probs = probabilities.mapIndexed { index, value ->
            ClassProbability(
                label = formatLabel(labels.getOrNull(index) ?: "label_$index"),
                probability = value.coerceIn(0f, 1f)
            )
        }.sortedByDescending { it.probability }
        val top = probs.firstOrNull() ?: ClassProbability("unknown", 0f)
        return ModelPrediction(
            modelId = modelId,
            topLabel = top.label,
            confidence = top.probability,
            probabilities = probs
        )
    }

    private fun formatLabel(label: String): String =
        label.replace('_', ' ').replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }

    companion object {
        private const val AUTO_STOP_MS = 20_000L
    }
}

