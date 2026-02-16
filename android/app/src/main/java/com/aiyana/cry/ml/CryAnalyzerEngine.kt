package com.aiyana.cry.ml

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import java.io.BufferedReader
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.jtransforms.fft.DoubleFFT_1D
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate

class CryAnalyzerEngine(
    private val appContext: Context
) : AutoCloseable {

    private val assets = appContext.assets

    private val rawLabels = listOf(
        "hungry",
        "needs burping",
        "belly pain",
        "discomfort",
        "tired",
        "lonely",
        "cold or hot",
        "scared",
        "unknown"
    )

    private val featureLabels: List<String> by lazy { loadFeatureLabels() }

    private val rawScaler by lazy { loadScaler("sanity_scaler.json") }
    private val featureScaler by lazy { loadScaler("feature_scaler.json") }

    private val featureExtractor by lazy { AudioFeatureExtraction() }

    private val rawInterpreter: Interpreter? by lazy {
        try {
            createRawInterpreter()
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to initialize raw interpreter", t)
            null
        }
    }
    private val featureInterpreter: Interpreter? by lazy {
        try {
            createFeatureInterpreter()
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to initialize feature interpreter", t)
            null
        }
    }

    private val melFilterBank by lazy { buildMelFilterBank(RAW_FFT_SIZE, RAW_MEL_BINS, RAW_SAMPLE_RATE) }
    private val hannWindow by lazy { buildHannWindow(RAW_FFT_SIZE) }
    private val fft by lazy { DoubleFFT_1D(RAW_FFT_SIZE.toLong()) }

    suspend fun analyze(
        monoPcmFloat: FloatArray,
        sampleRateHz: Int
    ): EngineResult = withContext(Dispatchers.Default) {
        Log.d(TAG, "Analyze request samples=${monoPcmFloat.size} rate=$sampleRateHz")
        if (monoPcmFloat.isEmpty()) {
            return@withContext EngineResult.EMPTY
        }

        // Don't normalize before raw audio preprocessing - librosa.load() normalizes by dtype max,
        // not signal max, so we should preserve the original signal amplitude
        val resampled = if (sampleRateHz == RAW_SAMPLE_RATE) monoPcmFloat.copyOf() else resampleLinear(monoPcmFloat, sampleRateHz, RAW_SAMPLE_RATE)
        Log.d(TAG, "Resampled length=${resampled.size}")
        if (resampled.isEmpty()) {
            return@withContext EngineResult.EMPTY
        }
        if (rawInterpreter == null) {
            Log.e(TAG, "Raw interpreter is not available; ensure model asset is packaged.")
        }
        val rawInput = prepareRawAudioInput(resampled)
        val rawOutput = runRawModel(rawInput)
        if (rawOutput == null) {
            Log.w(TAG, "Raw model returned null output")
        }

        val featureOutput = runFeatureBranch(resampled)

        EngineResult(rawAudio = rawOutput, feature = featureOutput)
    }

    override fun close() {
        try {
            rawInterpreter?.close()
        } catch (_: Throwable) {
        }
        try {
            featureInterpreter?.close()
        } catch (_: Throwable) {
        }
    }

    private fun runRawModel(input: Array<Array<Array<FloatArray>>>): EngineResult.ModelOutput? {
        return try {
            val output = Array(1) { FloatArray(rawLabels.size) }
            val interpreter = rawInterpreter ?: run {
                Log.e(TAG, "rawInterpreter instance is null during inference")
                return null
            }
            interpreter.run(input, output)
            val probs = output[0]
            Log.d(TAG, "Raw logits=" + probs.joinToString(prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
            Log.d(TAG, "Raw model probs=" + formatProbabilities(rawLabels, probs))
            EngineResult.ModelOutput(
                logits = probs.copyOf(),
                probabilities = probs.copyOf(),
                labels = rawLabels
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Raw model inference failed", t)
            null
        }
    }

    private fun runFeatureBranch(resampled: FloatArray): EngineResult.ModelOutput? {
        val interpreter = featureInterpreter ?: run {
            Log.w(TAG, "Feature interpreter not available; skipping feature model.")
            return null
        }
        val featureVector = featureExtractor.extractFeatureVector(resampled, RAW_SAMPLE_RATE)
        if (featureVector.isEmpty()) {
            Log.w(TAG, "Feature extractor returned empty vector.")
            return null
        }
        Log.d(TAG, "Feature vector length=${featureVector.size}")
        Log.d(TAG, "Feature vector sample=" + featureVector.joinToString(limit = 25, prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
        val scaled = FloatArray(featureVector.size) { index ->
            featureScaler.normalize(featureVector[index], index)
        }
        val mean = scaled.average()
        val variance = scaled.map { (it - mean) * (it - mean) }.average()
        val std = kotlin.math.sqrt(variance)
        Log.d(TAG, "Feature scaled stats: mean=${String.format(Locale.US, "%.4f", mean)} std=${String.format(Locale.US, "%.4f", std)} min=${String.format(Locale.US, "%.4f", scaled.minOrNull() ?: 0f)} max=${String.format(Locale.US, "%.4f", scaled.maxOrNull() ?: 0f)} length=${scaled.size}")
        val input = arrayOf(scaled)
        val output = Array(1) { FloatArray(featureLabels.size) }
        return try {
            interpreter.run(input, output)
            val probs = output[0]
            val labelsToUse = if (featureLabels.isNotEmpty()) featureLabels else rawLabels
            Log.d(TAG, "Feature logits=" + probs.joinToString(prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
            Log.d(TAG, "Feature model probs=" + formatProbabilities(labelsToUse, probs))
            EngineResult.ModelOutput(
                logits = probs.copyOf(),
                probabilities = probs.copyOf(),
                labels = labelsToUse
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Feature model inference failed", t)
            null
        }
    }

    private fun prepareRawAudioInput(samples: FloatArray): Array<Array<Array<FloatArray>>> {
        val stereo = arrayOf(samples, samples) // enforce stereo by duplication
        val featuresPerChannel = ArrayList<FloatArray>(RAW_CHANNELS)
        for (channel in 0 until stereo.size) {
            val mel = computeLogMelSpectrogram(stereo[channel])
            val flattened = FloatArray(RAW_TARGET_FRAMES * RAW_MEL_BINS)
            var index = 0
            for (frame in mel) {
                for (value in frame) {
                    flattened[index++] = value
                }
            }
            featuresPerChannel.add(flattened)
        }

        val fetStats = featuresPerChannel.firstOrNull()
        if (fetStats != null) {
            val slice = fetStats.take(40)
            val meanVal = slice.average()
            val stdVal = kotlin.math.sqrt(slice.map { val diff = it - meanVal; diff * diff }.average())
            Log.d(TAG, "Raw mel sample=" + slice.joinToString(prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
            Log.d(TAG, "Raw mel stats: mean=${String.format(Locale.US, "%.4f", meanVal)} std=${String.format(Locale.US, "%.4f", stdVal)} length=${fetStats.size}")
        }

        val batch = Array(1) { Array(RAW_CHANNELS) { Array(RAW_TARGET_FRAMES) { FloatArray(RAW_MEL_BINS) } } }

        for (channel in 0 until RAW_CHANNELS) {
            val channelFeatures = featuresPerChannel[channel]
            for (frame in 0 until RAW_TARGET_FRAMES) {
                for (mel in 0 until RAW_MEL_BINS) {
                    val flattenedIndex = frame * RAW_MEL_BINS + mel
                    val rawValue = if (flattenedIndex < channelFeatures.size) channelFeatures[flattenedIndex] else 0f
                    val featureIndex = channel * RAW_MEL_BINS + mel
                    val normalized = rawScaler.normalize(rawValue, featureIndex)
                    batch[0][channel][frame][mel] = normalized
                }
            }
        }
        val sampleSlice = batch[0][0][0]
        Log.d(TAG, "Raw feature length=${sampleSlice.size}")
        Log.d(TAG, "Raw feature scaled sample=\n" + sampleSlice.joinToString(limit = 40, prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
        val stats = sampleSlice.filter { !it.isNaN() && !it.isInfinite() }
        if (stats.isNotEmpty()) {
            val mean = stats.average()
            val variance = stats.map { val diff = it - mean; diff * diff }.average()
            val std = kotlin.math.sqrt(variance)
            Log.d(TAG, "Raw feature stats: mean=${String.format(Locale.US, "%.4f", mean)} std=${String.format(Locale.US, "%.4f", std)} min=${String.format(Locale.US, "%.4f", stats.minOrNull() ?: 0.0)} max=${String.format(Locale.US, "%.4f", stats.maxOrNull() ?: 0.0)} length=${stats.size}")
        }
        return batch
    }

    private fun computeLogMelSpectrogram(channelSamples: FloatArray): List<FloatArray> {
        val padding = RAW_FFT_SIZE / 2
        val padded = padWithZeros(channelSamples, padding)
        val frameCount = max(
            1,
            ((padded.size - RAW_FFT_SIZE) / RAW_HOP_LENGTH) + 1
        )
        val results = ArrayList<FloatArray>(min(frameCount, RAW_TARGET_FRAMES))
        val frameBuffer = DoubleArray(RAW_FFT_SIZE)
        var frameIndex = 0
        var offset = 0

        while (offset + RAW_FFT_SIZE <= padded.size && frameIndex < RAW_TARGET_FRAMES) {
            for (i in 0 until RAW_FFT_SIZE) {
                frameBuffer[i] = padded[offset + i].toDouble() * hannWindow[i]
            }
            fft.realForward(frameBuffer)
            val magnitude = FloatArray(RAW_FFT_SIZE / 2 + 1)
            magnitude[0] = abs(frameBuffer[0]).toFloat()
            magnitude[magnitude.lastIndex] = abs(frameBuffer[1]).toFloat()
            var bin = 1
            var bufferIndex = 2
            while (bin < magnitude.lastIndex) {
                val realPart = frameBuffer[bufferIndex]
                val imagPart = frameBuffer[bufferIndex + 1]
                magnitude[bin] = sqrt(realPart * realPart + imagPart * imagPart).toFloat()
                bin += 1
                bufferIndex += 2
            }
            val melEnergies = FloatArray(RAW_MEL_BINS)
            for (m in 0 until RAW_MEL_BINS) {
                val filter = melFilterBank[m]
                var sum = 0f
                for (k in magnitude.indices) {
                    sum += filter[k] * magnitude[k]
                }
                melEnergies[m] = ln(max(sum, 1e-10f))
            }
            results.add(melEnergies)
            frameIndex++
            offset += RAW_HOP_LENGTH
        }

        while (results.size < RAW_TARGET_FRAMES) {
            results.add(FloatArray(RAW_MEL_BINS))
        }

        return results
    }

    private fun normalizeInPlace(buffer: FloatArray) {
        var maxAbs = 1e-6f
        for (sample in buffer) {
            maxAbs = max(maxAbs, abs(sample))
        }
        if (maxAbs <= 0f) return
        val inv = 1f / maxAbs
        for (i in buffer.indices) {
            buffer[i] = (buffer[i] * inv).coerceIn(-1f, 1f)
        }
    }

    private fun loadFeatureLabels(): List<String> = try {
        val json = assets.open("label_map.json").use { stream ->
            stream.bufferedReader().readText()
        }
        val obj = JSONObject(json)
        obj.keys().asSequence()
            .sortedBy { it.toIntOrNull() ?: Int.MAX_VALUE }
            .mapNotNull { key -> obj.optString(key, null) }
            .toList()
    } catch (_: Throwable) {
        emptyList()
    }

    private fun loadScaler(fileName: String): StandardScalerParams = try {
        val json = assets.open(fileName).use { stream ->
            stream.bufferedReader().use(BufferedReader::readText)
        }
        val obj = JSONObject(json)
        val meanArray = obj.getJSONArray("mean")
        val scaleArray = obj.getJSONArray("scale")
        val mean = FloatArray(meanArray.length()) { index -> meanArray.getDouble(index).toFloat() }
        val scale = FloatArray(scaleArray.length()) { index -> scaleArray.getDouble(index).toFloat() }
        Log.d(TAG, "Loaded scaler mean sample=" + mean.take(5).joinToString(prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
        Log.d(TAG, "Loaded scaler scale sample=" + scale.take(5).joinToString(prefix = "[", postfix = "]") { String.format(Locale.US, "%.4f", it) })
        StandardScalerParams(mean, scale)
    } catch (_: Throwable) {
        StandardScalerParams(FloatArray(1) { 0f }, FloatArray(1) { 1f })
    }

    private fun createRawInterpreter(): Interpreter {
        val options = Interpreter.Options().apply {
            addDelegate(FlexDelegate())
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        return Interpreter(loadModelFile("babycry_sanity.tflite"), options)
    }

    private fun createFeatureInterpreter(): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        return Interpreter(loadModelFile("feature_model.tflite"), options)
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val afd: AssetFileDescriptor = assets.openFd(fileName)
        return try {
            FileInputStream(afd.fileDescriptor).use { input ->
                input.channel.use { channel ->
                    val mapped = channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
                    mapped.order(ByteOrder.nativeOrder())
                }
            }
        } finally {
            afd.close()
        }
    }

    private fun padWithZeros(samples: FloatArray, padding: Int): FloatArray {
        val padded = FloatArray(samples.size + padding * 2)
        samples.copyInto(padded, destinationOffset = padding)
        return padded
    }

    private fun buildHannWindow(size: Int): FloatArray {
        val window = FloatArray(size)
        if (size <= 1) return window
        val denom = (size - 1).toDouble()
        for (i in 0 until size) {
            window[i] = (0.5f - 0.5f * cos((2.0 * PI * i) / denom)).toFloat()
        }
        return window
    }

    private fun buildMelFilterBank(fftSize: Int, melBins: Int, sampleRate: Int): Array<FloatArray> {
        val numFftBins = fftSize / 2 + 1
        val melLow = hzToMel(0.0)
        val melHigh = hzToMel(sampleRate / 2.0)
        val melPoints = DoubleArray(melBins + 2) { i ->
            melLow + (melHigh - melLow) * i / (melBins + 1)
        }
        val hzPoints = melPoints.map(::melToHz)
        val binIndices = hzPoints.map { hz ->
            kotlin.math.floor((fftSize + 1) * hz / sampleRate).toInt().coerceIn(0, numFftBins - 1)
        }

        val filters = Array(melBins) { FloatArray(numFftBins) }
        for (m in 1..melBins) {
            val left = binIndices[m - 1]
            val center = binIndices[m]
            val right = binIndices[m + 1]
            if (center == left || right == center) continue
            for (k in left until center) {
                filters[m - 1][k] = ((k - left).toFloat() / (center - left).toFloat()).coerceAtLeast(0f)
            }
            for (k in center until right) {
                filters[m - 1][k] = ((right - k).toFloat() / (right - center).toFloat()).coerceAtLeast(0f)
            }
        }
        return filters
    }

    private fun hzToMel(hz: Double): Double =
        2595.0 * kotlin.math.log10(1.0 + hz / 700.0)

    private fun melToHz(mel: Double): Double =
        700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun formatProbabilities(labels: List<String>, probs: FloatArray): String {
        return labels.indices.joinToString { index ->
            val label = labels[index]
            val value = probs.getOrElse(index) { 0f }
            "$label=${String.format(Locale.US, "%.2f", value)}"
        }
    }

    private fun resampleLinear(
        source: FloatArray,
        sourceRate: Int,
        targetRate: Int
    ): FloatArray {
        if (sourceRate == targetRate || source.isEmpty()) return source
        val durationSeconds = source.size.toDouble() / sourceRate
        val targetLength = (durationSeconds * targetRate).roundToInt()
        if (targetLength <= 1) return source
        val result = FloatArray(targetLength)
        val step = (source.size - 1).toDouble() / (targetLength - 1)
        for (i in 0 until targetLength) {
            val position = i * step
            val leftIndex = position.toInt()
            val rightIndex = min(leftIndex + 1, source.size - 1)
            val frac = position - leftIndex
            result[i] = ((1 - frac) * source[leftIndex] + frac * source[rightIndex]).toFloat()
        }
        return result
    }

    data class StandardScalerParams(
        val mean: FloatArray,
        val scale: FloatArray
    ) {
        fun normalize(value: Float, index: Int): Float {
            val meanValue = mean.getOrNull(index) ?: 0f
            val scaleValue = scale.getOrNull(index)?.takeIf { kotlin.math.abs(it) > 1e-6f } ?: 1f
            return (value - meanValue) / scaleValue
        }
    }

    class EngineResult(
        val rawAudio: ModelOutput?,
        val feature: ModelOutput?
    ) {
        data class ModelOutput(
            val logits: FloatArray,
            val probabilities: FloatArray,
            val labels: List<String>
        )

        companion object {
            val EMPTY = EngineResult(null, null)
        }
    }

    companion object {
        private const val TAG = "CryAnalyzer"
        private const val RAW_SAMPLE_RATE = 44_100
        private const val RAW_FFT_SIZE = 2048
        private const val RAW_HOP_LENGTH = RAW_FFT_SIZE / 2
        private const val RAW_MEL_BINS = 40
        private const val RAW_TARGET_FRAMES = 381
        private const val RAW_CHANNELS = 2
    }
}
