package com.aiyana.cry.ml

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.jtransforms.fft.DoubleFFT_1D

class AudioFeatureExtraction {

    private val fftSize = 2048
    private val hopLength = 512
    private val envelopeFrame = 1024
    private val melBins = 128
    private val mfccCount = 20
    private val deltaWindow = 2
    private val eps = 1e-10f

    fun extractFeatureVector(samples: FloatArray, sampleRate: Int): FloatArray {
        if (samples.isEmpty()) {
            return FloatArray(FEATURE_DIM)
        }

        val amplitudeMean = amplitudeEnvelopeMean(samples)
        val rmsValues = rmsPerFrame(samples)
        val zcrValues = zeroCrossingPerFrame(samples)

        val stftResult = computeStft(samples, sampleRate)
        val magnitudes = stftResult.magnitudes
        val powerSpectrogram = stftResult.power
        val frequencies = stftResult.frequencies

        val stftMean = meanOfMatrix(magnitudes)
        val spectralCentroidMean = spectralCentroidMean(magnitudes, frequencies)
        val spectralBandwidthMean = spectralBandwidthMean(magnitudes, frequencies)
        val spectralContrastMean = spectralContrastMean(magnitudes, frequencies, sampleRate)

        val melSpectrogramDb = melSpectrogramDb(powerSpectrogram, sampleRate)
        val melSpecMean = melSpectrogramMean(melSpectrogramDb)

        val mfccFrames = computeMfcc(melSpectrogramDb)
        val mfccSummary = summarizeMfcc(mfccFrames)

        val delta1 = computeDelta(mfccFrames.takeFirstCoefficients(13))
        val delta2 = computeDelta(delta1)
        val deltaSummary = delta1.flatten().average().toFloat()
        val delta2Summary = delta2.flatten().average().toFloat()

        val features = FloatArray(FEATURE_DIM)
        var index = 0
        features[index++] = amplitudeMean
        features[index++] = rmsValues.average().toFloat()
        features[index++] = zcrValues.average().toFloat()
        features[index++] = stftMean
        features[index++] = spectralCentroidMean
        features[index++] = spectralBandwidthMean
        features[index++] = spectralContrastMean
        features[index++] = mfccSummary.first13Mean
        features[index++] = deltaSummary
        features[index++] = delta2Summary
        features[index++] = melSpecMean
        features[index++] = mfccSummary.coefficients[19]
        for (i in 0 until 13) {
            features[index++] = mfccSummary.coefficients[i]
        }
        return features
    }

    private fun amplitudeEnvelopeMean(samples: FloatArray): Float {
        var idx = 0
        var sum = 0f
        var count = 0
        while (idx < samples.size) {
            val end = min(idx + envelopeFrame, samples.size)
            var maxAbs = 0f
            for (i in idx until end) {
                val v = abs(samples[i])
                if (v > maxAbs) maxAbs = v
            }
            sum += maxAbs
            count++
            idx += envelopeFrame
        }
        return if (count > 0) sum / count else 0f
    }

    private fun rmsPerFrame(samples: FloatArray): FloatArray {
        val frames = frameCount(samples.size)
        val result = FloatArray(frames)
        var frame = 0
        var offset = 0
        while (frame < frames) {
            val end = min(offset + fftSize, samples.size)
            var sum = 0.0
            for (i in offset until end) {
                val v = samples[i].toDouble()
                sum += v * v
            }
            val value = sqrt(sum / (end - offset).coerceAtLeast(1))
            result[frame] = value.toFloat()
            offset += hopLength
            frame++
        }
        return result
    }

    private fun zeroCrossingPerFrame(samples: FloatArray): FloatArray {
        val frames = frameCount(samples.size)
        val result = FloatArray(frames)
        var frame = 0
        var offset = 0
        while (frame < frames) {
            val end = min(offset + fftSize, samples.size)
            var crossings = 0
            var prev = if (offset < samples.size) samples[offset] else 0f
            for (i in offset + 1 until end) {
                val current = samples[i]
                if (prev == 0f) {
                    prev = current
                    continue
                }
                if (current == 0f) {
                    continue
                }
                if (prev > 0f && current < 0f || prev < 0f && current > 0f) {
                    crossings++
                }
                prev = current
            }
            val frameLen = (end - offset).coerceAtLeast(1)
            result[frame] = crossings.toFloat() / frameLen.toFloat()
            offset += hopLength
            frame++
        }
        return result
    }

    private fun frameCount(length: Int): Int =
        max(1, 1 + ceil((length - fftSize).toDouble() / hopLength.toDouble()).toInt())

    private data class StftResult(
        val magnitudes: Array<FloatArray>,
        val power: Array<FloatArray>,
        val frequencies: DoubleArray
    )

    private fun computeStft(samples: FloatArray, sampleRate: Int): StftResult {
        val frames = frameCount(samples.size)
        val nBins = fftSize / 2 + 1
        val magnitudes = Array(frames) { FloatArray(nBins) }
        val power = Array(frames) { FloatArray(nBins) }
        val fft = DoubleFFT_1D(fftSize.toLong())
        val window = hannWindow()
        val frequencies = DoubleArray(nBins) { it.toDouble() * sampleRate / fftSize }
        val frameBuffer = DoubleArray(fftSize)

        var frame = 0
        var offset = 0
        while (frame < frames) {
            for (i in 0 until fftSize) {
                val idx = offset + i
                val sample = if (idx < samples.size) samples[idx] else 0f
                frameBuffer[i] = sample * window[i]
            }
            fft.realForward(frameBuffer)
            var binIndex = 0
            var bufferIdx = 0
            val magFrame = magnitudes[frame]
            val powerFrame = power[frame]
            magFrame[binIndex] = abs(frameBuffer[0]).toFloat()
            powerFrame[binIndex] = frameBuffer[0].toFloat().let { it * it }
            binIndex++
            bufferIdx = 1
            while (binIndex < nBins - 1) {
                val realPart = frameBuffer[bufferIdx]
                val imagPart = frameBuffer[bufferIdx + 1]
                val magnitude = sqrt(realPart * realPart + imagPart * imagPart)
                magFrame[binIndex] = magnitude.toFloat()
                powerFrame[binIndex] = (magnitude * magnitude).toFloat()
                binIndex++
                bufferIdx += 2
            }
            val nyquist = frameBuffer[1]
            magFrame[nBins - 1] = abs(nyquist).toFloat()
            powerFrame[nBins - 1] = (nyquist * nyquist).toFloat()

            offset += hopLength
            frame++
        }
        return StftResult(magnitudes, power, frequencies)
    }

    private fun hannWindow(): DoubleArray =
        DoubleArray(fftSize) { i ->
            if (fftSize <= 1) 1.0 else 0.5 - 0.5 * cos((2.0 * PI * i) / (fftSize - 1))
        }

    private fun meanOfMatrix(matrix: Array<FloatArray>): Float {
        var sum = 0.0
        var count = 0
        for (row in matrix) {
            for (value in row) {
                sum += value.toDouble()
                count++
            }
        }
        return if (count > 0) (sum / count).toFloat() else 0f
    }

    private fun spectralCentroidMean(magnitudes: Array<FloatArray>, frequencies: DoubleArray): Float {
        var sum = 0.0
        var count = 0
        for (frame in magnitudes) {
            var magSum = 0.0
            var centroid = 0.0
            for (i in frame.indices) {
                val mag = frame[i].toDouble()
                magSum += mag
                centroid += mag * frequencies[i]
            }
            if (magSum > 0) {
                sum += centroid / magSum
                count++
            }
        }
        return if (count > 0) (sum / count).toFloat() else 0f
    }

    private fun spectralBandwidthMean(magnitudes: Array<FloatArray>, frequencies: DoubleArray): Float {
        var sum = 0.0
        var count = 0
        for (frame in magnitudes) {
            var magSum = 0.0
            var centroid = 0.0
            for (i in frame.indices) {
                val mag = frame[i].toDouble()
                magSum += mag
                centroid += mag * frequencies[i]
            }
            if (magSum <= 0) continue
            centroid /= magSum
            var variance = 0.0
            for (i in frame.indices) {
                val mag = frame[i].toDouble()
                val diff = frequencies[i] - centroid
                variance += mag * diff * diff
            }
            sum += sqrt(variance / magSum)
            count++
        }
        return if (count > 0) (sum / count).toFloat() else 0f
    }

    private fun spectralContrastMean(
        magnitudes: Array<FloatArray>,
        frequencies: DoubleArray,
        sampleRate: Int
    ): Float {
        val nBands = 6
        val fMin = 200.0
        val upper = sampleRate / 2.0
        val bandEdges = DoubleArray(nBands + 2)
        bandEdges[0] = max(frequencies[1], fMin / 2)
        for (i in 1..nBands) {
            bandEdges[i] = min(fMin * 2.0.pow((i - 1).toDouble()), upper)
        }
        bandEdges[nBands + 1] = upper

        val frameContrasts = DoubleArray(magnitudes.size) { 0.0 }
        for (frameIndex in magnitudes.indices) {
            val frame = magnitudes[frameIndex]
            var bandSum = 0.0
            var bandCount = 0
            for (band in 0 until nBands) {
                val low = bandEdges[band]
                val high = bandEdges[band + 1]
                var maxVal = 0.0
                var minVal = Double.POSITIVE_INFINITY
                for (i in frame.indices) {
                    val freq = frequencies[i]
                    if (freq < low || freq >= high) continue
                    val energy = frame[i].toDouble()
                    if (energy > maxVal) maxVal = energy
                    if (energy > 0 && energy < minVal) minVal = energy
                }
                if (maxVal > 0) {
                    val floor = if (minVal.isFinite()) minVal else maxVal
                    val contrast = 10.0 * log10((maxVal + eps) / (floor + eps))
                    bandSum += contrast
                    bandCount++
                }
            }
            frameContrasts[frameIndex] = if (bandCount > 0) bandSum / bandCount else 0.0
        }
        return frameContrasts.average().toFloat()
    }

    private fun melSpectrogramDb(powerSpectrogram: Array<FloatArray>, sampleRate: Int): Array<FloatArray> {
        val melFilters = buildMelFilterBank(sampleRate)
        val melSpectrogram = Array(powerSpectrogram.size) { FloatArray(melBins) }
        for (frameIndex in powerSpectrogram.indices) {
            val frame = powerSpectrogram[frameIndex]
            val melFrame = melSpectrogram[frameIndex]
            for (mel in 0 until melBins) {
                var energy = 0.0
                val filter = melFilters[mel]
                for (i in frame.indices) {
                    energy += filter[i] * frame[i]
                }
                melFrame[mel] = powerToDb(energy.toFloat())
            }
        }
        return melSpectrogram
    }

    private fun melSpectrogramMean(melSpectrogramDb: Array<FloatArray>): Float {
        var sum = 0.0
        var count = 0
        for (frame in melSpectrogramDb) {
            for (value in frame) {
                sum += value.toDouble()
                count++
            }
        }
        return if (count > 0) (sum / count).toFloat() else 0f
    }

    private fun powerToDb(value: Float): Float {
        val ref = 1.0f
        val db = 10f * log10(max(eps, value / ref))
        return if (db.isFinite()) db else -80f
    }

    private fun buildMelFilterBank(sampleRate: Int): Array<DoubleArray> {
        val nBins = fftSize / 2 + 1
        val fMin = 0.0
        val fMax = sampleRate / 2.0
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = DoubleArray(melBins + 2) { i ->
            melMin + (melMax - melMin) * i / (melBins + 1)
        }
        val binFrequencies = DoubleArray(nBins) { it.toDouble() * sampleRate / fftSize }
        val filters = Array(melBins) { DoubleArray(nBins) }
        for (m in 0 until melBins) {
            val left = melPoints[m]
            val center = melPoints[m + 1]
            val right = melPoints[m + 2]
            val leftHz = melToHz(left)
            val centerHz = melToHz(center)
            val rightHz = melToHz(right)
            for (i in 0 until nBins) {
                val freq = binFrequencies[i]
                val weight = when {
                    freq < leftHz -> 0.0
                    freq <= centerHz -> (freq - leftHz) / (centerHz - leftHz).coerceAtLeast(1e-6)
                    freq <= rightHz -> (rightHz - freq) / (rightHz - centerHz).coerceAtLeast(1e-6)
                    else -> 0.0
                }
                filters[m][i] = weight
            }
        }
        return filters
    }

    private fun hzToMel(hz: Double): Double =
        2595.0 * kotlin.math.log10(1.0 + hz / 700.0)

    private fun melToHz(mel: Double): Double =
        700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun computeMfcc(melSpectrogramDb: Array<FloatArray>): Array<FloatArray> {
        val dctMatrix = dctMatrix()
        val frames = melSpectrogramDb.size
        val mfcc = Array(frames) { FloatArray(mfccCount) }
        for (frameIndex in 0 until frames) {
            val melFrame = melSpectrogramDb[frameIndex]
            for (i in 0 until mfccCount) {
                var sum = 0.0
                val dctRow = dctMatrix[i]
                for (j in 0 until melBins) {
                    sum += dctRow[j] * melFrame[j]
                }
                mfcc[frameIndex][i] = sum.toFloat()
            }
        }
        return mfcc
    }

    private fun dctMatrix(): Array<DoubleArray> {
        val matrix = Array(mfccCount) { DoubleArray(melBins) }
        val factor = sqrt(2.0 / melBins)
        for (i in 0 until mfccCount) {
            for (j in 0 until melBins) {
                matrix[i][j] = factor * cos(PI * i * (j + 0.5) / melBins)
            }
        }
        for (j in 0 until melBins) {
            matrix[0][j] = matrix[0][j] * (1.0 / sqrt(2.0))
        }
        return matrix
    }

    private fun summarizeMfcc(mfccFrames: Array<FloatArray>): MccSummary {
        val frames = mfccFrames.size
        val coeffMeans = FloatArray(mfccCount)
        var first13Sum = 0.0
        var totalFirst13 = 0
        for (i in 0 until mfccCount) {
            var sum = 0.0
            for (frame in 0 until frames) {
                sum += mfccFrames[frame][i].toDouble()
            }
            coeffMeans[i] = (sum / frames.coerceAtLeast(1)).toFloat()
        }
        for (frame in 0 until frames) {
            for (i in 0 until 13) {
                first13Sum += mfccFrames[frame][i].toDouble()
                totalFirst13++
            }
        }
        val first13Mean = if (totalFirst13 > 0) (first13Sum / totalFirst13).toFloat() else 0f
        return MccSummary(coeffMeans, first13Mean)
    }

    private data class MccSummary(
        val coefficients: FloatArray,
        val first13Mean: Float
    )

    private fun Array<FloatArray>.takeFirstCoefficients(count: Int): Array<FloatArray> {
        return Array(size) { frame ->
            FloatArray(count) { coeff ->
                this[frame][coeff]
            }
        }
    }

    private fun computeDelta(data: Array<FloatArray>): Array<FloatArray> {
        val frames = data.size
        if (frames == 0) return emptyArray()
        val coefficients = data[0].size
        val denominator = 2 * (1..deltaWindow).sumOf { it * it }
        val result = Array(frames) { FloatArray(coefficients) }
        for (t in 0 until frames) {
            for (c in 0 until coefficients) {
                var acc = 0.0
                for (n in 1..deltaWindow) {
                    val prevIndex = (t - n).coerceAtLeast(0)
                    val nextIndex = (t + n).coerceAtMost(frames - 1)
                    acc += n * (data[nextIndex][c] - data[prevIndex][c])
                }
                result[t][c] = (acc / denominator).toFloat()
            }
        }
        return result
    }

    private fun Array<FloatArray>.flatten(): FloatArray {
        val result = FloatArray(size * (if (isNotEmpty()) this[0].size else 0))
        var index = 0
        for (row in this) {
            for (value in row) {
                result[index++] = value
            }
        }
        return result
    }

    companion object {
        const val FEATURE_DIM = 25
    }
}

