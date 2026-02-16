package com.aiyana.cry.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

class AudioCaptureManager(
    val sampleRateHz: Int = DEFAULT_SAMPLE_RATE,
    private val maxBufferSeconds: Int = DEFAULT_BUFFER_SECONDS,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    private val buffer = RollingAudioBuffer(sampleRateHz * maxBufferSeconds)
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    @Volatile
    private var lastLevel: Float = 0f

    val rmsLevel: Float
        get() = lastLevel

    fun start(onLevelUpdate: (Float) -> Unit = {}) {
        if (captureJob != null) return

        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        val bytesPerFrame = Float.SIZE_BYTES
        val bufferSize = maxOf(minBuffer, CHUNK_SAMPLES * bytesPerFrame)

        audioRecord = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .setSampleRate(sampleRateHz)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .build()
            .also { it.startRecording() }

        val record = audioRecord ?: return
        buffer.clear()
        captureJob = scope.launch {
            val temp = FloatArray(CHUNK_SAMPLES)
            while (isActive) {
                val read = record.read(temp, 0, temp.size, AudioRecord.READ_BLOCKING)
                if (read > 0) {
                    buffer.append(temp, read)
                    val level = computeLevel(temp, read)
                    lastLevel = level
                    onLevelUpdate(level)
                }
            }
        }
    }

    suspend fun stop(): FloatArray =
        withContext(Dispatchers.IO) {
            captureJob?.cancelAndJoin()
            captureJob = null
            audioRecord?.apply {
                try {
                    stop()
                } catch (_: IllegalStateException) {
                }
                release()
            }
            audioRecord = null
            val data = buffer.snapshot()
            buffer.clear()
            data
        }

    suspend fun shutdown() =
        withContext(Dispatchers.IO) {
            captureJob?.cancelAndJoin()
            captureJob = null
            try {
                audioRecord?.release()
            } catch (_: IllegalStateException) {
            }
            audioRecord = null
            buffer.clear()
            scope.cancel()
        }

    private fun computeLevel(samples: FloatArray, count: Int): Float {
        var sum = 0.0
        for (i in 0 until count) {
            val v = samples[i].toDouble()
            sum += v * v
        }
        val rms = sqrt(sum / count.coerceAtLeast(1))
        return (rms / REFERENCE_RMS).toFloat().coerceIn(0f, 1.5f)
    }

    companion object {
        private const val DEFAULT_SAMPLE_RATE = 44_100
        private const val DEFAULT_BUFFER_SECONDS = 8
        private const val CHUNK_SAMPLES = 2048
        private const val REFERENCE_RMS = 0.25
    }
}

