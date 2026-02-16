package com.aiyana.cry.audio

import android.content.ContentResolver
import android.net.Uri
import java.io.BufferedInputStream
import java.io.InputStream
import kotlin.math.pow

data class WavData(
    val samples: FloatArray,
    val sampleRate: Int,
    val durationMillis: Long
)

@Throws(IllegalArgumentException::class)
fun ContentResolver.readWavData(uri: Uri): WavData {
    openInputStream(uri)?.use { stream ->
        return stream.readWavData()
    }
    throw IllegalArgumentException("Unable to open WAV uri: $uri")
}

@Throws(IllegalArgumentException::class)
fun InputStream.readWavData(): WavData {
    BufferedInputStream(this).use { input ->
        fun readExactly(size: Int): ByteArray {
            val buffer = ByteArray(size)
            var offset = 0
            while (offset < size) {
                val read = input.read(buffer, offset, size - offset)
                if (read == -1) {
                    throw IllegalArgumentException("Malformed WAV chunk")
                }
                offset += read
            }
            return buffer
        }

        fun skipFully(bytes: Long) {
            var remaining = bytes
            while (remaining > 0) {
                val skipped = input.skip(remaining)
                if (skipped <= 0) {
                    if (input.read() == -1) {
                        throw IllegalArgumentException("Malformed WAV chunk")
                    }
                    remaining -= 1
                } else {
                    remaining -= skipped
                }
            }
        }

        val riffHeader = readExactly(12)
        if (!riffHeader.copyOfRange(0, 4).contentEquals("RIFF".toByteArray())) {
            throw IllegalArgumentException("Invalid WAV file: missing RIFF")
        }
        if (!riffHeader.copyOfRange(8, 12).contentEquals("WAVE".toByteArray())) {
            throw IllegalArgumentException("Invalid WAV file: missing WAVE")
        }

        var fmtChunkFound = false
        var audioFormat = 0
        var numChannels = 0
        var sampleRate = 0
        var bitsPerSample = 0

        while (true) {
            val chunkHeader = input.readNBytesCompat(8)
            if (chunkHeader.size < 8) break
            val chunkId = String(chunkHeader.copyOfRange(0, 4))
            val chunkSize = chunkHeader.copyOfRange(4, 8).toLittleEndianInt()

            when (chunkId) {
                "fmt " -> {
                    val payload = readExactly(chunkSize)
                    fmtChunkFound = true
                    audioFormat = payload.copyOfRange(0, 2).toLittleEndianShort().toInt()
                    numChannels = payload.copyOfRange(2, 4).toLittleEndianShort().toInt()
                    sampleRate = payload.copyOfRange(4, 8).toLittleEndianInt()
                    bitsPerSample = payload.copyOfRange(14, 16).toLittleEndianShort().toInt()
                }

                "data" -> {
                    val pcm = readExactly(chunkSize)
                    if (audioFormat != 1) {
                        throw IllegalArgumentException("Unsupported audio format (expected PCM)")
                    }
                    return decodePcmData(
                        pcm,
                        sampleRate = sampleRate,
                        bitsPerSample = bitsPerSample,
                        channels = numChannels
                    )
                }

                else -> {
                    skipFully(chunkSize.toLong())
                }
            }

            if (chunkSize % 2 == 1) {
                skipFully(1)
            }
        }

        if (!fmtChunkFound) {
            throw IllegalArgumentException("WAV file missing fmt chunk")
        }
        throw IllegalArgumentException("WAV file missing data chunk")
    }
    throw IllegalArgumentException("WAV stream missing data chunk")
}

private fun ByteArray.toLittleEndianInt(): Int =
    (this[0].toInt() and 0xFF) or
        ((this[1].toInt() and 0xFF) shl 8) or
        ((this[2].toInt() and 0xFF) shl 16) or
        ((this[3].toInt() and 0xFF) shl 24)

private fun ByteArray.toLittleEndianShort(): Short =
    (((this[1].toInt() and 0xFF) shl 8) or (this[0].toInt() and 0xFF)).toShort()

private fun decodePcmData(
    data: ByteArray,
    sampleRate: Int,
    bitsPerSample: Int,
    channels: Int
): WavData {
    require(bitsPerSample == 16) { "Only 16-bit PCM WAV supported (found $bitsPerSample-bit)" }
    require(channels in 1..2) { "Only mono or stereo WAV supported (found $channels channels)" }

    val totalSamples = data.size / 2 / channels
    val output = FloatArray(totalSamples)
    var inputIndex = 0
    for (i in 0 until totalSamples) {
        var sampleSum = 0f
        for (channel in 0 until channels) {
            val lo = data[inputIndex++].toInt() and 0xFF
            val hi = data[inputIndex++].toInt()
            val pcm = ((hi shl 8) or lo) / 32768.0f
            sampleSum += pcm
        }
        output[i] = sampleSum / channels
    }

    val durationMillis = (totalSamples.toDouble() / sampleRate.toDouble() * 1000.0).toLong()
    return WavData(output, sampleRate, durationMillis)
}

private fun InputStream.readNBytesCompat(length: Int): ByteArray {
    val buffer = ByteArray(length)
    var totalRead = 0
    while (totalRead < length) {
        val read = read(buffer, totalRead, length - totalRead)
        if (read <= 0) break
        totalRead += read
    }
    return if (totalRead == length) buffer else buffer.copyOf(totalRead)
}

