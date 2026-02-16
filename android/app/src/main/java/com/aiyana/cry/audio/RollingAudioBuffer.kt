package com.aiyana.cry.audio

/**
 * Fixed-size circular buffer for mono PCM float samples. Designed to mirror
 * the "Shazam-style" rolling capture where we always keep the most recent
 * window (e.g., last 8 seconds) while the microphone is active.
 */
class RollingAudioBuffer(
    private val capacity: Int
) {
    private val buffer = FloatArray(capacity)
    private var writeIndex = 0
    private var filled = 0

    @Synchronized
    fun append(samples: FloatArray, count: Int = samples.size) {
        val n = count.coerceIn(0, samples.size)
        if (n == 0) return
        var remaining = n
        var srcIndex = 0
        while (remaining > 0) {
            val chunk = minOf(remaining, capacity - writeIndex)
            samples.copyInto(
                destination = buffer,
                destinationOffset = writeIndex,
                startIndex = srcIndex,
                endIndex = srcIndex + chunk
            )
            writeIndex = (writeIndex + chunk) % capacity
            srcIndex += chunk
            remaining -= chunk
            filled = minOf(capacity, filled + chunk)
        }
    }

    @Synchronized
    fun snapshot(): FloatArray {
        if (filled == 0) return FloatArray(0)
        val result = FloatArray(filled)
        val tail = filled
        val start = (writeIndex - tail + capacity) % capacity
        val firstChunk = minOf(tail, capacity - start)
        buffer.copyInto(
            destination = result,
            startIndex = start,
            endIndex = start + firstChunk
        )
        if (tail > firstChunk) {
            buffer.copyInto(
                destination = result,
                destinationOffset = firstChunk,
                startIndex = 0,
                endIndex = tail - firstChunk
            )
        }
        return result
    }

    @Synchronized
    fun clear() {
        writeIndex = 0
        filled = 0
    }

    fun size(): Int = filled
    fun maxSize(): Int = capacity
}

