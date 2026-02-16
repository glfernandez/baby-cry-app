package com.aiyana.cry.model

import com.aiyana.cry.R

data class CrySample(
    val index: Int,
    val resId: Int,
    val displayName: String
)

object CrySamples {
    val samples: List<CrySample> = listOf(
        CrySample(0, R.raw.cry_1, "cry_1.wav"),
        CrySample(1, R.raw.cry_2, "cry_2.wav"),
        CrySample(2, R.raw.cry_3, "cry_3.wav"),
        CrySample(3, R.raw.cry_4, "cry_4.wav"),
        CrySample(4, R.raw.cry_5, "cry_5.wav"),
        CrySample(5, R.raw.cry_6, "cry_6.wav"),
        CrySample(6, R.raw.cry_7, "cry_7.wav"),
        CrySample(7, R.raw.cry_8, "cry_8.wav"),
        CrySample(8, R.raw.cry_9, "cry_9.wav"),
        CrySample(9, R.raw.cry_10, "cry_10.wav"),
        CrySample(10, R.raw.cry_11, "cry_11.wav"),
    )

    val rawIds: IntArray = samples.map { it.resId }.toIntArray()
}

