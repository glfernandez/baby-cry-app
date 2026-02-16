package com.aiyana.cry.monitor

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * Information about a discovered or paired device.
 */
@Parcelize
data class DeviceInfo(
    val deviceId: String,
    val deviceName: String,
    val role: DeviceRole,
    val ipAddress: String? = null,
    val isPaired: Boolean = false,
    val lastSeen: Long = System.currentTimeMillis()
) : Parcelable

