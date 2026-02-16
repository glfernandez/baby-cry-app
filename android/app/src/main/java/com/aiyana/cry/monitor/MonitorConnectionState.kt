package com.aiyana.cry.monitor

/**
 * Represents the connection state of the baby monitor.
 */
sealed class MonitorConnectionState {
    /**
     * Not connected - device is in standalone mode (cry analyzer only).
     */
    object Disconnected : MonitorConnectionState()

    /**
     * Discovering devices on the network.
     */
    object Discovering : MonitorConnectionState()

    /**
     * Connecting to a device.
     */
    data class Connecting(val targetDevice: DeviceInfo) : MonitorConnectionState()

    /**
     * Connected and streaming.
     */
    data class Connected(val pairedDevice: DeviceInfo) : MonitorConnectionState()

    /**
     * Connection error occurred.
     */
    data class Error(val message: String) : MonitorConnectionState()
}

