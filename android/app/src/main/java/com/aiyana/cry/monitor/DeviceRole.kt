package com.aiyana.cry.monitor

/**
 * Represents the role of a device in the baby monitor system.
 */
enum class DeviceRole {
    /**
     * Baby device - captures audio/video and runs cry analysis locally.
     * Sends alerts and streams to parent devices.
     */
    BABY_DEVICE,

    /**
     * Parent device - receives streams and alerts from baby devices.
     * Can view live feed and receive notifications.
     */
    PARENT_DEVICE
}

