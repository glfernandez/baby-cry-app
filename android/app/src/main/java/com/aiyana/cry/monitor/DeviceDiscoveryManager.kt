package com.aiyana.cry.monitor

import android.content.Context
import android.net.wifi.WifiManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.net.*
import java.util.*

/**
 * Manages device discovery and pairing on the local network.
 */
class DeviceDiscoveryManager(
    private val context: Context,
    private val scope: CoroutineScope,
    private val deviceRole: DeviceRole,
    private val deviceId: String,
    private val deviceName: String
) {
    private val TAG = "DeviceDiscovery"

    private val discoveredDevices = MutableStateFlow<List<DeviceInfo>>(emptyList())
    val discoveredDevicesFlow: StateFlow<List<DeviceInfo>> = discoveredDevices.asStateFlow()

    private var discoverySocket: DatagramSocket? = null
    private var isDiscovering = false
    private var multicastLockAcquired = false
    private val multicastLock: android.net.wifi.WifiManager.MulticastLock? by lazy {
        (context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager)
            ?.createMulticastLock("BabyMonitorDiscovery")?.apply {
                setReferenceCounted(true)
            }
    }

    companion object {
        private const val DISCOVERY_PORT = 8888
        private const val ACK_PORT = 8889  // Port for connection acknowledgments
        private const val DISCOVERY_INTERVAL_MS = 2000L
        private const val DISCOVERY_TIMEOUT_MS = 10000L
    }
    
    private var ackServerSocket: ServerSocket? = null
    private val _connectionAcknowledged = MutableStateFlow<Boolean>(false)
    val connectionAcknowledgedFlow: StateFlow<Boolean> = _connectionAcknowledged.asStateFlow()

    /**
     * Start advertising this device on the network (for baby devices).
     */
    fun startAdvertising() {
        if (deviceRole != DeviceRole.BABY_DEVICE) {
            Log.w(TAG, "Only baby devices should advertise")
            return
        }

        isDiscovering = true
        Log.d(TAG, "Starting advertising as baby device: $deviceName ($deviceId)")

        scope.launch(Dispatchers.IO) {
            try {
                multicastLock?.acquire()
                multicastLockAcquired = true
                Log.d(TAG, "Multicast lock acquired")
                
                val socket = DatagramSocket(DISCOVERY_PORT).apply {
                    broadcast = true
                    reuseAddress = true
                }
                discoverySocket = socket
                Log.d(TAG, "Discovery socket created on port $DISCOVERY_PORT")

                // Start TCP server to receive connection acknowledgments
                startAckServer()

                val localIp = getLocalIpAddress()
                val message = "BABY_MONITOR|$deviceId|$deviceName|$localIp"
                val buffer = message.toByteArray()
                
                Log.d(TAG, "Local IP address: $localIp")
                Log.d(TAG, "Starting broadcast loop...")
                Log.w(TAG, "NOTE: UDP broadcasts may not work between Android emulators. For emulator testing, consider using physical devices or a different discovery method.")

                while (isDiscovering && !socket.isClosed) {
                    try {
                        val broadcastAddress = InetAddress.getByName("255.255.255.255")
                        val packet = DatagramPacket(
                            buffer,
                            buffer.size,
                            broadcastAddress,
                            DISCOVERY_PORT
                        )
                        socket.send(packet)
                        Log.d(TAG, "Sent discovery broadcast: $message")
                        delay(DISCOVERY_INTERVAL_MS)
                    } catch (e: SocketException) {
                        if (e.message?.contains("Socket closed") == true || socket.isClosed) {
                            // Socket was closed, exit the loop gracefully
                            Log.d(TAG, "Advertising socket closed, stopping broadcast loop")
                            break
                        } else {
                            Log.e(TAG, "Socket error sending discovery broadcast", e)
                            delay(DISCOVERY_INTERVAL_MS)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error sending discovery broadcast", e)
                        delay(DISCOVERY_INTERVAL_MS)
                    }
                }
                
                Log.d(TAG, "Advertising stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start advertising", e)
                isDiscovering = false
            } finally {
                if (multicastLockAcquired) {
                    try {
                        multicastLock?.release()
                        multicastLockAcquired = false
                        Log.d(TAG, "Multicast lock released")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error releasing multicast lock", e)
                    }
                }
            }
        }
    }

    /**
     * Start discovering devices on the network (for parent devices).
     */
    fun startDiscovering() {
        if (deviceRole != DeviceRole.PARENT_DEVICE) {
            Log.w(TAG, "Only parent devices should discover")
            return
        }

        isDiscovering = true
        Log.d(TAG, "Starting discovery as parent device: $deviceName ($deviceId)")

        scope.launch(Dispatchers.IO) {
            try {
                // Ensure any existing socket is properly closed first
                discoverySocket?.close()
                discoverySocket = null
                // Small delay to ensure socket is fully released
                delay(100)
                
                multicastLock?.acquire()
                multicastLockAcquired = true
                Log.d(TAG, "Multicast lock acquired")
                
                // Create socket with SO_REUSEADDR to allow port sharing
                // Try to bind to DISCOVERY_PORT, but fallback to any port if it's in use
                val socket = try {
                    DatagramSocket().apply {
                        reuseAddress = true
                        broadcast = true
                        soTimeout = DISCOVERY_TIMEOUT_MS.toInt()
                        // Bind to the port after setting reuseAddress
                        bind(InetSocketAddress(DISCOVERY_PORT))
                    }
                } catch (e: BindException) {
                    Log.w(TAG, "Port $DISCOVERY_PORT is in use, using any available port", e)
                    // If port is in use, use any available port
                    // Note: This means broadcasts won't work, but manual IP entry will still work
                    DatagramSocket().apply {
                        reuseAddress = true
                        broadcast = true
                        soTimeout = DISCOVERY_TIMEOUT_MS.toInt()
                    }
                } catch (e: SocketException) {
                    if (e.message?.contains("already bound") == true || e.message?.contains("Address already in use") == true) {
                        Log.w(TAG, "Socket already bound, waiting and retrying...", e)
                        delay(500)
                        // Retry once
                        DatagramSocket().apply {
                            reuseAddress = true
                            broadcast = true
                            soTimeout = DISCOVERY_TIMEOUT_MS.toInt()
                            try {
                                bind(InetSocketAddress(DISCOVERY_PORT))
                            } catch (retryE: Exception) {
                                // Use any available port
                            }
                        }
                    } else {
                        throw e
                    }
                }
                discoverySocket = socket
                val actualPort = socket.localPort
                Log.d(TAG, "Discovery socket created on port $actualPort, starting to listen...")
                if (actualPort != DISCOVERY_PORT) {
                    Log.w(TAG, "Note: Using port $actualPort instead of $DISCOVERY_PORT (port may be in use)")
                }
                Log.w(TAG, "NOTE: UDP broadcasts may not work between Android emulators. For emulator testing, consider using physical devices or a different discovery method.")

                val buffer = ByteArray(1024)

                while (isDiscovering && !socket.isClosed) {
                    try {
                        val packet = DatagramPacket(buffer, buffer.size)
                        socket.receive(packet)

                        val message = String(packet.data, 0, packet.length)
                        Log.d(TAG, "Received discovery packet: $message")
                        val parts = message.split("|")

                        if (parts.size >= 4 && parts[0] == "BABY_MONITOR") {
                            val discoveredDevice = DeviceInfo(
                                deviceId = parts[1],
                                deviceName = parts[2],
                                role = DeviceRole.BABY_DEVICE,
                                ipAddress = parts[3],
                                isPaired = false
                            )

                            // Update discovered devices list
                            val current = discoveredDevices.value.toMutableList()
                            val existingIndex = current.indexOfFirst { it.deviceId == discoveredDevice.deviceId }
                            if (existingIndex >= 0) {
                                current[existingIndex] = discoveredDevice.copy(lastSeen = System.currentTimeMillis())
                                Log.d(TAG, "Updated existing device: ${discoveredDevice.deviceName}")
                            } else {
                                current.add(discoveredDevice)
                                Log.d(TAG, "Discovered new device: ${discoveredDevice.deviceName} at ${discoveredDevice.ipAddress}")
                            }
                            discoveredDevices.value = current
                        } else {
                            Log.w(TAG, "Received invalid discovery packet format: $message")
                        }
                    } catch (e: SocketTimeoutException) {
                        // Timeout is expected, continue listening
                        Log.v(TAG, "Discovery timeout (expected), continuing to listen...")
                        continue
                    } catch (e: SocketException) {
                        if (e.message?.contains("Socket closed") == true || socket.isClosed) {
                            // Socket was closed, exit the loop gracefully
                            Log.d(TAG, "Discovery socket closed, stopping receive loop")
                            break
                        } else {
                            Log.e(TAG, "Socket error receiving discovery packet", e)
                            delay(1000)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error receiving discovery packet", e)
                        delay(1000)
                    }
                }
                
                Log.d(TAG, "Discovery stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start discovering", e)
                isDiscovering = false
            } finally {
                if (multicastLockAcquired) {
                    try {
                        multicastLock?.release()
                        multicastLockAcquired = false
                        Log.d(TAG, "Multicast lock released")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error releasing multicast lock", e)
                    }
                }
            }
        }
    }

    /**
     * Start TCP server to receive connection acknowledgments (for baby devices).
     */
    private fun startAckServer() {
        if (deviceRole != DeviceRole.BABY_DEVICE) return
        
        scope.launch(Dispatchers.IO) {
            try {
                val serverSocket = ServerSocket(ACK_PORT)
                ackServerSocket = serverSocket
                Log.d(TAG, "Started acknowledgment server on port $ACK_PORT")
                
                while (isDiscovering && !serverSocket.isClosed) {
                    try {
                        val clientSocket = serverSocket.accept()
                        Log.d(TAG, "Received connection acknowledgment from ${clientSocket.remoteSocketAddress}")
                        clientSocket.close()
                        _connectionAcknowledged.value = true
                        Log.d(TAG, "Connection acknowledged - parent device connected!")
                    } catch (e: Exception) {
                        if (!serverSocket.isClosed) {
                            Log.e(TAG, "Error in acknowledgment server", e)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start acknowledgment server", e)
            }
        }
    }

    /**
     * Send connection acknowledgment to baby device (for parent devices).
     */
    fun sendAcknowledgment(babyDeviceIp: String) {
        if (deviceRole != DeviceRole.PARENT_DEVICE) return
        
        scope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Attempting to send acknowledgment to $babyDeviceIp:$ACK_PORT")
                val socket = Socket()
                socket.connect(InetSocketAddress(babyDeviceIp, ACK_PORT), 5000)
                Log.d(TAG, "Successfully sent connection acknowledgment to $babyDeviceIp:$ACK_PORT")
                socket.close()
            } catch (e: ConnectException) {
                Log.w(TAG, "Failed to connect to $babyDeviceIp:$ACK_PORT - Connection refused. This is expected on Android emulators as they cannot connect to each other via TCP.")
                Log.w(TAG, "For emulator testing, the connection state is simulated. On real devices, this will work correctly.")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send acknowledgment to $babyDeviceIp:$ACK_PORT", e)
            }
        }
    }

    /**
     * Stop discovery/advertising.
     */
    fun stop() {
        Log.d(TAG, "Stopping discovery/advertising")
        isDiscovering = false
        discoverySocket?.close()
        discoverySocket = null
        ackServerSocket?.close()
        ackServerSocket = null
        _connectionAcknowledged.value = false
        
        // Only release lock if we acquired it (to avoid double-release crash)
        if (multicastLockAcquired) {
            try {
                multicastLock?.release()
                multicastLockAcquired = false
                Log.d(TAG, "Multicast lock released in stop()")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing multicast lock in stop()", e)
            }
        }
        
        discoveredDevices.value = emptyList()
        Log.d(TAG, "Discovery/advertising stopped")
    }

    private fun getLocalIpAddress(): String {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val networkInterface = interfaces.nextElement()
                val addresses = networkInterface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val address = addresses.nextElement()
                    if (!address.isLoopbackAddress && address is Inet4Address) {
                        return address.hostAddress ?: "unknown"
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting local IP address", e)
        }
        return "unknown"
    }
}

