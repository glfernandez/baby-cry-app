package com.aiyana.cry.monitor

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import org.webrtc.IceCandidate
import org.webrtc.PeerConnection
import org.webrtc.SessionDescription
import org.webrtc.VideoTrack
import java.util.UUID

/**
 * ViewModel for managing baby monitor functionality.
 */
class MonitorViewModel(application: Application) : AndroidViewModel(application) {
    private val TAG = "MonitorViewModel"

    private val deviceId = UUID.randomUUID().toString()
    private val deviceName = android.os.Build.MODEL

    private var currentRole: DeviceRole? = null
    private var webRTCManager: WebRTCManager? = null
    private var signalingManager: WebRTCSignalingManager? = null
    private var discoveryManager: DeviceDiscoveryManager? = null
    private var connectionTimeoutJob: Job? = null

    private val _selectedRole = MutableStateFlow<DeviceRole?>(null)
    val selectedRole: StateFlow<DeviceRole?> = _selectedRole.asStateFlow()

    private val _connectionState = MutableStateFlow<MonitorConnectionState>(MonitorConnectionState.Disconnected)
    val connectionState: StateFlow<MonitorConnectionState> = _connectionState.asStateFlow()

    private val _discoveredDevices = MutableStateFlow<List<DeviceInfo>>(emptyList())
    val discoveredDevices: StateFlow<List<DeviceInfo>> = _discoveredDevices.asStateFlow()

    private val _remoteVideoTrack = MutableStateFlow<VideoTrack?>(null)
    val remoteVideoTrack: StateFlow<VideoTrack?> = _remoteVideoTrack.asStateFlow()

    companion object {
        private const val CONNECTION_TIMEOUT_MS = 60000L // 60 seconds
    }

    init {
        // Load saved role preference
        viewModelScope.launch {
            val savedRole = loadSavedRole()
            savedRole?.let { _selectedRole.value = it }
        }
    }

    /**
     * Select device role (Baby Device or Parent Device).
     */
    fun selectRole(role: DeviceRole) {
        viewModelScope.launch {
            _selectedRole.value = role
            currentRole = role
            saveRole(role)
            initializeForRole(role)
        }
    }

    /**
     * Start device discovery/advertising based on role.
     */
    fun startMonitoring() {
        viewModelScope.launch {
            val role = currentRole ?: run {
                Log.w(TAG, "Cannot start monitoring: no role selected")
                _connectionState.value = MonitorConnectionState.Error("No role selected")
                return@launch
            }

            Log.d(TAG, "Starting monitoring as ${role.name}")

            when (role) {
                DeviceRole.BABY_DEVICE -> {
                    if (discoveryManager == null) {
                        Log.e(TAG, "DiscoveryManager is null for baby device")
                        _connectionState.value = MonitorConnectionState.Error("Discovery manager not initialized")
                        return@launch
                    }
                    discoveryManager?.startAdvertising()
                    signalingManager?.startServer()
                    webRTCManager?.startCapture()
                    _connectionState.value = MonitorConnectionState.Discovering
                    Log.d(TAG, "Baby device started advertising, waiting for parent...")
                    
                    // For emulator testing: Since TCP acknowledgment won't work between emulators,
                    // we'll simulate connection after a delay if we detect we're on an emulator
                    // This is a workaround - on real devices, TCP acknowledgment will work
                    viewModelScope.launch {
                        delay(5000) // Wait 5 seconds for parent to connect via TCP
                        if (_connectionState.value is MonitorConnectionState.Discovering) {
                            // Check if we're on an emulator (IP starts with 10.0.2.)
                            try {
                                val interfaces = java.net.NetworkInterface.getNetworkInterfaces()
                                while (interfaces.hasMoreElements()) {
                                    val networkInterface = interfaces.nextElement()
                                    val addresses = networkInterface.inetAddresses
                                    while (addresses.hasMoreElements()) {
                                        val address = addresses.nextElement()
                                        if (!address.isLoopbackAddress && address is java.net.Inet4Address) {
                                            val ip = address.hostAddress ?: ""
                                            if (ip.startsWith("10.0.2.")) {
                                                Log.w(TAG, "Emulator detected - simulating connection for testing (TCP won't work on emulators)")
                                                val parentDevice = DeviceInfo(
                                                    deviceId = "simulated-parent",
                                                    deviceName = "Parent Device (Simulated)",
                                                    role = DeviceRole.PARENT_DEVICE,
                                                    ipAddress = null,
                                                    isPaired = true
                                                )
                                                _connectionState.value = MonitorConnectionState.Connected(parentDevice)
                                                connectionTimeoutJob?.cancel()
                                                connectionTimeoutJob = null
                                                return@launch
                                            }
                                        }
                                    }
                                }
                            } catch (e: Exception) {
                                // Ignore
                            }
                        }
                    }
                    
                    // Set up timeout for baby device
                    // Note: Currently, baby device won't know when parent connects until WebRTC signaling is implemented
                    // This timeout provides user feedback if no connection is established
                    connectionTimeoutJob = launch {
                        delay(CONNECTION_TIMEOUT_MS)
                        if (_connectionState.value is MonitorConnectionState.Discovering) {
                            Log.w(TAG, "Connection timeout: No parent device connected after ${CONNECTION_TIMEOUT_MS}ms")
                            _connectionState.value = MonitorConnectionState.Error(
                                "Couldn't find a parent device. Make sure both phones are on the same Wi-Fi and Parent Mode is open."
                            )
                        }
                    }
                }
                DeviceRole.PARENT_DEVICE -> {
                    if (discoveryManager == null) {
                        Log.e(TAG, "DiscoveryManager is null for parent device")
                        _connectionState.value = MonitorConnectionState.Error("Discovery manager not initialized")
                        return@launch
                    }
                    discoveryManager?.startDiscovering()
                    _connectionState.value = MonitorConnectionState.Discovering
                    Log.d(TAG, "Parent device started discovering...")
                }
            }
        }
    }

    /**
     * Stop monitoring and disconnect.
     */
    fun stopMonitoring() {
        viewModelScope.launch {
            Log.d(TAG, "Stopping monitoring")
            connectionTimeoutJob?.cancel()
            connectionTimeoutJob = null
            discoveryManager?.stop()
            signalingManager?.stop()
            webRTCManager?.stopCapture()
            _connectionState.value = MonitorConnectionState.Disconnected
            _discoveredDevices.value = emptyList()
            _remoteVideoTrack.value = null
            Log.d(TAG, "Monitoring stopped")
        }
    }

    /**
     * Connect to a discovered device (for parent devices).
     */
    fun connectToDevice(device: DeviceInfo) {
        viewModelScope.launch {
            Log.d(TAG, "Connecting to device: ${device.deviceName} (${device.deviceId}) at ${device.ipAddress}")
            _connectionState.value = MonitorConnectionState.Connecting(device)
            
            // TODO: Establish WebRTC connection
            // This will be implemented when we add signaling
            // For now, simulate connection after a short delay
            // In production, this should wait for actual WebRTC connection establishment
            kotlinx.coroutines.delay(1000)
            _connectionState.value = MonitorConnectionState.Connected(device)
            Log.d(TAG, "Connected to device: ${device.deviceName}")
            
            // TODO: Send connection acknowledgment to baby device
            // This should be done via a TCP socket or WebRTC signaling
        }
    }

    /**
     * Connect to a device by IP address (for emulator testing).
     * This bypasses UDP broadcast discovery and allows direct connection.
     */
    fun connectToIpAddress(ipAddress: String) {
        viewModelScope.launch {
            if (ipAddress.isBlank()) {
                _connectionState.value = MonitorConnectionState.Error("IP address cannot be empty")
                return@launch
            }

            // Validate IP address format (basic check)
            val ipPattern = Regex("^([0-9]{1,3}\\.){3}[0-9]{1,3}$")
            if (!ipPattern.matches(ipAddress)) {
                _connectionState.value = MonitorConnectionState.Error("Invalid IP address format")
                return@launch
            }

            Log.d(TAG, "Connecting to device at IP: $ipAddress (manual entry for testing)")
            
            // Create a DeviceInfo from the IP address
            val manualDevice = DeviceInfo(
                deviceId = "manual-$ipAddress",
                deviceName = "Baby Device ($ipAddress)",
                role = DeviceRole.BABY_DEVICE,
                ipAddress = ipAddress,
                isPaired = false
            )

            _connectionState.value = MonitorConnectionState.Connecting(manualDevice)
            
            // Establish WebRTC connection
            establishWebRTCConnection(ipAddress, manualDevice)
        }
    }

    /**
     * Clear role selection and return to role selection screen.
     */
    fun clearRole() {
        viewModelScope.launch {
            stopMonitoring()
            _selectedRole.value = null
            currentRole = null
            cleanup()
        }
    }

    private fun initializeForRole(role: DeviceRole) {
        cleanup()

        val context = getApplication<Application>()
        
        // Initialize WebRTC manager
        webRTCManager = WebRTCManager(
            context = context,
            scope = viewModelScope,
            deviceRole = role,
            onIceCandidate = { candidate ->
                signalingManager?.sendIceCandidate(candidate)
            },
            onRemoteVideoTrack = { track ->
                _remoteVideoTrack.value = track
            }
        )
        
        // Initialize signaling manager
        signalingManager = WebRTCSignalingManager(
            scope = viewModelScope,
            deviceRole = role,
            onOffer = { offer ->
                handleOffer(offer)
            },
            onAnswer = { answer ->
                handleAnswer(answer)
            },
            onIceCandidate = { candidate ->
                webRTCManager?.addIceCandidate(candidate)
            }
        )

        discoveryManager = DeviceDiscoveryManager(
            context,
            viewModelScope,
            role,
            deviceId,
            deviceName
        )
        
        // Collect discovered devices
        viewModelScope.launch {
            discoveryManager?.discoveredDevicesFlow?.collect { devices ->
                Log.d(TAG, "Discovered devices updated: ${devices.size} device(s)")
                devices.forEach { device ->
                    Log.d(TAG, "  - ${device.deviceName} (${device.deviceId}) at ${device.ipAddress}")
                }
                _discoveredDevices.value = devices
            }
        }
        
        // Collect connection acknowledgments (for baby devices)
        if (role == DeviceRole.BABY_DEVICE) {
            viewModelScope.launch {
                discoveryManager?.connectionAcknowledgedFlow?.collect { acknowledged ->
                    if (acknowledged) {
                        Log.d(TAG, "Parent device connected - updating baby device state")
                        // Create a dummy device info for the connected state
                        val parentDevice = DeviceInfo(
                            deviceId = "connected-parent",
                            deviceName = "Parent Device",
                            role = DeviceRole.PARENT_DEVICE,
                            ipAddress = null,
                            isPaired = true
                        )
                        _connectionState.value = MonitorConnectionState.Connected(parentDevice)
                        // Cancel timeout job since we're now connected
                        connectionTimeoutJob?.cancel()
                        connectionTimeoutJob = null
                    }
                }
            }
        }
    }

    private fun establishWebRTCConnection(ipAddress: String, targetDevice: DeviceInfo) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Connect to signaling server
                signalingManager?.connectToServer(ipAddress)
                delay(500) // Wait for connection

                // Create peer connection with STUN servers
                val iceServers = listOf(
                    PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer(),
                    PeerConnection.IceServer.builder("stun:stun1.l.google.com:19302").createIceServer()
                )
                webRTCManager?.createPeerConnection(iceServers)

                // Create offer
                val offer = webRTCManager?.createOffer()
                if (offer != null) {
                    signalingManager?.sendOffer(offer)
                    Log.d(TAG, "Sent WebRTC offer to $ipAddress")
                } else {
                    Log.e(TAG, "Failed to create offer")
                    _connectionState.value = MonitorConnectionState.Error("Failed to create WebRTC offer")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error establishing WebRTC connection", e)
                _connectionState.value = MonitorConnectionState.Error("Connection failed: ${e.message}")
            }
        }
    }

    private fun handleOffer(offer: SessionDescription) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Create peer connection with STUN servers
                val iceServers = listOf(
                    PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer(),
                    PeerConnection.IceServer.builder("stun:stun1.l.google.com:19302").createIceServer()
                )
                webRTCManager?.createPeerConnection(iceServers)

                // Create answer
                val answer = webRTCManager?.createAnswer(offer)
                if (answer != null) {
                    signalingManager?.sendAnswer(answer)
                    Log.d(TAG, "Sent WebRTC answer")
                } else {
                    Log.e(TAG, "Failed to create answer")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error handling offer", e)
            }
        }
    }

    private fun handleAnswer(answer: SessionDescription) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                webRTCManager?.setRemoteDescription(answer)
                Log.d(TAG, "Set remote description (answer)")
            } catch (e: Exception) {
                Log.e(TAG, "Error handling answer", e)
            }
        }
    }

    private fun cleanup() {
        connectionTimeoutJob?.cancel()
        connectionTimeoutJob = null
        webRTCManager?.dispose()
        webRTCManager = null
        signalingManager?.stop()
        signalingManager = null
        discoveryManager?.stop()
        discoveryManager = null
        _remoteVideoTrack.value = null
    }

    private suspend fun loadSavedRole(): DeviceRole? {
        return try {
            val prefs = getApplication<Application>().getSharedPreferences("monitor_prefs", Context.MODE_PRIVATE)
            val roleName = prefs.getString("device_role", null)
            roleName?.let { DeviceRole.valueOf(it) }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading saved role", e)
            null
        }
    }

    private suspend fun saveRole(role: DeviceRole) {
        try {
            val prefs = getApplication<Application>().getSharedPreferences("monitor_prefs", Context.MODE_PRIVATE)
            prefs.edit().putString("device_role", role.name).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Error saving role", e)
        }
    }

    override fun onCleared() {
        super.onCleared()
        cleanup()
    }
}

