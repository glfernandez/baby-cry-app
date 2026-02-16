package com.aiyana.cry.monitor

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.webrtc.*
import java.util.concurrent.Executors

/**
 * Manages WebRTC peer connections for audio/video streaming between devices.
 */
class WebRTCManager(
    private val context: Context,
    private val scope: CoroutineScope,
    private val deviceRole: DeviceRole,
    private val onIceCandidate: (IceCandidate) -> Unit,
    private val onRemoteVideoTrack: (VideoTrack) -> Unit
) {
    private val TAG = "WebRTCManager"

    private val connectionState = MutableStateFlow<MonitorConnectionState>(MonitorConnectionState.Disconnected)
    val connectionStateFlow: StateFlow<MonitorConnectionState> = connectionState.asStateFlow()

    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var localVideoTrack: VideoTrack? = null
    private var localAudioTrack: AudioTrack? = null
    private var videoCapturer: CameraVideoCapturer? = null
    private var videoSource: VideoSource? = null
    private var audioSource: AudioSource? = null
    private var surfaceTextureHelper: SurfaceTextureHelper? = null

    private val rootEglBase: EglBase by lazy {
        EglBase.create()
    }

    private var isInitialized = false

    init {
        initializePeerConnectionFactory()
    }

    private fun initializePeerConnectionFactory() {
        scope.launch(Dispatchers.IO) {
            try {
                val initializationOptions = PeerConnectionFactory.InitializationOptions.builder(context)
                    .setEnableInternalTracer(false)
                    .createInitializationOptions()
                PeerConnectionFactory.initialize(initializationOptions)

                val encoderFactory = DefaultVideoEncoderFactory(
                    rootEglBase.eglBaseContext,
                    true,  // enableIntelVp8Encoder
                    true   // enableH264HighProfile
                )
                val decoderFactory = DefaultVideoDecoderFactory(rootEglBase.eglBaseContext)

                val options = PeerConnectionFactory.Options()
                peerConnectionFactory = PeerConnectionFactory.builder()
                    .setOptions(options)
                    .setVideoEncoderFactory(encoderFactory)
                    .setVideoDecoderFactory(decoderFactory)
                    .createPeerConnectionFactory()

                isInitialized = true
                Log.d(TAG, "PeerConnectionFactory initialized successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize PeerConnectionFactory", e)
                connectionState.value = MonitorConnectionState.Error("Failed to initialize WebRTC: ${e.message}")
            }
        }
    }

    /**
     * Start capturing audio and video (for baby device).
     */
    fun startCapture() {
        scope.launch(Dispatchers.IO) {
            if (!isInitialized) {
                Log.e(TAG, "PeerConnectionFactory not initialized yet")
                return@launch
            }

            try {
                val factory = peerConnectionFactory ?: run {
                    Log.e(TAG, "PeerConnectionFactory is null")
                    return@launch
                }

                // Create audio source and track
                val audioConstraints = MediaConstraints()
                audioSource = factory.createAudioSource(audioConstraints)
                localAudioTrack = factory.createAudioTrack("audio_track", audioSource)

                // Create video source and track
                videoSource = factory.createVideoSource(false)
                localVideoTrack = factory.createVideoTrack("video_track", videoSource)

                // Set up camera capturer
                videoCapturer = createVideoCapturer()
                videoCapturer?.let { capturer ->
                    surfaceTextureHelper = SurfaceTextureHelper.create(
                        "CaptureThread",
                        rootEglBase.eglBaseContext
                    )
                    surfaceTextureHelper?.let { helper ->
                        capturer.initialize(helper, context, videoSource?.capturerObserver)
                        capturer.startCapture(640, 480, 30)
                    }
                }

                Log.d(TAG, "Started audio/video capture")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start capture", e)
                connectionState.value = MonitorConnectionState.Error("Failed to start capture: ${e.message}")
            }
        }
    }

    /**
     * Stop capturing audio and video.
     */
    fun stopCapture() {
        scope.launch(Dispatchers.IO) {
            try {
                videoCapturer?.stopCapture()
                videoCapturer?.dispose()
                videoCapturer = null
                surfaceTextureHelper?.dispose()
                surfaceTextureHelper = null
                localVideoTrack?.dispose()
                localVideoTrack = null
                localAudioTrack?.dispose()
                localAudioTrack = null
                videoSource?.dispose()
                videoSource = null
                audioSource?.dispose()
                audioSource = null
                Log.d(TAG, "Stopped capture")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping capture", e)
            }
        }
    }

    /**
     * Create a peer connection for streaming.
     */
    fun createPeerConnection(iceServers: List<PeerConnection.IceServer>): PeerConnection? {
        return try {
            if (!isInitialized) {
                Log.e(TAG, "PeerConnectionFactory not initialized")
                return null
            }

            val rtcConfig = PeerConnection.RTCConfiguration(iceServers).apply {
                tcpCandidatePolicy = PeerConnection.TcpCandidatePolicy.ENABLED
                bundlePolicy = PeerConnection.BundlePolicy.MAXBUNDLE
                rtcpMuxPolicy = PeerConnection.RtcpMuxPolicy.REQUIRE
                continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
            }

            val constraints = MediaConstraints()
            val pcObserver = object : PeerConnection.Observer {
                override fun onSignalingChange(state: PeerConnection.SignalingState?) {
                    Log.d(TAG, "Signaling state: $state")
                }

                override fun onIceConnectionChange(state: PeerConnection.IceConnectionState?) {
                    Log.d(TAG, "ICE connection state: $state")
                    when (state) {
                        PeerConnection.IceConnectionState.CONNECTED,
                        PeerConnection.IceConnectionState.COMPLETED -> {
                            connectionState.value = MonitorConnectionState.Connected(
                                DeviceInfo("", "", DeviceRole.PARENT_DEVICE)
                            )
                        }
                        PeerConnection.IceConnectionState.DISCONNECTED,
                        PeerConnection.IceConnectionState.FAILED,
                        PeerConnection.IceConnectionState.CLOSED -> {
                            connectionState.value = MonitorConnectionState.Disconnected
                        }
                        else -> {}
                    }
                }

                override fun onIceGatheringChange(state: PeerConnection.IceGatheringState?) {
                    Log.d(TAG, "ICE gathering state: $state")
                }

                override fun onIceCandidate(candidate: IceCandidate?) {
                    candidate?.let {
                        Log.d(TAG, "ICE candidate: ${it.sdpMid}:${it.sdpMLineIndex} ${it.sdp}")
                        onIceCandidate(it)
                    }
                }

                override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>?) {
                    Log.d(TAG, "ICE candidates removed: ${candidates?.size}")
                }

                override fun onAddStream(stream: MediaStream?) {
                    Log.d(TAG, "Stream added: ${stream?.id}")
                }

                override fun onRemoveStream(stream: MediaStream?) {
                    Log.d(TAG, "Stream removed: ${stream?.id}")
                }

                override fun onDataChannel(channel: DataChannel?) {
                    Log.d(TAG, "Data channel: ${channel?.label()}")
                }

                override fun onRenegotiationNeeded() {
                    Log.d(TAG, "Renegotiation needed")
                }

                override fun onAddTrack(receiver: RtpReceiver?, streams: Array<out MediaStream>?) {
                    Log.d(TAG, "Track added")
                    receiver?.track()?.let { track ->
                        if (track is VideoTrack) {
                            Log.d(TAG, "Remote video track received")
                            onRemoteVideoTrack(track)
                        }
                    }
                }
            }

            peerConnectionFactory?.createPeerConnection(rtcConfig, constraints, pcObserver)?.also {
                peerConnection = it
                // Add local tracks (only for baby device)
                if (deviceRole == DeviceRole.BABY_DEVICE) {
                    localAudioTrack?.let { track -> it.addTrack(track) }
                    localVideoTrack?.let { track -> it.addTrack(track) }
                }
                Log.d(TAG, "Peer connection created")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create peer connection", e)
            null
        }
    }

    /**
     * Create offer (for baby device initiating connection).
     */
    fun createOffer(): SessionDescription? {
        return try {
            val pc = peerConnection ?: run {
                Log.e(TAG, "Peer connection is null")
                return null
            }

            val constraints = MediaConstraints().apply {
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
            }

            var offer: SessionDescription? = null
            val latch = java.util.concurrent.CountDownLatch(1)

            pc.createOffer(object : SdpObserver {
                override fun onCreateSuccess(sdp: SessionDescription?) {
                    offer = sdp
                    pc.setLocalDescription(object : SdpObserver {
                        override fun onSetSuccess() {
                            Log.d(TAG, "Local description set (offer)")
                            latch.countDown()
                        }

                        override fun onSetFailure(error: String?) {
                            Log.e(TAG, "Failed to set local description: $error")
                            latch.countDown()
                        }

                        override fun onCreateSuccess(p0: SessionDescription?) {}
                        override fun onCreateFailure(p0: String?) {}
                    }, sdp)
                }

                override fun onCreateFailure(error: String?) {
                    Log.e(TAG, "Failed to create offer: $error")
                    latch.countDown()
                }

                override fun onSetSuccess() {}
                override fun onSetFailure(error: String?) {}
            }, constraints)

            latch.await()
            offer
        } catch (e: Exception) {
            Log.e(TAG, "Error creating offer", e)
            null
        }
    }

    /**
     * Create answer (for parent device responding to offer).
     */
    fun createAnswer(offer: SessionDescription): SessionDescription? {
        return try {
            val pc = peerConnection ?: run {
                Log.e(TAG, "Peer connection is null")
                return null
            }

            pc.setRemoteDescription(object : SdpObserver {
                override fun onSetSuccess() {
                    Log.d(TAG, "Remote description set (offer)")
                }

                override fun onSetFailure(error: String?) {
                    Log.e(TAG, "Failed to set remote description: $error")
                }

                override fun onCreateSuccess(p0: SessionDescription?) {}
                override fun onCreateFailure(p0: String?) {}
            }, offer)

            val constraints = MediaConstraints().apply {
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
            }

            var answer: SessionDescription? = null
            val latch = java.util.concurrent.CountDownLatch(1)

            pc.createAnswer(object : SdpObserver {
                override fun onCreateSuccess(sdp: SessionDescription?) {
                    answer = sdp
                    pc.setLocalDescription(object : SdpObserver {
                        override fun onSetSuccess() {
                            Log.d(TAG, "Local description set (answer)")
                            latch.countDown()
                        }

                        override fun onSetFailure(error: String?) {
                            Log.e(TAG, "Failed to set local description: $error")
                            latch.countDown()
                        }

                        override fun onCreateSuccess(p0: SessionDescription?) {}
                        override fun onCreateFailure(p0: String?) {}
                    }, sdp)
                }

                override fun onCreateFailure(error: String?) {
                    Log.e(TAG, "Failed to create answer: $error")
                    latch.countDown()
                }

                override fun onSetSuccess() {}
                override fun onSetFailure(error: String?) {}
            }, constraints)

            latch.await()
            answer
        } catch (e: Exception) {
            Log.e(TAG, "Error creating answer", e)
            null
        }
    }

    /**
     * Set remote description (for receiving offer/answer).
     */
    fun setRemoteDescription(sdp: SessionDescription) {
        try {
            val pc = peerConnection ?: run {
                Log.e(TAG, "Peer connection is null")
                return
            }

            pc.setRemoteDescription(object : SdpObserver {
                override fun onSetSuccess() {
                    Log.d(TAG, "Remote description set successfully")
                }

                override fun onSetFailure(error: String?) {
                    Log.e(TAG, "Failed to set remote description: $error")
                }

                override fun onCreateSuccess(p0: SessionDescription?) {}
                override fun onCreateFailure(p0: String?) {}
            }, sdp)
        } catch (e: Exception) {
            Log.e(TAG, "Error setting remote description", e)
        }
    }

    /**
     * Add ICE candidate.
     */
    fun addIceCandidate(candidate: IceCandidate) {
        try {
            val pc = peerConnection ?: run {
                Log.e(TAG, "Peer connection is null")
                return
            }

            pc.addIceCandidate(candidate)
            Log.d(TAG, "ICE candidate added")
        } catch (e: Exception) {
            Log.e(TAG, "Error adding ICE candidate", e)
        }
    }

    private fun createVideoCapturer(): CameraVideoCapturer? {
        return try {
            val enumerator = Camera2Enumerator(context)
            val deviceNames = enumerator.deviceNames

            // Prefer front-facing camera
            for (deviceName in deviceNames) {
                if (enumerator.isFrontFacing(deviceName)) {
                    return enumerator.createCapturer(deviceName, null)
                }
            }

            // Fallback to any available camera
            for (deviceName in deviceNames) {
                val capturer = enumerator.createCapturer(deviceName, null)
                if (capturer != null) {
                    return capturer
                }
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error creating video capturer", e)
            null
        }
    }

    fun dispose() {
        stopCapture()
        peerConnection?.close()
        peerConnection?.dispose()
        peerConnection = null
        peerConnectionFactory?.dispose()
        peerConnectionFactory = null
        rootEglBase.release()
        Log.d(TAG, "WebRTCManager disposed")
    }
}
