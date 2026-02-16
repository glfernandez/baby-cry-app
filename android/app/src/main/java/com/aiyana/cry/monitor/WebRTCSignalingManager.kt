package com.aiyana.cry.monitor

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.webrtc.IceCandidate
import org.webrtc.SessionDescription
import java.io.*
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Manages WebRTC signaling via TCP socket.
 * Handles exchange of SDP offers/answers and ICE candidates.
 */
class WebRTCSignalingManager(
    private val scope: CoroutineScope,
    private val deviceRole: DeviceRole,
    private val onOffer: (SessionDescription) -> Unit,
    private val onAnswer: (SessionDescription) -> Unit,
    private val onIceCandidate: (IceCandidate) -> Unit
) {
    private val TAG = "WebRTCSignaling"

    companion object {
        private const val SIGNALING_PORT = 8890
    }

    private var serverSocket: ServerSocket? = null
    private var clientSocket: Socket? = null
    private var dataInputStream: DataInputStream? = null
    private var dataOutputStream: DataOutputStream? = null
    private val isRunning = AtomicBoolean(false)

    /**
     * Start signaling server (for baby device).
     */
    fun startServer() {
        if (deviceRole != DeviceRole.BABY_DEVICE) return

        scope.launch(Dispatchers.IO) {
            try {
                isRunning.set(true)
                serverSocket = ServerSocket(SIGNALING_PORT)
                Log.d(TAG, "Signaling server started on port $SIGNALING_PORT")

                while (isRunning.get() && !serverSocket!!.isClosed) {
                    try {
                        val socket = serverSocket!!.accept()
                        Log.d(TAG, "Client connected to signaling server")
                        handleConnection(socket)
                    } catch (e: Exception) {
                        if (!serverSocket!!.isClosed) {
                            Log.e(TAG, "Error accepting connection", e)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start signaling server", e)
            }
        }
    }

    /**
     * Connect to signaling server (for parent device).
     */
    fun connectToServer(serverIp: String) {
        if (deviceRole != DeviceRole.PARENT_DEVICE) return

        scope.launch(Dispatchers.IO) {
            try {
                isRunning.set(true)
                val socket = Socket(serverIp, SIGNALING_PORT)
                clientSocket = socket
                Log.d(TAG, "Connected to signaling server at $serverIp:$SIGNALING_PORT")
                handleConnection(socket)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to connect to signaling server", e)
                isRunning.set(false)
            }
        }
    }

    private fun handleConnection(socket: Socket) {
        scope.launch(Dispatchers.IO) {
            try {
                dataInputStream = DataInputStream(socket.getInputStream())
                dataOutputStream = DataOutputStream(socket.getOutputStream())

                while (isRunning.get() && !socket.isClosed) {
                    try {
                        val messageType = dataInputStream!!.readUTF()
                        Log.d(TAG, "Received message type: $messageType")

                        when (messageType) {
                            "OFFER" -> {
                                val sdpType = dataInputStream!!.readUTF()
                                val sdp = dataInputStream!!.readUTF()
                                val offer = SessionDescription(
                                    SessionDescription.Type.fromCanonicalForm(sdpType),
                                    sdp
                                )
                                Log.d(TAG, "Received offer")
                                onOffer(offer)
                            }
                            "ANSWER" -> {
                                val sdpType = dataInputStream!!.readUTF()
                                val sdp = dataInputStream!!.readUTF()
                                val answer = SessionDescription(
                                    SessionDescription.Type.fromCanonicalForm(sdpType),
                                    sdp
                                )
                                Log.d(TAG, "Received answer")
                                onAnswer(answer)
                            }
                            "ICE_CANDIDATE" -> {
                                val sdpMid = dataInputStream!!.readUTF()
                                val sdpMLineIndex = dataInputStream!!.readInt()
                                val sdp = dataInputStream!!.readUTF()
                                val candidate = IceCandidate(sdpMid, sdpMLineIndex, sdp)
                                Log.d(TAG, "Received ICE candidate")
                                onIceCandidate(candidate)
                            }
                            else -> {
                                Log.w(TAG, "Unknown message type: $messageType")
                            }
                        }
                    } catch (e: EOFException) {
                        Log.d(TAG, "Connection closed")
                        break
                    } catch (e: Exception) {
                        Log.e(TAG, "Error reading message", e)
                        break
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error handling connection", e)
            } finally {
                closeConnection()
            }
        }
    }

    /**
     * Send offer to remote peer.
     */
    fun sendOffer(offer: SessionDescription) {
        scope.launch(Dispatchers.IO) {
            try {
                val output = dataOutputStream ?: run {
                    Log.e(TAG, "DataOutputStream is null")
                    return@launch
                }
                output.writeUTF("OFFER")
                output.writeUTF(offer.type.canonicalForm())
                output.writeUTF(offer.description)
                output.flush()
                Log.d(TAG, "Sent offer")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send offer", e)
            }
        }
    }

    /**
     * Send answer to remote peer.
     */
    fun sendAnswer(answer: SessionDescription) {
        scope.launch(Dispatchers.IO) {
            try {
                val output = dataOutputStream ?: run {
                    Log.e(TAG, "DataOutputStream is null")
                    return@launch
                }
                output.writeUTF("ANSWER")
                output.writeUTF(answer.type.canonicalForm())
                output.writeUTF(answer.description)
                output.flush()
                Log.d(TAG, "Sent answer")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send answer", e)
            }
        }
    }

    /**
     * Send ICE candidate to remote peer.
     */
    fun sendIceCandidate(candidate: IceCandidate) {
        scope.launch(Dispatchers.IO) {
            try {
                val output = dataOutputStream ?: run {
                    Log.e(TAG, "DataOutputStream is null")
                    return@launch
                }
                output.writeUTF("ICE_CANDIDATE")
                output.writeUTF(candidate.sdpMid ?: "")
                output.writeInt(candidate.sdpMLineIndex)
                output.writeUTF(candidate.sdp)
                output.flush()
                Log.d(TAG, "Sent ICE candidate")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send ICE candidate", e)
            }
        }
    }

    fun stop() {
        isRunning.set(false)
        closeConnection()
        serverSocket?.close()
        serverSocket = null
        Log.d(TAG, "Signaling manager stopped")
    }

    private fun closeConnection() {
        try {
            dataInputStream?.close()
            dataOutputStream?.close()
            clientSocket?.close()
            dataInputStream = null
            dataOutputStream = null
            clientSocket = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing connection", e)
        }
    }
}

