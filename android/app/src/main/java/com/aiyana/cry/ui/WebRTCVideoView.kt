package com.aiyana.cry.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import org.webrtc.EglBase
import org.webrtc.RendererCommon
import org.webrtc.SurfaceViewRenderer
import org.webrtc.VideoTrack

/**
 * Composable for rendering WebRTC video stream.
 */
@Composable
fun WebRTCVideoView(
    videoTrack: VideoTrack?,
    modifier: Modifier = Modifier
) {
    val eglBase = remember { EglBase.create() }
    var surfaceViewRenderer by remember { mutableStateOf<SurfaceViewRenderer?>(null) }

    AndroidView(
        factory = { context ->
            SurfaceViewRenderer(context).apply {
                init(eglBase.eglBaseContext, null)
                setMirror(false)
                setEnableHardwareScaler(true)
                setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FIT)
                surfaceViewRenderer = this
                videoTrack?.addSink(this)
            }
        },
        modifier = modifier.fillMaxSize(),
        update = { view ->
            // Remove old track
            videoTrack?.removeSink(view)
            // Add new track
            videoTrack?.addSink(view)
        },
        onRelease = { view ->
            videoTrack?.removeSink(view)
            view.release()
        }
    )

    DisposableEffect(videoTrack) {
        onDispose {
            videoTrack?.removeSink(surfaceViewRenderer)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            eglBase.release()
        }
    }
}

