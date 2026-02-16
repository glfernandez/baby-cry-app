# Baby Monitor Setup Guide

## Overview

The child monitor feature has been integrated into the app, allowing two devices to connect:
- **Baby Device**: Captures audio/video and analyzes cries locally, sends alerts to parent device
- **Parent Device**: Receives live stream and alerts from baby device

## Current Implementation Status

✅ **Completed:**
- Device role selection UI (Baby Device vs Parent Device)
- Device discovery mechanism (UDP broadcast on local network)
- WebRTC manager setup for peer-to-peer streaming
- Basic UI for pairing and connection status
- Navigation integration with main app

🚧 **In Progress:**
- WebRTC peer connection establishment
- Signaling server integration (Firebase)

⏳ **Pending:**
- Cry analysis integration into baby device stream
- Video stream rendering on parent device
- Firebase for signaling and notifications
- Event history and alerts UI

## Setup Instructions

### 1. Firebase Configuration

The app requires Firebase for signaling between devices. You need to:

1. Create a Firebase project at https://console.firebase.google.com
2. Add an Android app to your Firebase project
3. Download `google-services.json` and place it in `app/` directory
4. Enable the following Firebase services:
   - **Firestore** (for signaling and device pairing)
   - **Realtime Database** (alternative signaling option)
   - **Cloud Messaging** (for push notifications)
   - **Authentication** (optional, for user accounts)

### 2. Permissions

The app now requires additional permissions:
- `INTERNET` - For network communication
- `ACCESS_NETWORK_STATE` - To check network connectivity
- `ACCESS_WIFI_STATE` - For device discovery
- `CAMERA` - For video capture (baby device)
- `CHANGE_WIFI_MULTICAST_STATE` - For device discovery

These are already added to `AndroidManifest.xml`.

### 3. Dependencies

All required dependencies have been added to `build.gradle.kts`:
- WebRTC (org.webrtc:google-webrtc)
- Firebase BOM and services
- CameraX for video capture

## Usage

1. **Open the app** and tap the phone icon in the top bar to access Baby Monitor mode
2. **Select device role**: Choose "Baby Device" or "Parent Device"
3. **For Baby Device**:
   - Tap "Start Monitoring" to begin advertising on the network
   - The device will capture audio/video and analyze cries
4. **For Parent Device**:
   - Tap "Start Searching" to discover baby devices on the same Wi-Fi network
   - Select a device from the list to connect
   - View live stream and receive alerts

## Architecture

### Components

- **DeviceRole**: Enum for device roles (BABY_DEVICE, PARENT_DEVICE)
- **DeviceInfo**: Data class for device information
- **MonitorConnectionState**: Sealed class for connection states
- **WebRTCManager**: Handles WebRTC peer connections and media capture
- **DeviceDiscoveryManager**: Manages UDP-based device discovery on local network
- **MonitorViewModel**: ViewModel managing monitor state and business logic

### Network Flow

1. **Discovery**: Baby device broadcasts UDP packets on port 8888
2. **Pairing**: Parent device receives broadcasts and displays available devices
3. **Connection**: WebRTC peer connection established (requires signaling)
4. **Streaming**: Audio/video streamed via WebRTC

## Next Steps

1. **Add Firebase Signaling**: Implement WebRTC signaling via Firebase Firestore/Realtime Database
2. **Integrate Cry Analysis**: Hook `CryAnalyzerEngine` into baby device audio stream
3. **Video Rendering**: Add SurfaceView/TextureView for video display on parent device
4. **Event Notifications**: Send cry detection events via WebRTC data channel or Firebase
5. **Connection Persistence**: Handle reconnection and connection state management

## Notes

- Device discovery currently works on local Wi-Fi network only
- WebRTC requires a signaling server (Firebase) for NAT traversal
- Camera capture is optional - audio-only mode is also supported
- The app falls back to standalone cry analyzer mode if monitor is not used

