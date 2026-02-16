#!/bin/bash

# Build and install Android app on connected devices
# Usage: ./install.sh [device_id]

set -e

echo "🔨 Building APK..."
./gradlew assembleDebug

APK_PATH="app/build/outputs/apk/debug/app-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    echo "❌ APK not found at $APK_PATH"
    exit 1
fi

echo "✅ Build successful!"

# Check if specific device ID provided
if [ -n "$1" ]; then
    DEVICE_ID="$1"
    echo "📱 Installing on device: $DEVICE_ID"
    adb -s "$DEVICE_ID" install -r "$APK_PATH"
else
    # Get all connected devices
    DEVICES=$(adb devices | grep -v "List" | grep "device" | awk '{print $1}')
    
    if [ -z "$DEVICES" ]; then
        echo "❌ No devices connected. Connect a device and run 'adb devices'"
        exit 1
    fi
    
    # Install on all devices
    for device in $DEVICES; do
        echo "📱 Installing on device: $device"
        adb -s "$device" install -r "$APK_PATH"
    done
fi

echo "✅ Installation complete!"

