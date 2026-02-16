#!/bin/bash

# View filtered logs from Android device(s)
# Usage: ./view_logs.sh [device_id] [filter]

set -e

DEVICE_ID="$1"
FILTER="${2:-DeviceDiscovery|MonitorViewModel}"

if [ -n "$DEVICE_ID" ]; then
    echo "📱 Viewing logs from device: $DEVICE_ID"
    echo "🔍 Filter: $FILTER"
    echo "Press Ctrl+C to stop"
    echo ""
    adb -s "$DEVICE_ID" logcat | grep -E "$FILTER"
else
    # Get all connected devices
    DEVICES=$(adb devices | grep -v "List" | grep "device" | awk '{print $1}')
    
    if [ -z "$DEVICES" ]; then
        echo "❌ No devices connected"
        exit 1
    fi
    
    DEVICE_COUNT=$(echo "$DEVICES" | wc -l | tr -d ' ')
    
    if [ "$DEVICE_COUNT" -eq 1 ]; then
        DEVICE=$(echo "$DEVICES" | head -n 1)
        echo "📱 Viewing logs from device: $DEVICE"
        echo "🔍 Filter: $FILTER"
        echo "Press Ctrl+C to stop"
        echo ""
        adb -s "$DEVICE" logcat | grep -E "$FILTER"
    else
        echo "📱 Multiple devices connected. Please specify device ID:"
        echo "$DEVICES"
        echo ""
        echo "Usage: ./view_logs.sh DEVICE_ID"
        exit 1
    fi
fi

