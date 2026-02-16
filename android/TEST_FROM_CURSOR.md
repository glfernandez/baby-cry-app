# Testing Android App from Cursor IDE

This guide shows you how to build, install, and test the app directly from Cursor IDE without Android Studio.

## Prerequisites

1. **Android SDK** - You need the Android SDK installed (usually comes with Android Studio)
   - Location: `~/Library/Android/sdk` (macOS) or `%LOCALAPPDATA%\Android\Sdk` (Windows)
   - Or set `ANDROID_HOME` environment variable

2. **ADB (Android Debug Bridge)** - Should be in `$ANDROID_HOME/platform-tools/`
   - Add to PATH: `export PATH=$PATH:$ANDROID_HOME/platform-tools`

3. **Two Physical Android Devices** - Connected via USB or Wi-Fi ADB
   - Enable Developer Options and USB Debugging on both devices

## Quick Setup

### 1. Check ADB is Available

```bash
# In Cursor terminal
adb version
```

If not found, add to your `~/.zshrc` or `~/.bashrc`:
```bash
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/tools
```

### 2. Connect Devices

```bash
# List connected devices
adb devices

# If devices show as "unauthorized", accept the prompt on the phone
# You should see something like:
# List of devices attached
# ABC123XYZ    device
# DEF456UVW    device
```

### 3. Build the APK

```bash
cd babycry/android

# Build debug APK
./gradlew assembleDebug

# APK will be at: app/build/outputs/apk/debug/app-debug.apk
```

### 4. Install on Both Devices

```bash
# Install on first device (Baby Device)
adb -s DEVICE_ID_1 install -r app/build/outputs/apk/debug/app-debug.apk

# Install on second device (Parent Device)
adb -s DEVICE_ID_2 install -r app/build/outputs/apk/debug/app-debug.apk

# Or if only one device connected:
adb install -r app/build/outputs/apk/debug/app-debug.apk
# Then disconnect, connect second device, and run again
```

### 5. View Logs from Both Devices

Open two terminal windows in Cursor:

**Terminal 1 (Baby Device):**
```bash
adb -s DEVICE_ID_1 logcat | grep -E "DeviceDiscovery|MonitorViewModel|MonitorViewModel"
```

**Terminal 2 (Parent Device):**
```bash
adb -s DEVICE_ID_2 logcat | grep -E "DeviceDiscovery|MonitorViewModel|MonitorViewModel"
```

Or view all logs:
```bash
# Baby Device
adb -s DEVICE_ID_1 logcat

# Parent Device  
adb -s DEVICE_ID_2 logcat
```

## One-Command Build & Install Script

Create a helper script `install.sh` in `babycry/android/`:

```bash
#!/bin/bash

# Build
./gradlew assembleDebug

# Get device IDs
DEVICES=$(adb devices | grep -v "List" | grep "device" | awk '{print $1}')

# Install on all connected devices
for device in $DEVICES; do
    echo "Installing on device: $device"
    adb -s $device install -r app/build/outputs/apk/debug/app-debug.apk
done
```

Make it executable:
```bash
chmod +x install.sh
```

Then run:
```bash
./install.sh
```

## Testing Workflow

1. **Build and Install:**
   ```bash
   cd babycry/android
   ./gradlew assembleDebug
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

2. **Start Log Monitoring:**
   ```bash
   # In separate terminal
   adb logcat | grep -E "DeviceDiscovery|MonitorViewModel"
   ```

3. **Test on Devices:**
   - Device 1: Open app → Monitor → Baby Device → Start Monitoring
   - Device 2: Open app → Monitor → Parent Device → Start Searching
   - Watch logs for discovery messages

4. **Rebuild After Code Changes:**
   ```bash
   ./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

## Wi-Fi ADB (No USB Cable Needed)

If you want to connect devices wirelessly:

```bash
# First connect via USB, then:
adb tcpip 5555

# Get device IP (from phone Settings → About → Status → IP address)
# Then connect wirelessly:
adb connect DEVICE_IP:5555

# Now you can disconnect USB and use Wi-Fi ADB
```

## Troubleshooting

**"adb: command not found"**
- Install Android SDK Platform Tools
- Or add to PATH (see step 1)

**"No devices/emulators found"**
- Enable USB Debugging on phone
- Accept authorization prompt on phone
- Check `adb devices` output

**"INSTALL_FAILED_INSUFFICIENT_STORAGE"**
- Free up space on device

**"INSTALL_FAILED_UPDATE_INCOMPATIBLE"**
- Uninstall existing app first: `adb uninstall com.aiyana.cry`

## Cursor IDE Extensions (Optional)

While there's no direct Android Studio replacement, you can use:

1. **Android iOS Emulator** (VS Code extension) - For basic emulator management
2. **ADB Interface** - For device management
3. **Gradle for Java** - For Gradle task running

But command-line tools (ADB + Gradle) are the most reliable approach.

## Quick Reference Commands

```bash
# List devices
adb devices

# Install APK
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Uninstall app
adb uninstall com.aiyana.cry

# View logs (filtered)
adb logcat | grep DeviceDiscovery

# Clear logs
adb logcat -c

# Reboot device
adb reboot

# Get device info
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
```

