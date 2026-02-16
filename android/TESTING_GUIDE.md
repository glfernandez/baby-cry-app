# Testing Guide for Child Monitor

## Pre-Testing Checklist

### 1. Build Configuration
✅ Firebase plugin is commented out (no `google-services.json` needed for initial testing)
✅ All dependencies are included in `build.gradle.kts`
✅ Permissions are added to `AndroidManifest.xml`

### 2. What to Test

#### Basic Navigation
1. Open the app
2. Look for the phone icon (📱) in the top-right corner of the app bar
3. Tap it to navigate to Baby Monitor mode
4. You should see the role selection screen

#### Role Selection
1. **Baby Device Card**: Tap to select baby device role
   - Card should highlight
   - "Continue" button should appear
2. **Parent Device Card**: Tap to select parent device role
   - Card should highlight
   - "Continue" button should appear
3. Tap "Continue" to proceed to pairing screen

#### Baby Device Mode
1. After selecting Baby Device and continuing:
   - Should see "Baby Device Mode" screen
   - "Start Monitoring" button should be visible
2. Tap "Start Monitoring":
   - Button should change to show "Stop"
   - Status should show "Waiting for parent device to connect..."
   - Device should start advertising on network (check logcat)

#### Parent Device Mode
1. After selecting Parent Device and continuing:
   - Should see "Search for baby devices" screen
   - "Start Searching" button should be visible
2. Tap "Start Searching":
   - Should show "Searching for devices..." with spinner
   - If devices are found, they should appear in a list
   - Each device card should show device name and IP address

### 3. Testing Device Discovery

**Requirements:**
- Two Android devices (or two emulators, or one device + emulator)
- Both devices on the same Wi-Fi network (or emulators on same host network)
- Both devices have the app installed

#### Option A: Testing with Physical Devices

**Steps:**
1. **Device 1 (Baby Device)**:
   - Select "Baby Device" role
   - Tap "Start Monitoring"
   - Device starts advertising

2. **Device 2 (Parent Device)**:
   - Select "Parent Device" role
   - Tap "Start Searching"
   - Should discover Device 1 within a few seconds
   - Device 1 should appear in the list

3. **On Parent Device**:
   - Tap on the discovered device card
   - Should show "Connecting..." state
   - (Note: Full connection requires WebRTC signaling, which needs Firebase)

#### Option B: Testing with Android Emulators

**Yes, you can test with Android emulators!** Here's how:

**Setup:**
1. **Launch two Android emulators** in Android Studio:
   - Emulator 1 (Baby Device)
   - Emulator 2 (Parent Device)
   - Both should be running simultaneously

2. **Network Configuration:**
   - By default, Android emulators can communicate with each other on the same host machine
   - Each emulator gets its own IP address (usually `10.0.2.15` for the first, `10.0.2.16` for the second, etc.)
   - UDP broadcasts should work between emulators on the same host

3. **Install the app on both emulators:**
   ```bash
   # Install on first emulator
   adb -s emulator-5554 install app-debug.apk
   
   # Install on second emulator
   adb -s emulator-5556 install app-debug.apk
   ```

4. **Testing Steps:**
   - **Emulator 1 (Baby Device)**:
     - Open the app
     - Navigate to Monitor mode
     - Select "Baby Device" role
     - Tap "Start Monitoring"
     - Check logcat: `adb -s emulator-5554 logcat | grep DeviceDiscovery`
     - Should see: "Sent discovery broadcast: BABY_MONITOR|..."
   
   - **Emulator 2 (Parent Device)**:
     - Open the app
     - Navigate to Monitor mode
     - Select "Parent Device" role
     - Tap "Start Searching"
     - Check logcat: `adb -s emulator-5556 logcat | grep DeviceDiscovery`
     - Should see: "Discovered device: ..."
     - Device should appear in the list

**Important Notes for Emulator Testing:**
- ⚠️ **UDP Broadcasts DO NOT WORK between emulators**: Android emulators use isolated network stacks. UDP broadcasts to `255.255.255.255` will NOT reach other emulators. This is a known limitation.
- ✅ **Multicast Lock**: Should work in emulators (fixed in latest code)
- ⚠️ **Network Isolation**: Emulators cannot discover each other using UDP broadcasts. This is expected behavior.
- ⚠️ **IP Addresses**: Emulators use `10.0.2.x` addresses by default, but they're on separate virtual networks.
- ⚠️ **Wi-Fi Permissions**: Emulators may not require actual Wi-Fi permissions, but the code should still work

**⚠️ CRITICAL: For Testing Device Discovery, Use Physical Devices!**

The current UDP broadcast discovery mechanism **will NOT work between Android emulators**. This is a fundamental limitation of how Android emulators handle networking. You have two options:

1. **Use Physical Devices (Recommended)**: 
   - Install the app on two physical Android devices
   - Connect both to the same Wi-Fi network
   - Device discovery should work correctly

2. **Use One Emulator + One Physical Device**:
   - Run one emulator and one physical device
   - Connect the physical device to the same Wi-Fi network as your development machine
   - This may work if the emulator can reach the physical device's network

**Why Emulators Don't Work:**
- Each Android emulator runs in its own isolated network namespace
- UDP broadcasts sent from one emulator don't reach other emulators
- The emulator's `10.0.2.15` address is only accessible from the host machine, not from other emulators
- This is a limitation of the Android emulator architecture, not a bug in the code

**Alternative: Using Emulator Network Configuration**

If emulators can't discover each other, you can manually configure their network:

```bash
# Check emulator network settings
adb -s emulator-5554 emu network

# Or use Android Studio's Extended Controls:
# Settings > Extended Controls > Network > Network Settings
```

**Debugging Emulator Network Issues:**

1. **Check if emulators can ping each other:**
   ```bash
   # Get IP of emulator 1
   adb -s emulator-5554 shell getprop net.dns1
   
   # Get IP of emulator 2
   adb -s emulator-5556 shell getprop net.dns1
   ```

2. **Test UDP connectivity:**
   - Use `adb shell` to run `netcat` or similar tools
   - Or check logcat for network errors

3. **Check multicast:**
   - Emulators should support multicast, but if not, you may need to use a different discovery method for emulator testing

### 4. Logcat Monitoring

Watch for these log tags:
- `DeviceDiscovery` - Device discovery/advertising logs
- `WebRTCManager` - WebRTC initialization and capture logs
- `MonitorViewModel` - State management logs

**Expected logs for Baby Device:**
```
DeviceDiscovery: Sent discovery broadcast: BABY_MONITOR|...
WebRTCManager: PeerConnectionFactory initialized
WebRTCManager: Started audio/video capture
```

**Expected logs for Parent Device:**
```
DeviceDiscovery: Discovered device: [device name] at [IP address]
```

### 5. Known Limitations (Current Implementation)

⚠️ **WebRTC Connection**: 
- Peer connection establishment is not fully implemented yet
- Requires Firebase signaling server for NAT traversal
- Video capture may not work without proper camera permissions

⚠️ **Firebase**: 
- Currently commented out in build.gradle.kts
- Will be needed for full signaling implementation

⚠️ **Network Discovery**:
- Only works on local Wi-Fi network
- Requires multicast to be enabled
- May not work on some corporate/public networks

### 6. Troubleshooting

**Issue: App crashes on monitor screen**
- Check logcat for errors
- Verify all permissions are granted
- Check if WebRTC library loaded correctly

**Issue: Devices not discovering each other**
- Verify both devices are on same Wi-Fi network
- Check if Wi-Fi multicast is enabled
- Try restarting the discovery on both devices
- Check firewall settings on router

**Issue: Camera not working (Baby Device)**
- Grant camera permission when prompted
- Check if device has a front-facing camera
- Verify CameraX dependencies are included

**Issue: Build errors**
- If Firebase plugin error: It's already commented out, should be fine
- If WebRTC import errors: Check if `org.webrtc:google-webrtc` dependency is resolved
- Sync Gradle files in Android Studio

### 7. Testing Checklist

- [ ] App builds successfully
- [ ] Monitor button appears in top bar
- [ ] Role selection screen displays correctly
- [ ] Can select Baby Device role
- [ ] Can select Parent Device role
- [ ] Can navigate to pairing screen
- [ ] Baby Device can start monitoring
- [ ] Parent Device can start searching
- [ ] Device discovery works (with 2 devices)
- [ ] Discovered devices appear in list
- [ ] Can tap on discovered device (shows connecting state)
- [ ] Back navigation works correctly
- [ ] No crashes or ANRs

### 8. Next Steps After Testing

Once basic functionality is confirmed:
1. Add Firebase project and `google-services.json`
2. Implement WebRTC signaling via Firebase
3. Integrate cry analysis into baby device stream
4. Add video rendering on parent device
5. Implement event notifications

