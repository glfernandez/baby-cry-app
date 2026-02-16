package com.aiyana.cry.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aiyana.cry.monitor.DeviceInfo
import com.aiyana.cry.monitor.DeviceRole
import com.aiyana.cry.monitor.MonitorConnectionState
import com.aiyana.cry.monitor.MonitorViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MonitorPairingScreen(
    viewModel: MonitorViewModel,
    onBack: () -> Unit
) {
    val selectedRole by viewModel.selectedRole.collectAsState()
    val connectionState by viewModel.connectionState.collectAsState()
    val discoveredDevices by viewModel.discoveredDevices.collectAsState()
    val remoteVideoTrack by viewModel.remoteVideoTrack.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Connect Device") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            when (selectedRole) {
                DeviceRole.BABY_DEVICE -> {
                    BabyDeviceView(
                        connectionState = connectionState,
                        onStartMonitoring = { viewModel.startMonitoring() },
                        onStopMonitoring = { viewModel.stopMonitoring() }
                    )
                }
                DeviceRole.PARENT_DEVICE -> {
                    ParentDeviceView(
                        connectionState = connectionState,
                        discoveredDevices = discoveredDevices,
                        remoteVideoTrack = remoteVideoTrack,
                        onStartMonitoring = { viewModel.startMonitoring() },
                        onStopMonitoring = { viewModel.stopMonitoring() },
                        onConnectToDevice = { viewModel.connectToDevice(it) },
                        onConnectToIp = { ip -> viewModel.connectToIpAddress(ip) }
                    )
                }
                null -> {
                    // Should not happen, but handle gracefully
                    Text("No role selected")
                }
            }
        }
    }
}

@Composable
private fun BabyDeviceView(
    connectionState: MonitorConnectionState,
    onStartMonitoring: () -> Unit,
    onStopMonitoring: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Filled.BabyChangingStation,
            contentDescription = null,
            modifier = Modifier.size(96.dp),
            tint = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Baby Device Mode",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(16.dp))

        when (connectionState) {
            is MonitorConnectionState.Disconnected -> {
                Text(
                    text = "Ready to start monitoring",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(
                    onClick = onStartMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Start Monitoring")
                }
            }
            is MonitorConnectionState.Discovering -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Waiting for parent device to connect...",
                    style = MaterialTheme.typography.bodyLarge
                )
                Spacer(modifier = Modifier.height(24.dp))
                OutlinedButton(
                    onClick = onStopMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Stop")
                }
            }
            is MonitorConnectionState.Connected -> {
                Text(
                    text = "Connected to parent device",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(24.dp))
                OutlinedButton(
                    onClick = onStopMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Disconnect")
                }
            }
            is MonitorConnectionState.Error -> {
                Text(
                    text = "Error: ${connectionState.message}",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(
                    onClick = onStartMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Retry")
                }
            }
            else -> {}
        }
    }
}

@Composable
private fun ParentDeviceView(
    connectionState: MonitorConnectionState,
    discoveredDevices: List<DeviceInfo>,
    remoteVideoTrack: org.webrtc.VideoTrack?,
    onStartMonitoring: () -> Unit,
    onStopMonitoring: () -> Unit,
    onConnectToDevice: (DeviceInfo) -> Unit,
    onConnectToIp: (String) -> Unit
) {
    var manualIpAddress by remember { mutableStateOf("") }
    var showManualEntry by remember { mutableStateOf(false) }
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        when (connectionState) {
            is MonitorConnectionState.Disconnected -> {
                Text(
                    text = "Search for baby devices",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Make sure both devices are on the same Wi-Fi network",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(
                    onClick = onStartMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Start Searching")
                }
            }
            is MonitorConnectionState.Discovering -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Searching for devices...",
                        style = MaterialTheme.typography.titleMedium
                    )
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                }
                Spacer(modifier = Modifier.height(16.dp))

                if (discoveredDevices.isEmpty()) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Spacer(modifier = Modifier.weight(1f))
                        Text(
                            text = "No devices found yet",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Manual IP entry for emulator testing
                        if (!showManualEntry) {
                            Text(
                                text = "Testing on emulators?",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            TextButton(
                                onClick = { showManualEntry = true }
                            ) {
                                Text("Enter IP Address Manually")
                            }
                        } else {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    Text(
                                        text = "Manual Connection (For Testing)",
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = "Enter the baby device's IP address (e.g., 10.0.2.15)",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    OutlinedTextField(
                                        value = manualIpAddress,
                                        onValueChange = { manualIpAddress = it },
                                        label = { Text("IP Address") },
                                        placeholder = { Text("10.0.2.15") },
                                        modifier = Modifier.fillMaxWidth(),
                                        singleLine = true,
                                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal)
                                    )
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        OutlinedButton(
                                            onClick = { 
                                                showManualEntry = false
                                                manualIpAddress = ""
                                            },
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Text("Cancel")
                                        }
                                        Button(
                                            onClick = { 
                                                if (manualIpAddress.isNotBlank()) {
                                                    onConnectToIp(manualIpAddress.trim())
                                                }
                                            },
                                            modifier = Modifier.weight(1f),
                                            enabled = manualIpAddress.isNotBlank()
                                        ) {
                                            Text("Connect")
                                        }
                                    }
                                }
                            }
                        }
                        Spacer(modifier = Modifier.weight(1f))
                    }
                } else {
                    LazyColumn(
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(discoveredDevices) { device ->
                            DeviceCard(
                                device = device,
                                onClick = { onConnectToDevice(device) }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
                OutlinedButton(
                    onClick = onStopMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Stop Searching")
                }
            }
            is MonitorConnectionState.Connecting -> {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Connecting to ${connectionState.targetDevice.deviceName}...",
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
            is MonitorConnectionState.Connected -> {
                Column(
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(
                        text = "Connected to ${connectionState.pairedDevice.deviceName}",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(16.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    // Show video stream (for parent device)
                    if (remoteVideoTrack != null) {
                        WebRTCVideoView(
                            videoTrack = remoteVideoTrack,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                                .background(MaterialTheme.colorScheme.surfaceVariant),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Waiting for video stream...",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    OutlinedButton(
                        onClick = onStopMonitoring,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp)
                    ) {
                        Text("Disconnect")
                    }
                }
            }
            is MonitorConnectionState.Error -> {
                Text(
                    text = "Error: ${connectionState.message}",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(
                    onClick = onStartMonitoring,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Retry")
                }
            }
        }
    }
}

@Composable
private fun DeviceCard(
    device: DeviceInfo,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = device.deviceName,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = device.ipAddress ?: "Unknown IP",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Icon(
                imageVector = Icons.Filled.ArrowForward,
                contentDescription = "Connect",
                tint = MaterialTheme.colorScheme.primary
            )
        }
    }
}

