package com.aiyana.cry.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.aiyana.cry.monitor.DeviceRole
import com.aiyana.cry.monitor.MonitorViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MonitorRoleSelectionScreen(
    viewModel: MonitorViewModel,
    onRoleSelected: () -> Unit
) {
    val selectedRole by viewModel.selectedRole.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Baby Monitor Setup",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = "Select this device's role",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Baby Device Card
            Card(
                modifier = Modifier
                    .weight(1f)
                    .height(200.dp)
                    .clickable { viewModel.selectRole(DeviceRole.BABY_DEVICE) },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (selectedRole == DeviceRole.BABY_DEVICE) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surface
                    }
                ),
                elevation = CardDefaults.cardElevation(
                    defaultElevation = if (selectedRole == DeviceRole.BABY_DEVICE) 8.dp else 2.dp
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.BabyChangingStation,
                        contentDescription = "Baby Device",
                        modifier = Modifier.size(64.dp),
                        tint = if (selectedRole == DeviceRole.BABY_DEVICE) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Baby Device",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Captures audio/video and analyzes cries",
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Parent Device Card
            Card(
                modifier = Modifier
                    .weight(1f)
                    .height(200.dp)
                    .clickable { viewModel.selectRole(DeviceRole.PARENT_DEVICE) },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (selectedRole == DeviceRole.PARENT_DEVICE) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surface
                    }
                ),
                elevation = CardDefaults.cardElevation(
                    defaultElevation = if (selectedRole == DeviceRole.PARENT_DEVICE) 8.dp else 2.dp
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.PhoneAndroid,
                        contentDescription = "Parent Device",
                        modifier = Modifier.size(64.dp),
                        tint = if (selectedRole == DeviceRole.PARENT_DEVICE) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Parent Device",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Receives alerts and views live feed",
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        if (selectedRole != null) {
            Button(
                onClick = { onRoleSelected() },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text(
                    text = "Continue",
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        }
    }
}

