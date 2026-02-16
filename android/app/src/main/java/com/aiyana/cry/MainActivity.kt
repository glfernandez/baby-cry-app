package com.aiyana.cry

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.compose.runtime.remember
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.aiyana.cry.model.CryAnalyzerUiState
import com.aiyana.cry.model.CryAnalyzerViewModel
import com.aiyana.cry.monitor.MonitorViewModel
import com.aiyana.cry.ui.CryAnalyzerRoot
import com.aiyana.cry.ui.MonitorPairingScreen
import com.aiyana.cry.ui.MonitorRoleSelectionScreen
import com.aiyana.cry.ui.theme.PixelCryTheme

class MainActivity : ComponentActivity() {

    private val viewModel: CryAnalyzerViewModel by viewModels()
    private val monitorViewModel: MonitorViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        setContent {
            PixelCryTheme {
                val uiState by viewModel.uiState.collectAsState()
                val context = LocalContext.current
                val lifecycleOwner = LocalLifecycleOwner.current
                var permissionDialogDismissed by remember { mutableStateOf(false) }
                val permissionLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestMultiplePermissions()
                ) { result ->
                    val granted = result.values.all { it }
                    viewModel.onPermissionUpdated(granted)
                }
                val importLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.OpenDocument()
                ) { uri: Uri? ->
                    uri?.let {
                        runCatching {
                            context.contentResolver.takePersistableUriPermission(
                                it,
                                Intent.FLAG_GRANT_READ_URI_PERMISSION
                            )
                        }
                        viewModel.analyzeImportedAudio(it)
                    }
                }

                LaunchedEffect(Unit) {
                    val requestedPermissions = buildPermissionList()
                    if (context.hasAllPermissions(requestedPermissions)) {
                        viewModel.onPermissionUpdated(true)
                    } else {
                        permissionLauncher.launch(requestedPermissions.toTypedArray())
                    }
                }

                LaunchedEffect(uiState) {
                    if (uiState is CryAnalyzerUiState.PermissionRequired && !permissionDialogDismissed) {
                        // The UI will present a rationale sheet; keep track that we already prompted once.
                        permissionDialogDismissed = true
                    }
                }

                DisposableEffect(lifecycleOwner, context) {
                    val observer = LifecycleEventObserver { _, event ->
                        if (event == Lifecycle.Event.ON_RESUME) {
                            val permissions = buildPermissionList()
                            viewModel.onPermissionUpdated(context.hasAllPermissions(permissions))
                        }
                    }
                    lifecycleOwner.lifecycle.addObserver(observer)
                    onDispose {
                        lifecycleOwner.lifecycle.removeObserver(observer)
                    }
                }

                val navController = rememberNavController()
                
                NavHost(
                    navController = navController,
                    startDestination = "cry_analyzer"
                ) {
                    composable("cry_analyzer") {
                        CryAnalyzerRoot(
                            state = uiState,
                            onRequestRecording = { viewModel.startListening() },
                            onStopRecording = { viewModel.stopListening() },
                            onReset = { viewModel.reset() },
                            onOpenSettings = { context.openAppSettings() },
                            onImportRecording = { importLauncher.launch(arrayOf("audio/*", "audio/wav", "audio/x-wav")) },
                            onSelectSample = { viewModel.selectSample(it) },
                            onAnalyzeSample = { viewModel.analyzeEmbeddedSample(it) },
                            onOpenMonitor = { navController.navigate("monitor_role_selection") }
                        )
                    }
                    
                    composable("monitor_role_selection") {
                        MonitorRoleSelectionScreen(
                            viewModel = monitorViewModel,
                            onRoleSelected = { navController.navigate("monitor_pairing") }
                        )
                    }
                    
                    composable("monitor_pairing") {
                        MonitorPairingScreen(
                            viewModel = monitorViewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                }
            }
        }
    }
}

private fun android.content.Context.openAppSettings() {
    val intent = android.content.Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
        data = android.net.Uri.fromParts("package", packageName, null)
        flags = android.content.Intent.FLAG_ACTIVITY_NEW_TASK
    }
    ContextCompat.startActivity(this, intent, null)
}

private fun buildPermissionList(): List<String> =
    buildList {
        add(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.READ_MEDIA_AUDIO)
        } else {
            add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

private fun android.content.Context.hasAllPermissions(permissions: List<String>): Boolean =
    permissions.all { perm ->
        ContextCompat.checkSelfPermission(this, perm) == android.content.pm.PackageManager.PERMISSION_GRANTED
    }

