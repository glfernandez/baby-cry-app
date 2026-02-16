package com.aiyana.cry.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PhoneAndroid
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.aiyana.cry.model.ClassProbability
import com.aiyana.cry.model.CryAnalyzerUiState
import com.aiyana.cry.model.CrySamples
import com.aiyana.cry.model.ModelId
import com.aiyana.cry.model.ModelPrediction
import kotlin.math.roundToInt
import com.aiyana.cry.R

@OptIn(ExperimentalMaterial3Api::class, ExperimentalAnimationApi::class)
@Composable
fun CryAnalyzerRoot(
    state: CryAnalyzerUiState,
    onRequestRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onReset: () -> Unit,
    onOpenSettings: () -> Unit,
    onImportRecording: () -> Unit,
    onSelectSample: (Int?) -> Unit,
    onAnalyzeSample: (Int) -> Unit,
    onOpenMonitor: () -> Unit = {}
) {
    val showPermissionDialog = remember { mutableStateOf(false) }
    var showSamplePicker by remember { mutableStateOf(false) }
    val selectedSampleIndex = (state as? CryAnalyzerUiState.Idle)?.selectedSample
    val selectedSampleName = selectedSampleIndex?.let { idx ->
        CrySamples.samples.getOrNull(idx)?.displayName
    }

    LaunchedEffect(state) {
        showPermissionDialog.value = state is CryAnalyzerUiState.PermissionRequired
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = { Text("Pixel Cry Analyzer") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                ),
                actions = {
                    IconButton(onClick = onOpenMonitor) {
                        Icon(
                            imageVector = Icons.Filled.PhoneAndroid,
                            contentDescription = "Baby Monitor",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    IconButton(onClick = {}) { // Coming soon
                        Icon(
                            imageVector = Icons.Outlined.History,
                            contentDescription = "History (coming soon)",
                            modifier = Modifier
                                .padding(end = 16.dp)
                                .size(24.dp),
                            tint = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.48f)
                        )
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 24.dp, vertical = 16.dp)
        ) {
            AnimatedContent(
                targetState = state,
                modifier = Modifier.fillMaxSize(),
                label = "stateTransition"
            ) { uiState ->
                when (uiState) {
                    is CryAnalyzerUiState.PermissionRequired -> PermissionView(onOpenSettings)
                    is CryAnalyzerUiState.Idle -> IdleView(
                        onRequestRecording = onRequestRecording,
                        onImportRecording = onImportRecording,
                        onPickSample = { showSamplePicker = true },
                        onClearSample = { onSelectSample(null) },
                        selectedSampleName = selectedSampleName,
                        onAnalyzeSample = selectedSampleIndex?.let { idx -> { onAnalyzeSample(idx) } }
                    )
                    is CryAnalyzerUiState.Listening -> ListeningView(uiState, onStopRecording)
                    is CryAnalyzerUiState.Processing -> ProcessingView()
                    is CryAnalyzerUiState.Completed -> ResultView(uiState, onReset)
                    is CryAnalyzerUiState.Error -> ErrorView(uiState.message, onReset)
                }
            }
        }
    }

    if (showPermissionDialog.value) {
        PermissionDialog(
            onOpenSettings = onOpenSettings,
            onDismiss = { showPermissionDialog.value = false }
        )
    }

    if (showSamplePicker) {
        AlertDialog(
            onDismissRequest = { showSamplePicker = false },
            title = { Text(stringResource(R.string.sample_picker_title)) },
            text = {
                Column {
                    Text(
                        text = stringResource(R.string.sample_picker_label),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(Modifier.size(12.dp))
                    LazyColumn(
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.heightIn(max = 320.dp)
                    ) {
                        items(CrySamples.samples) { sample ->
                            TextButton(
                                onClick = {
                                    onSelectSample(sample.index)
                                    showSamplePicker = false
                                }
                            ) {
                                Text(sample.displayName)
                            }
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showSamplePicker = false }) {
                    Text(stringResource(android.R.string.cancel))
                }
            }
        )
    }
}

@Composable
private fun PermissionView(onOpenSettings: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Microphone access needed",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.size(16.dp))
        Text(
            text = "Grant microphone permission so we can capture a short cry clip and analyze it entirely on your device.",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.size(24.dp))
        FilledTonalButton(onClick = onOpenSettings) {
            Text("Open Settings")
        }
    }
}

@Composable
private fun IdleView(
    onRequestRecording: () -> Unit,
    onImportRecording: () -> Unit,
    onPickSample: () -> Unit,
    onClearSample: () -> Unit,
    selectedSampleName: String?,
    onAnalyzeSample: (() -> Unit)?
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.size(16.dp))
        Text(
            text = "Tap to listen",
            style = MaterialTheme.typography.titleMedium.copy(
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        )
        MicButton(
            label = "Listen",
            onClick = onRequestRecording,
            background = MaterialTheme.colorScheme.primary,
            iconTint = MaterialTheme.colorScheme.onPrimary
        )
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.padding(bottom = 16.dp)
        ) {
            FilledTonalButton(onClick = onImportRecording) {
                Text(stringResource(R.string.sample_picker_import))
            }
            FilledTonalButton(onClick = onPickSample) {
                Text(stringResource(R.string.sample_picker_cta))
            }
            Text(
                text = selectedSampleName ?: stringResource(R.string.sample_picker_none),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (selectedSampleName != null) {
                TextButton(onClick = onClearSample) {
                    Text(stringResource(R.string.sample_picker_clear))
                }
            }
            if (onAnalyzeSample != null) {
                FilledTonalButton(onClick = onAnalyzeSample) {
                    Text(stringResource(R.string.sample_picker_analyze))
                }
            }
        }
        Spacer(Modifier.size(64.dp))
    }
}

@Composable
private fun ListeningView(
    state: CryAnalyzerUiState.Listening,
    onStopRecording: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.size(8.dp))
        Text(
            text = "Listening… ${(state.elapsedMillis / 1000.0).formatSeconds()}s",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = "Mic level ${(state.averageLevel * 100f).coerceIn(0f, 100f).roundToInt()}%",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        ListeningPulse(level = state.averageLevel)

        MicButton(
            label = "Stop",
            onClick = onStopRecording,
            background = MaterialTheme.colorScheme.errorContainer,
            iconTint = MaterialTheme.colorScheme.onErrorContainer,
            icon = Icons.Rounded.Stop
        )
        Spacer(Modifier.size(64.dp))
    }
}

@Composable
private fun ListeningPulse(level: Float) {
    val animatedLevel by animateFloatAsState(
        targetValue = level.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 250),
        label = "level"
    )
    val pulseColor = MaterialTheme.colorScheme.primary
    val micTint = MaterialTheme.colorScheme.onPrimary
    Box(
        modifier = Modifier
            .size(240.dp)
            .padding(8.dp),
        contentAlignment = Alignment.Center
    ) {
        val gradient = Brush.radialGradient(
            colors = listOf(
                pulseColor.copy(alpha = 0.25f),
                Color.Transparent
            )
        )
        Canvas(modifier = Modifier.fillMaxSize()) {
            val radius = size.minDimension / 2
            drawCircle(
                brush = gradient,
                radius = radius
            )
            drawCircle(
                color = pulseColor.copy(alpha = 0.35f + (animatedLevel * 0.4f)),
                radius = radius * (0.55f + (animatedLevel * 0.35f))
            )
        }
        Icon(
            imageVector = Icons.Outlined.Mic,
            contentDescription = null,
            tint = micTint,
            modifier = Modifier
                .size(72.dp)
                .background(
                    color = pulseColor,
                    shape = CircleShape
                )
                .padding(16.dp)
        )
    }
}

@Composable
private fun ProcessingView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Analyzing…",
            style = MaterialTheme.typography.headlineMedium
        )
        Spacer(Modifier.size(24.dp))
        PulsingProgress()
    }
}

@Composable
private fun PulsingProgress() {
    val transition = rememberInfiniteAnimation()
    val sweep by transition.animateFloat(
        initialValue = 40f,
        targetValue = 320f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1400),
            repeatMode = RepeatMode.Restart
        ),
        label = "progress"
    )
    val arcColor = MaterialTheme.colorScheme.primary
    Canvas(
        modifier = Modifier
            .size(96.dp)
            .padding(12.dp)
    ) {
        drawArc(
            color = arcColor,
            startAngle = sweep,
            sweepAngle = 120f,
            useCenter = false,
            topLeft = Offset(0f, 0f),
            size = size,
            style = Stroke(
                width = 12f,
                cap = StrokeCap.Round
            )
        )
    }
}

@Composable
private fun rememberInfiniteAnimation() =
    androidx.compose.animation.core.rememberInfiniteTransition(label = "infinite")

@Composable
private fun ResultView(
    state: CryAnalyzerUiState.Completed,
    onReset: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Analysis complete",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.size(4.dp))
        Text(
            text = "Recorded ${(state.capturedDurationMillis / 1000.0).formatSeconds()}s clip",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.size(16.dp))

        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 24.dp)
        ) {
            items(state.predictions) { prediction ->
                ModelResultCard(prediction)
            }
        }

        FilledTonalButton(
            onClick = onReset,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("New recording")
        }
        Spacer(Modifier.size(24.dp))
    }
}

@Composable
private fun ModelResultCard(prediction: ModelPrediction) {
    val badgeColor = when (prediction.modelId) {
        ModelId.RawAudio -> MaterialTheme.colorScheme.primary
        ModelId.Feature -> MaterialTheme.colorScheme.secondary
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = prediction.modelId.displayName,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(badgeColor.copy(alpha = 0.15f))
                        .padding(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = "${(prediction.confidence * 100).roundToInt()}%",
                        style = MaterialTheme.typography.labelLarge,
                        color = badgeColor
                    )
                }
            }

            Spacer(Modifier.size(12.dp))
            Text(
                text = prediction.topLabel.replaceFirstChar { it.titlecase() },
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.size(20.dp))
            Text(
                text = "Other possibilities",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.size(12.dp))

            for (probability in prediction.probabilities.take(5)) {
                ProbabilityRow(probability, badgeColor)
            }
        }
    }
}

@Composable
private fun ProbabilityRow(probability: ClassProbability, accent: Color) {
    val ratio = probability.probability.coerceIn(0f, 1f)
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = probability.label.replaceFirstChar { it.titlecase() },
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "${(ratio * 100).roundToInt()}%",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(Modifier.size(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(percent = 50))
                .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(ratio)
                    .height(6.dp)
                    .clip(RoundedCornerShape(percent = 50))
                    .background(
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                accent,
                                accent.copy(alpha = 0.6f)
                            )
                        )
                    )
            )
        }
    }
}

@Composable
private fun MicButton(
    label: String,
    onClick: () -> Unit,
    background: Color,
    iconTint: Color,
    icon: androidx.compose.ui.graphics.vector.ImageVector = Icons.Outlined.Mic
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .size(164.dp)
                .clip(CircleShape)
                .background(background)
                .clickable(onClick = onClick),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = iconTint,
                modifier = Modifier.size(64.dp)
            )
        }
        Spacer(Modifier.size(16.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

@Composable
private fun ErrorView(message: String, onReset: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Something went wrong",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.size(12.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.size(24.dp))
        FilledTonalButton(onClick = onReset) {
            Text("Try again")
        }
    }
}

@Composable
private fun PermissionDialog(
    onOpenSettings: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Microphone required") },
        text = {
            Text(
                "Allow microphone access so the app can capture short audio snippets and analyze them locally."
            )
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onDismiss()
                    onOpenSettings()
                }
            ) {
                Text("Open settings")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

private fun Double.formatSeconds(): String = if (this >= 1.0) {
    String.format("%.1f", this)
} else {
    String.format("%.2f", this)
}

