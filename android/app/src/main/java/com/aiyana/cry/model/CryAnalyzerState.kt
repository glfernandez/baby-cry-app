package com.aiyana.cry.model

data class ClassProbability(
    val label: String,
    val probability: Float
)

enum class ModelId(val displayName: String) {
    RawAudio("Raw Audio Model"),
    Feature("Feature Model")
}

data class ModelPrediction(
    val modelId: ModelId,
    val topLabel: String,
    val confidence: Float,
    val probabilities: List<ClassProbability>
) {
    init {
        require(probabilities.isNotEmpty()) { "probabilities must not be empty" }
    }
}

sealed interface CryAnalyzerUiState {
    data object PermissionRequired : CryAnalyzerUiState
    data class Idle(
        val ready: Boolean = true,
        val selectedSample: Int? = null
    ) : CryAnalyzerUiState
    data class Listening(
        val elapsedMillis: Long,
        val averageLevel: Float
    ) : CryAnalyzerUiState

    data object Processing : CryAnalyzerUiState
    data class Completed(
        val predictions: List<ModelPrediction>,
        val capturedDurationMillis: Long
    ) : CryAnalyzerUiState

    data class Error(val message: String) : CryAnalyzerUiState
}

