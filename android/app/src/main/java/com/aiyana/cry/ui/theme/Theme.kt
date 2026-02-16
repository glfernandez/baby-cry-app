package com.aiyana.cry.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val ColorWhite = Color(0xFFF2F5FF)
private val ColorBlack = Color(0xFF020409)

private val DarkColors = darkColorScheme(
    primary = BrandPrimary,
    onPrimary = BrandOnPrimary,
    secondary = BrandSecondary,
    background = BrandBackground,
    surface = BrandSurface,
    surfaceVariant = BrandSurfaceVariant,
    outline = BrandOutline,
    onSurface = BrandOnSurface
)

private val LightColors = lightColorScheme(
    primary = BrandPrimary,
    onPrimary = BrandOnPrimary,
    secondary = BrandSecondary,
    background = ColorWhite,
    surface = ColorWhite,
    surfaceVariant = BrandSurfaceVariant,
    outline = BrandOutline,
    onSurface = ColorBlack
)

@Composable
fun PixelCryTheme(
    useDarkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val context = LocalContext.current
    val colorScheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            if (useDarkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        useDarkTheme -> DarkColors
        else -> LightColors
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val activity = view.context.findActivity()
            val window = activity.window
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !useDarkTheme
            window.statusBarColor = colorScheme.surface.toArgb()
            window.navigationBarColor = colorScheme.surface.toArgb()
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = PixelCryTypography,
        content = content
    )
}

private fun android.content.Context.findActivity(): android.app.Activity {
    var current = this
    while (current is android.content.ContextWrapper) {
        if (current is android.app.Activity) return current
        current = current.baseContext
    }
    throw IllegalStateException("Context does not contain an Activity")
}

