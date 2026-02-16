package com.aiyana.cry

import android.app.Application
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class CryAnalyzerApp : Application() {
    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onCreate() {
        super.onCreate()
        warmUpTensorFlow()
    }

    private fun warmUpTensorFlow() {
        appScope.launch {
            try {
                System.loadLibrary("tensorflowlite_jni")
                // Select TF Ops package loads its native libraries automatically when the interpreter is instantiated.
                Log.d("CryAnalyzerApp", "TensorFlow Lite libraries preloaded")
            } catch (t: Throwable) {
                Log.w("CryAnalyzerApp", "Failed to preload TensorFlow Lite native libs", t)
            }
        }
    }
}

