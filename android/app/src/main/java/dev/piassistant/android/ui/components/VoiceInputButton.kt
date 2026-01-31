package dev.piassistant.android.ui.components

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import dev.piassistant.android.voice.SpeechRecognizerHelper

@Composable
fun VoiceInputButton(onResult: (String) -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val speechHelper = remember { SpeechRecognizerHelper(context) }
    val isListening by speechHelper.isListening.collectAsState()

    // Permission state
    var hasPermission by remember {
        mutableStateOf(
                ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                        PackageManager.PERMISSION_GRANTED
        )
    }

    val launcher =
            rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestPermission()
            ) { isGranted ->
                hasPermission = isGranted
                if (isGranted) {
                    speechHelper.startListening()
                }
            }

    // Set callback
    DisposableEffect(speechHelper) {
        speechHelper.onResult = onResult
        onDispose { speechHelper.destroy() }
    }

    val backgroundColor by
            animateColorAsState(
                    if (isListening) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primaryContainer,
                    label = "VoiceButtonColor"
            )

    Box(
            modifier =
                    modifier.size(48.dp).background(backgroundColor, CircleShape).pointerInput(
                                    Unit
                            ) {
                        detectTapGestures(
                                onPress = {
                                    if (!hasPermission) {
                                        launcher.launch(Manifest.permission.RECORD_AUDIO)
                                    } else {
                                        if (isListening) {
                                            speechHelper.stopListening()
                                        } else {
                                            speechHelper.startListening()
                                        }
                                    }
                                }
                        )
                    },
            contentAlignment = Alignment.Center
    ) {
        val iconId =
                if (isListening) android.R.drawable.ic_btn_speak_now
                else android.R.drawable.ic_btn_speak_now // Using system icon as placeholder
        // In a real app we'd use a propermic icon
        // For now let's just use text or a simple shape

        // Simulating an Icon (since we don't have non-system resources handy guaranteed)
        // using standard android resource for Mic if available, or just a generic one
        Icon(
                painter =
                        androidx.compose.ui.res.painterResource(
                                android.R.drawable.ic_btn_speak_now
                        ),
                contentDescription = "Voice Input",
                tint =
                        if (isListening) Color.White
                        else MaterialTheme.colorScheme.onPrimaryContainer
        )
    }
}
