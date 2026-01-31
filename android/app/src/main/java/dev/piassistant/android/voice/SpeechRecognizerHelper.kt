package dev.piassistant.android.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class SpeechRecognizerHelper(context: Context) {
    private val speechRecognizer: SpeechRecognizer =
            SpeechRecognizer.createSpeechRecognizer(context)
    private val recognizerIntent: Intent =
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(
                        RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                        RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
                )
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            }

    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _text = MutableStateFlow("")
    val text: StateFlow<String> = _text.asStateFlow()

    // Callback for final results
    var onResult: ((String) -> Unit)? = null

    init {
        speechRecognizer.setRecognitionListener(
                object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        Log.d(TAG, "Ready for speech")
                    }

                    override fun onBeginningOfSpeech() {
                        Log.d(TAG, "Beginning of speech")
                        _isListening.value = true
                        _text.value = ""
                    }

                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}

                    override fun onEndOfSpeech() {
                        Log.d(TAG, "End of speech")
                        _isListening.value = false
                    }

                    override fun onError(error: Int) {
                        Log.e(TAG, "Speech error: $error")
                        _isListening.value = false
                    }

                    override fun onResults(results: Bundle?) {
                        val matches =
                                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val result = matches?.firstOrNull() ?: ""
                        Log.d(TAG, "Results: $result")
                        _text.value = result
                        if (result.isNotBlank()) {
                            onResult?.invoke(result)
                        }
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        val matches =
                                partialResults?.getStringArrayList(
                                        SpeechRecognizer.RESULTS_RECOGNITION
                                )
                        val result = matches?.firstOrNull() ?: ""
                        if (result.isNotBlank()) {
                            _text.value = result
                        }
                    }

                    override fun onEvent(eventType: Int, params: Bundle?) {}
                }
        )
    }

    fun startListening() {
        try {
            speechRecognizer.startListening(recognizerIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Start listening failed", e)
        }
    }

    fun stopListening() {
        try {
            speechRecognizer.stopListening()
        } catch (e: Exception) {
            Log.e(TAG, "Stop listening failed", e)
        }
    }

    fun destroy() {
        speechRecognizer.destroy()
    }

    companion object {
        private const val TAG = "SpeechRecognizer"
    }
}
