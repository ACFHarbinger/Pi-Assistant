package dev.piassistant.android.network

import android.util.Log
import dev.piassistant.android.data.AgentState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

class WebSocketClient(
    private val serverUrl: String = "ws://10.0.2.2:9120/ws" // Android Emulator loopback
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null

    private val _agentState = MutableStateFlow<AgentState?>(null)
    val agentState: StateFlow<AgentState?> = _agentState.asStateFlow()

    private val _connectionState = MutableStateFlow(false)
    val connectionState: StateFlow<Boolean> = _connectionState.asStateFlow()

    private val json = Json { ignoreUnknownKeys = true }

    fun connect() {
        val request = Request.Builder().url(serverUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "Connected to WebSocket")
                _connectionState.value = true
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "Received message: $text")
                try {
                    val message = json.decodeFromString<WsServerMessage>(text)
                    when (message) {
                        is WsServerMessage.AgentStateMsg -> {
                            _agentState.value = message.payload
                        }
                        is WsServerMessage.ErrorMsg -> {
                            Log.e(TAG, "Server error: ${message.payload.message}")
                        }
                        is WsServerMessage.Pong -> {
                            // Heartbeat response
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to parse message", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "Closing: $code / $reason")
                _connectionState.value = false
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure", t)
                _connectionState.value = false
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "User disconnected")
        webSocket = null
    }

    fun sendCommand(command: ClientCommand) {
        val msg = WsClientMessage.Command(command)
        send(msg)
    }

    fun sendAnswer(response: String) {
        val msg = WsClientMessage.Answer(response)
        send(msg)
    }
    
    fun sendPermissionResponse(requestId: String, approved: Boolean, remember: Boolean) {
        val msg = WsClientMessage.PermissionResponse(requestId, approved, remember)
        send(msg)
    }

    private fun send(msg: WsClientMessage) {
        val text = json.encodeToString(msg)
        webSocket?.send(text)
    }

    companion object {
        private const val TAG = "WebSocketClient"
    }
}

// ── Wire Types ───────────────────────────────────────────────────────

@Serializable
@SerialName("WsServerMessage")
sealed class WsServerMessage {
    @Serializable @SerialName("agent_state")
    data class AgentStateMsg(val payload: AgentState) : WsServerMessage()
    
    @Serializable @SerialName("error")
    data class ErrorMsg(val payload: ErrorPayload) : WsServerMessage()
    
    @Serializable @SerialName("pong")
    data object Pong : WsServerMessage()
}

@Serializable
data class ErrorPayload(val message: String)

@Serializable
@SerialName("WsClientMessage")
sealed class WsClientMessage {
    @Serializable @SerialName("command")
    data class Command(val payload: ClientCommand) : WsClientMessage()
    
    @Serializable @SerialName("answer")
    data class Answer(val response: String) : WsClientMessage()
    
    @Serializable @SerialName("permission_response")
    data class PermissionResponse(
        val request_id: String,
        val approved: Boolean,
        val remember: Boolean
    ) : WsClientMessage()
    
    @Serializable @SerialName("ping")
    data object Ping : WsClientMessage()
}

@Serializable
sealed class ClientCommand {
    @Serializable @SerialName("start")
    data class Start(val task: String, val max_iterations: UInt? = null) : ClientCommand()
    
    @Serializable @SerialName("stop")
    data object Stop : ClientCommand()
    
    @Serializable @SerialName("pause")
    data object Pause : ClientCommand()
    
    @Serializable @SerialName("resume")
    data object Resume : ClientCommand()
}
