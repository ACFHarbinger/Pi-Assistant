package dev.piassistant.android.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.piassistant.android.data.AgentState
import dev.piassistant.android.data.Message
import dev.piassistant.android.network.ClientCommand
import dev.piassistant.android.network.WebSocketClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ChatViewModel : ViewModel() {
    private val webSocketClient = WebSocketClient()

    val agentState: StateFlow<AgentState?> = webSocketClient.agentState
    val isConnected: StateFlow<Boolean> = webSocketClient.connectionState

    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    init {
        connect()
    }

    fun connect() {
        webSocketClient.connect()
    }

    fun disconnect() {
        webSocketClient.disconnect()
    }

    fun startTask(task: String) {
        webSocketClient.sendCommand(ClientCommand.Start(task))
        addMessage("user", task)
    }

    fun stopAgent() {
        webSocketClient.sendCommand(ClientCommand.Stop)
    }

    fun pauseAgent() {
        webSocketClient.sendCommand(ClientCommand.Pause)
    }

    fun resumeAgent() {
        webSocketClient.sendCommand(ClientCommand.Resume)
    }

    fun sendAnswer(response: String) {
        webSocketClient.sendAnswer(response)
        addMessage("user", response)
    }

    fun approvePermission(requestId: String, remember: Boolean) {
        webSocketClient.sendPermissionResponse(requestId, approved = true, remember = remember)
    }

    fun denyPermission(requestId: String) {
        webSocketClient.sendPermissionResponse(requestId, approved = false, remember = false)
    }

    private fun addMessage(role: String, content: String) {
        val msg = Message(role, content, System.currentTimeMillis())
        _messages.value += msg
    }

    override fun onCleared() {
        super.onCleared()
        disconnect()
    }
}
