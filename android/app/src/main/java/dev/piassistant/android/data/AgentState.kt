package dev.piassistant.android.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class StopReason {
    Completed,
    ManualStop,
    Error,
    IterationLimit
}

@Serializable
data class PermissionRequest(
    val id: String,
    val tool_name: String,
    val command: String,
    val tier: String,
    val description: String
)

@Serializable
@SerialName("data") // Used for the content of the state
data class StateData(
    val task_id: String? = null,
    val iteration: UInt? = null,
    val question: String? = null,
    val awaiting_permission: PermissionRequest? = null,
    val reason: StopReason? = null // For Stopped state
)

@Serializable
data class AgentState(
    val status: String, // "Idle", "Running", "Paused", "Stopped"
    val data: StateData? = null
)

@Serializable
data class Message(
    val role: String, // "user", "assistant", "system"
    val content: String,
    val timestamp: Long
)
