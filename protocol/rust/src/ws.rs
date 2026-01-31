//! WebSocket message types for Desktop <-> Mobile communication.

use serde::{Deserialize, Serialize};

/// WebSocket message envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl WsMessage {
    pub fn new(msg_type: impl Into<String>, payload: Option<serde_json::Value>) -> Self {
        Self {
            msg_type: msg_type.into(),
            payload,
        }
    }

    pub fn agent_state_update(state: serde_json::Value) -> Self {
        Self::new("agent_state_update", Some(state))
    }

    pub fn message(role: &str, content: &str) -> Self {
        Self::new(
            "message",
            Some(serde_json::json!({ "role": role, "content": content })),
        )
    }

    pub fn permission_request(id: &str, command: &str, description: &str) -> Self {
        Self::new(
            "permission_request",
            Some(serde_json::json!({
                "id": id,
                "command": command,
                "description": description,
            })),
        )
    }
}
