//! Chat commands.

use crate::state::AppState;
use pi_core::agent_types::AgentCommand;
use tauri::State;

/// Send a message to the agent.
#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    message: String,
    provider: Option<String>,
    model_id: Option<String>,
) -> Result<(), String> {
    let session_id = *state.chat_session_id.read().await;
    // Store user message immediately
    let _ = state
        .memory
        .store_message(&session_id, "user", &message)
        .await;

    state
        .agent_cmd_tx
        .send(AgentCommand::ChatMessage {
            agent_id: None,
            content: message,
            provider,
            model_id,
        })
        .await
        .map_err(|e| format!("Failed to send message: {}", e))?;
    Ok(())
}

/// Get chat history.
#[tauri::command]
pub async fn get_history(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
    let session_id = *state.chat_session_id.read().await;
    let messages = state
        .memory
        .get_recent_messages(&session_id, 100)
        .map_err(|e| e.to_string())?;

    let history = messages
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at,
            })
        })
        .collect();

    Ok(history)
}
