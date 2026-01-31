//! Chat commands.

use crate::state::AppState;
use pi_core::agent_types::AgentCommand;
use tauri::State;

/// Send a message to the agent.
#[tauri::command]
pub async fn send_message(state: State<'_, AppState>, message: String) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::AnswerQuestion { response: message })
        .await
        .map_err(|e| format!("Failed to send message: {}", e))?;
    Ok(())
}

/// Get chat history (placeholder for now).
#[tauri::command]
pub fn get_history() -> Vec<serde_json::Value> {
    // TODO: Implement actual history retrieval from memory
    vec![]
}
