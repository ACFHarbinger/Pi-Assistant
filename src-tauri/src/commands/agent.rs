//! Agent control commands.

use crate::state::AppState;
use pi_core::agent_types::{AgentCommand, AgentState};
use tauri::State;

/// Start the agent with a task.
#[tauri::command]
pub async fn start_agent(
    state: State<'_, AppState>,
    task: String,
    max_iterations: Option<u32>,
    provider: Option<String>,
    model_id: Option<String>,
) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::Start {
            task,
            max_iterations,
            provider,
            model_id,
        })
        .await
        .map_err(|e| format!("Failed to send start command: {}", e))?;
    Ok(())
}

/// Stop the running agent.
#[tauri::command]
pub async fn stop_agent(state: State<'_, AppState>) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::Stop)
        .await
        .map_err(|e| format!("Failed to send stop command: {}", e))?;
    Ok(())
}

/// Pause the running agent.
#[tauri::command]
pub async fn pause_agent(state: State<'_, AppState>) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::Pause)
        .await
        .map_err(|e| format!("Failed to send pause command: {}", e))?;
    Ok(())
}

/// Resume a paused agent.
#[tauri::command]
pub async fn resume_agent(state: State<'_, AppState>) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::Resume)
        .await
        .map_err(|e| format!("Failed to send resume command: {}", e))?;
    Ok(())
}

/// Get the current agent state.
#[tauri::command]
pub fn get_agent_state(state: State<'_, AppState>) -> AgentState {
    state.agent_state_rx.borrow().clone()
}
