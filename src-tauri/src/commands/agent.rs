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
    cost_config: Option<pi_core::agent_types::CostConfig>,
) -> Result<(), String> {
    state
        .agent_cmd_tx
        .send(AgentCommand::Start {
            task,
            max_iterations,
            provider,
            model_id,
            cost_config,
        })
        .await
        .map_err(|e| format!("Failed to send start command: {}", e))?;
    Ok(())
}

/// Stop the running agent.
#[tauri::command]
pub async fn stop_agent(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<(), String> {
    let id = agent_id
        .map(|s| uuid::Uuid::parse_str(&s).map_err(|e| e.to_string()))
        .transpose()?;
    state
        .agent_cmd_tx
        .send(AgentCommand::Stop { agent_id: id })
        .await
        .map_err(|e| format!("Failed to send stop command: {}", e))?;
    Ok(())
}

/// Pause the running agent.
#[tauri::command]
pub async fn pause_agent(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<(), String> {
    let id = agent_id
        .map(|s| uuid::Uuid::parse_str(&s).map_err(|e| e.to_string()))
        .transpose()?;
    state
        .agent_cmd_tx
        .send(AgentCommand::Pause { agent_id: id })
        .await
        .map_err(|e| format!("Failed to send pause command: {}", e))?;
    Ok(())
}

/// Resume a paused agent.
#[tauri::command]
pub async fn resume_agent(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<(), String> {
    let id = agent_id
        .map(|s| uuid::Uuid::parse_str(&s).map_err(|e| e.to_string()))
        .transpose()?;
    state
        .agent_cmd_tx
        .send(AgentCommand::Resume { agent_id: id })
        .await
        .map_err(|e| format!("Failed to send resume command: {}", e))?;
    Ok(())
}

/// Answer a question from the agent.
#[tauri::command]
pub async fn answer_question(
    state: State<'_, AppState>,
    answer: String,
    agent_id: Option<String>,
) -> Result<(), String> {
    let id = agent_id
        .map(|s| uuid::Uuid::parse_str(&s).map_err(|e| e.to_string()))
        .transpose()?;
    state
        .agent_cmd_tx
        .send(AgentCommand::AnswerQuestion {
            agent_id: id,
            response: answer,
        })
        .await
        .map_err(|e| format!("Failed to send answer: {}", e))?;
    Ok(())
}

/// Get the current agent state.
#[tauri::command]
pub fn get_agent_state(state: State<'_, AppState>) -> AgentState {
    state.agent_state_rx.borrow().clone()
}

/// Get the execution timeline for a task.
#[tauri::command]
pub fn get_execution_timeline(
    state: State<'_, AppState>,
    task_id: String,
) -> Result<Vec<serde_json::Value>, String> {
    let uuid = uuid::Uuid::parse_str(&task_id).map_err(|e| e.to_string())?;
    state
        .memory
        .get_execution_timeline(&uuid)
        .map_err(|e| e.to_string())
}
