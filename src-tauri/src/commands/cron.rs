use crate::cron::CronJob;
use crate::state::AppState;
use anyhow::Result;
use tauri::State;
use uuid::Uuid;

#[tauri::command]
pub async fn get_cron_jobs(state: State<'_, AppState>) -> Result<Vec<CronJob>, String> {
    Ok(state.cron_manager.list_jobs().await)
}

#[tauri::command]
pub async fn add_cron_job(
    state: State<'_, AppState>,
    schedule: String,
    task_description: String,
) -> Result<Uuid, String> {
    state
        .cron_manager
        .add_job(schedule, task_description)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn remove_cron_job(state: State<'_, AppState>, id: Uuid) -> Result<(), String> {
    state
        .cron_manager
        .remove_job(id)
        .await
        .map_err(|e| e.to_string())
}
