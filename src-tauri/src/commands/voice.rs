use crate::state::AppState;
use tauri::State;
use tracing::info;

#[tauri::command]
pub async fn start_voice_listener(state: State<'_, AppState>) -> Result<(), String> {
    info!("Starting voice listener...");
    let voice_manager = state.voice_manager.lock().await;
    voice_manager
        .start_listening()
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn stop_voice_listener(_state: State<'_, AppState>) -> Result<(), String> {
    info!("Stopping voice listener (NOT YET FULLY IMPLEMENTED)");
    // In a full implementation, we would stop the background task
    Ok(())
}
