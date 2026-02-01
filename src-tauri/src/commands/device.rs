//! Tauri commands for device discovery, monitoring, and model migration.

use crate::state::AppState;
use serde_json::Value;
use tauri::State;

/// Get full hardware snapshot: CPU, GPUs, RAM, capabilities.
#[tauri::command]
pub async fn get_device_info(state: State<'_, AppState>) -> Result<Value, String> {
    let mut sidecar = state.ml_sidecar.lock().await;
    sidecar
        .request("device.info", serde_json::json!({}))
        .await
        .map_err(|e| e.to_string())
}

/// Re-poll memory usage for all devices (cheap, can call frequently).
#[tauri::command]
pub async fn refresh_device_memory(state: State<'_, AppState>) -> Result<Value, String> {
    let mut sidecar = state.ml_sidecar.lock().await;
    sidecar
        .request("device.refresh", serde_json::json!({}))
        .await
        .map_err(|e| e.to_string())
}

/// Move a loaded model between devices (e.g. CPU ↔ GPU).
#[tauri::command]
pub async fn migrate_model(
    state: State<'_, AppState>,
    model_id: String,
    target_device: String,
) -> Result<Value, String> {
    let mut sidecar = state.ml_sidecar.lock().await;
    sidecar
        .request(
            "model.migrate",
            serde_json::json!({
                "model_id": model_id,
                "target_device": target_device,
            }),
        )
        .await
        .map_err(|e| e.to_string())
}
