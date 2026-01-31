//! Command for communicating with the Python sidecar.

use crate::state::AppState;
use serde_json::Value;
use tauri::State;

#[tauri::command]
pub async fn sidecar_request(
    state: State<'_, AppState>,
    method: String,
    params: Value,
) -> Result<Value, String> {
    tracing::debug!("Sidecar request: method={}, params={}", method, params);

    // Dispatch to sidecar
    let mut sidecar = state.sidecar.lock().await;
    let result = sidecar
        .request(&method, params)
        .await
        .map_err(|e| format!("Sidecar request failed: {}", e))?;

    Ok(result)
}
