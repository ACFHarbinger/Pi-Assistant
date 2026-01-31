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
    // Logic Sidecar handles: "health.", "personality.", "lifecycle."
    // ML Sidecar handles: "inference.", "model.", "training.", "voice."

    let is_logic = method.starts_with("health.")
        || method.starts_with("personality.")
        || method.starts_with("lifecycle.");

    if is_logic {
        let mut sidecar = state.logic_sidecar.lock().await;
        let result = sidecar
            .request(&method, params)
            .await
            .map_err(|e| format!("Logic Sidecar request failed: {}", e))?;
        Ok(result)
    } else {
        let mut sidecar = state.ml_sidecar.lock().await;
        let result = sidecar
            .request(&method, params)
            .await
            .map_err(|e| format!("ML Sidecar request failed: {}", e))?;
        Ok(result)
    }
}
