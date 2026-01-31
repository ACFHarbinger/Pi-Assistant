use crate::state::AppState;
use pi_core::agent_types::AgentCommand;
use tauri::State;
use tracing::{info, warn};

#[tauri::command]
pub async fn start_voice_listener(state: State<'_, AppState>) -> Result<(), String> {
    info!("Starting voice listener...");
    let mut voice_manager = state.voice_manager.lock().await;
    voice_manager
        .start_listening()
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn stop_voice_listener(state: State<'_, AppState>) -> Result<(), String> {
    info!("Stopping voice listener...");
    let mut voice_manager = state.voice_manager.lock().await;
    voice_manager.stop_listening().await;
    Ok(())
}

#[tauri::command]
pub async fn push_to_talk_start(state: State<'_, AppState>) -> Result<(), String> {
    info!("Push-to-talk: starting recording...");
    let mut voice_manager = state.voice_manager.lock().await;
    voice_manager
        .push_to_talk_start()
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn push_to_talk_stop(state: State<'_, AppState>) -> Result<String, String> {
    info!("Push-to-talk: stopping recording and transcribing...");

    // 1. Stop recording and get WAV path
    let wav_path = {
        let mut voice_manager = state.voice_manager.lock().await;
        voice_manager
            .push_to_talk_stop()
            .await
            .map_err(|e| e.to_string())?
    };

    let audio_path = wav_path.to_string_lossy().to_string();

    // 2. Transcribe via sidecar STT
    let transcription = {
        let mut sidecar = state.sidecar.lock().await;
        match sidecar
            .request(
                "voice.transcribe",
                serde_json::json!({
                    "audio_path": audio_path,
                    "model_size": "base",
                    "device": "cpu"
                }),
            )
            .await
        {
            Ok(result) => result
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            Err(e) => {
                warn!("STT transcription failed: {}", e);
                return Err(format!("Transcription failed: {}", e));
            }
        }
    };

    if transcription.is_empty() {
        return Err("No speech detected in recording".to_string());
    }

    info!("Push-to-talk transcription: {}", transcription);

    // 3. Send transcription to agent as a chat message
    state
        .agent_cmd_tx
        .send(AgentCommand::ChatMessage {
            content: transcription.clone(),
            provider: None,
            model_id: None,
        })
        .await
        .map_err(|e| e.to_string())?;

    Ok(transcription)
}

#[tauri::command]
pub async fn push_to_talk_cancel(state: State<'_, AppState>) -> Result<(), String> {
    info!("Push-to-talk: cancelling recording...");
    let mut voice_manager = state.voice_manager.lock().await;
    voice_manager
        .push_to_talk_cancel()
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
