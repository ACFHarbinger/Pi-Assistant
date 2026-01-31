//! Pi-Assistant library - Tauri application setup and state management.

use tauri::Manager;

pub mod agent;
pub mod channels;
pub mod commands;
pub mod ipc;
pub mod mcp;
pub mod memory;
pub mod safety;
pub mod state;
pub mod tools;
pub mod ws;

/// Initialize and run the Tauri application.
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pi_assistant=debug,tauri=info".into()),
        )
        .init();

    tracing::info!("Starting Pi-Assistant");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            tauri::async_runtime::block_on(async {
                let state = state::AppState::new().await;
                app.manage(state);
            });
            tracing::info!("Application state initialized");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::agent::start_agent,
            commands::agent::stop_agent,
            commands::agent::pause_agent,
            commands::agent::resume_agent,
            commands::agent::get_agent_state,
            commands::chat::send_message,
            commands::chat::get_history,
            commands::config::get_mcp_config,
            commands::config::save_mcp_server,
            commands::config::remove_mcp_server,
            commands::config::get_tools_config,
            commands::config::toggle_tool,
            commands::config::get_models_config,
            commands::config::save_model,
            commands::config::load_model,
            commands::config::get_mcp_marketplace,
            commands::config::reset_agent,
            commands::sidecar::sidecar_request,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
