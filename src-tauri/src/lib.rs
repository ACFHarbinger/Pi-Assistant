//! Pi-Assistant library - Tauri application setup and state management.

use tauri::Emitter;
use tauri::Manager;

pub mod agent;
pub mod channels;
pub mod commands;
pub mod cron;
pub mod ipc;
pub mod mcp;
pub mod memory;
pub mod safety;
pub mod skills;
pub mod state;
pub mod tools;
pub mod voice;
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
                let sidecar = state.sidecar.clone();
                let tool_registry = state.tool_registry.clone();
                let memory = state.memory.clone();
                let permissions = state.permissions.clone();
                let agent_cmd_rx = state.agent_cmd_rx.clone();
                let agent_state_tx = state.agent_state_tx.clone();
                let channel_manager = state.channel_manager.clone();

                tauri::async_runtime::spawn(async move {
                    agent::spawn_agent_monitor(
                        agent_state_tx,
                        agent_cmd_rx,
                        tool_registry,
                        memory,
                        sidecar,
                        permissions,
                        channel_manager,
                    )
                    .await;
                });

                // Register canvas tool (requires app handle)
                {
                    let mut registry = state.tool_registry.write().await;
                    registry.register_canvas_tool(app.handle().clone());
                }

                // Bridge AgentState to Tauri events
                let mut state_rx = state.agent_state_rx.clone();
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    while state_rx.changed().await.is_ok() {
                        let state = state_rx.borrow().clone();
                        let _ = app_handle.emit("agent-state-changed", state);
                    }
                });

                // Start WebSocket/Webhook server
                let ws_server = ws::WebSocketServer::new(
                    8080,
                    state.agent_state_rx.clone(),
                    state.agent_cmd_tx.clone(),
                );
                ws_server.spawn();

                // Initialize channels from config
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    // Telegram
                    if let Ok(tg_config) = commands::config::get_telegram_config().await {
                        if tg_config.enabled {
                            if let Some(token) = tg_config.token {
                                let state = app_handle.state::<state::AppState>();
                                let channel =
                                    Box::new(crate::channels::telegram::TelegramChannel::new(
                                        token,
                                        state.agent_cmd_tx.clone(),
                                    ));
                                for user in tg_config.allowed_users {
                                    channel.allow_user(user).await;
                                }
                                state.channel_manager.add_channel(channel).await;
                                let _ = state.channel_manager.start_channel("telegram").await;
                            }
                        }
                    }
                    // Discord
                    if let Ok(discord_config) = commands::config::get_discord_config().await {
                        if discord_config.enabled {
                            if let Some(token) = discord_config.token {
                                let state = app_handle.state::<state::AppState>();
                                let channel =
                                    Box::new(crate::channels::discord::DiscordChannel::new(
                                        token,
                                        state.agent_cmd_tx.clone(),
                                    ));
                                state.channel_manager.add_channel(channel).await;
                                let _ = state.channel_manager.start_channel("discord").await;
                            }
                        }
                    }
                });

                app.manage(state);
            });
            tracing::info!("Application state and agent monitor initialized");
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
            commands::config::save_api_key,
            commands::config::save_agent_name,
            commands::config::get_current_model,
            commands::config::save_current_model,
            commands::config::get_telegram_config,
            commands::config::save_telegram_config,
            commands::config::get_discord_config,
            commands::config::save_discord_config,
            commands::cron::get_cron_jobs,
            commands::cron::add_cron_job,
            commands::cron::remove_cron_job,
            commands::voice::start_voice_listener,
            commands::voice::stop_voice_listener,
            commands::auth::start_oauth,
            commands::auth::exchange_oauth_code,
            commands::auth::start_claude_oauth,
            commands::auth::refresh_claude_token,
            commands::auth::check_claude_auth,
            commands::auth::disconnect_claude_auth,
            commands::auth::check_provider_auth,
            commands::auth::disconnect_provider_auth,
            commands::sidecar::sidecar_request,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
