use crate::agent::r#loop::{spawn_agent_loop, AgentLoopHandle, AgentTask};
use crate::channels::{ChannelManager, ChannelResponse};
use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::PermissionEngine;
use crate::tools::ToolRegistry;
use pi_core::agent_types::{AgentCommand, AgentState};

use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

/// Monitor task that processes background agent commands.
pub async fn spawn_agent_monitor(
    state_tx: watch::Sender<AgentState>,
    cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
    memory: Arc<MemoryManager>,
    ml_sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
    channel_manager: Arc<ChannelManager>,
    chat_session_id: Arc<RwLock<Uuid>>,
) {
    info!("Agent monitor task started");

    let mut current_loop: Option<AgentLoopHandle> = None;

    loop {
        let cmd: Option<AgentCommand> = {
            let mut rx = cmd_rx.lock().await;
            rx.recv().await
        };

        match cmd {
            Some(AgentCommand::Start {
                task: description,
                max_iterations,
                provider,
                model_id,
            }) => {
                if let Some(ref handle) = current_loop {
                    if !handle.join_handle.is_finished() {
                        warn!("Agent loop already running, ignoring start command");
                        continue;
                    }
                }

                info!("Starting agent task: {}", description);
                let task = AgentTask {
                    id: Uuid::new_v4(),
                    description,
                    max_iterations: max_iterations.unwrap_or(20),
                    session_id: Uuid::new_v4(), // TODO: Persistent session
                    provider: provider.unwrap_or_else(|| "local".to_string()),
                    model_id,
                };

                let handle = spawn_agent_loop(
                    task,
                    state_tx.clone(),
                    tool_registry.clone(),
                    memory.clone(),
                    ml_sidecar.clone(),
                    permission_engine.clone(),
                );

                current_loop = Some(handle);
            }
            Some(AgentCommand::Stop) => {
                if let Some(handle) = current_loop.take() {
                    info!("Stopping agent loop");
                    handle.cancel_token.cancel();
                    // Also send Stop command to break out of wait_for_* blocks
                    let _ = handle.cmd_tx.send(AgentCommand::Stop).await;
                }
            }
            Some(AgentCommand::ChatMessage {
                content,
                provider,
                model_id,
            }) => {
                if let Some(ref handle) = current_loop {
                    // Try to send to loop first (it might be waiting for AnswerQuestion,
                    // but we'll treat ChatMessage as AnswerQuestion too if it fits)
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::AnswerQuestion { response: content })
                        .await;
                } else {
                    info!("Received chat message while idle (provider: {:?}, model: {:?}), responding directly...", provider, model_id);
                    let state_tx_inner = state_tx.clone();
                    let sidecar = ml_sidecar.clone();
                    let tool_registry = tool_registry.clone();
                    let memory = memory.clone();
                    let chat_session_id = chat_session_id.clone();

                    tauri::async_runtime::spawn(async move {
                        let planner =
                            crate::agent::planner::AgentPlanner::new(sidecar, tool_registry);
                        match planner
                            .complete(
                                &content,
                                provider.as_deref(),
                                model_id.as_deref(),
                                None,
                                Some(state_tx_inner),
                            )
                            .await
                        {
                            Ok(response) => {
                                // Store user message
                                let session_id = *chat_session_id.read().await;
                                let _ = memory.store_message(&session_id, "user", &content).await;
                                // Store assistant message
                                let _ = memory
                                    .store_message(&session_id, "assistant", &response)
                                    .await;
                            }
                            Err(e) => {
                                warn!("Failed to generate chat response: {}", e);
                            }
                        }
                    });
                }
            }
            Some(AgentCommand::AnswerQuestion { response }) => {
                if let Some(ref handle) = current_loop {
                    if let Err(e) = handle
                        .cmd_tx
                        .send(AgentCommand::AnswerQuestion { response })
                        .await
                    {
                        warn!("Failed to relay answer to agent loop: {}", e);
                    }
                }
            }
            Some(AgentCommand::Pause) => {
                if let Some(ref handle) = current_loop {
                    if let Err(e) = handle.cmd_tx.send(AgentCommand::Pause).await {
                        warn!("Failed to relay pause to agent loop: {}", e);
                    }
                }
            }
            Some(AgentCommand::Resume) => {
                if let Some(ref handle) = current_loop {
                    if let Err(e) = handle.cmd_tx.send(AgentCommand::Resume).await {
                        warn!("Failed to relay resume to agent loop: {}", e);
                    }
                }
            }
            Some(AgentCommand::ApprovePermission {
                request_id,
                approved,
                remember,
            }) => {
                if let Some(ref handle) = current_loop {
                    if let Err(e) = handle
                        .cmd_tx
                        .send(AgentCommand::ApprovePermission {
                            request_id,
                            approved,
                            remember,
                        })
                        .await
                    {
                        warn!("Failed to relay permission to agent loop: {}", e);
                    }
                }
            }
            Some(AgentCommand::ChannelMessage {
                id,
                channel,
                sender_id,
                sender_name,
                text,
                chat_id,
                media,
            }) => {
                if let Some(ref handle) = current_loop {
                    // Forward to running loop
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::ChannelMessage {
                            id,
                            channel,
                            sender_id,
                            sender_name,
                            text,
                            chat_id,
                            media,
                        })
                        .await;
                } else {
                    info!(
                        "Received channel message ({}) from {} while idle, responding...",
                        channel, sender_id
                    );
                    let sidecar = ml_sidecar.clone();
                    let tool_registry = tool_registry.clone();
                    let memory = memory.clone();
                    let channel_manager = channel_manager.clone();

                    let state_tx_inner = state_tx.clone();
                    tauri::async_runtime::spawn(async move {
                        let planner =
                            crate::agent::planner::AgentPlanner::new(sidecar, tool_registry);
                        match planner
                            .complete(&text, None, None, None, Some(state_tx_inner))
                            .await
                        {
                            Ok(response) => {
                                // Store message
                                let session_id = Uuid::new_v4();
                                let _ = memory.store_message(&session_id, "user", &text).await;
                                let _ = memory
                                    .store_message(&session_id, "assistant", &response)
                                    .await;

                                // Reply to channel
                                let _ = channel_manager
                                    .send_response(
                                        &channel,
                                        ChannelResponse {
                                            chat_id,
                                            text: response,
                                            reply_to: Some(id),
                                        },
                                    )
                                    .await;
                            }
                            Err(e) => {
                                warn!("Failed to generate channel response: {}", e);
                            }
                        }
                    });
                }
            }
            Some(AgentCommand::InternalMessage {
                from_agent_id,
                to_agent_id,
                content,
            }) => {
                if let Some(ref handle) = current_loop {
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::InternalMessage {
                            from_agent_id,
                            to_agent_id,
                            content,
                        })
                        .await;
                }
            }
            None => {
                warn!("Agent command channel closed");
                break;
            }
        }
    }
}
