use crate::agent::r#loop::{spawn_agent_loop, AgentLoopHandle, AgentTask};
use crate::channels::{ChannelManager, ChannelResponse};
use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::PermissionEngine;
use crate::tools::ToolRegistry;
use pi_core::agent_types::{AgentCommand, AgentState};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex, RwLock};

use tracing::{info, warn};
use uuid::Uuid;

/// Monitor task that processes background agent commands and manages multiple agent loops.
pub async fn spawn_agent_coordinator(
    state_tx: watch::Sender<AgentState>,
    cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
    memory: Arc<MemoryManager>,
    ml_sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
    channel_manager: Arc<ChannelManager>,
    chat_session_id: Arc<RwLock<Uuid>>,
) {
    info!("Agent coordinator task started");

    let mut active_agents: HashMap<Uuid, AgentLoopHandle> = HashMap::new();

    loop {
        // Cleanup finished agents
        active_agents.retain(|_, handle| !handle.join_handle.is_finished());

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
                cost_config,
            }) => {
                let agent_id = Uuid::new_v4();
                info!(%agent_id, "Starting new agent task: {}", description);

                let task = AgentTask {
                    id: Uuid::new_v4(),
                    agent_id,
                    description,
                    max_iterations: max_iterations.unwrap_or(20),
                    session_id: Uuid::new_v4(), // TODO: Persistent session
                    provider: provider.unwrap_or_else(|| "local".to_string()),
                    model_id,
                    cost_config: cost_config.unwrap_or_default(),
                };

                let handle = spawn_agent_loop(
                    task,
                    state_tx.clone(),
                    tool_registry.clone(),
                    memory.clone(),
                    ml_sidecar.clone(),
                    permission_engine.clone(),
                );

                active_agents.insert(agent_id, handle);
            }
            Some(AgentCommand::Stop { agent_id }) => {
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.remove(&id) {
                        info!(%id, "Stopping agent loop");
                        handle.cancel_token.cancel();
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Stop { agent_id: Some(id) })
                            .await;
                    }
                } else {
                    // Stop all
                    info!("Stopping all agent loops");
                    for (_, handle) in active_agents.drain() {
                        handle.cancel_token.cancel();
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Stop { agent_id: None })
                            .await;
                    }
                }
            }
            Some(AgentCommand::Pause { agent_id }) => {
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.get(&id) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Pause { agent_id: Some(id) })
                            .await;
                    }
                } else {
                    // Pause all
                    for (_, handle) in &active_agents {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Pause { agent_id: None })
                            .await;
                    }
                }
            }
            Some(AgentCommand::Resume { agent_id }) => {
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.get(&id) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Resume { agent_id: Some(id) })
                            .await;
                    }
                } else {
                    // Resume all
                    for (_, handle) in &active_agents {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::Resume { agent_id: None })
                            .await;
                    }
                }
            }
            Some(AgentCommand::ChatMessage {
                content,
                provider,
                model_id,
                agent_id,
            }) => {
                // Route to specific agent if ID provided, else look for any running agent, else spawn one-off
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.get(&id) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::ChatMessage {
                                content: content.clone(),
                                provider: provider.clone(),
                                model_id: model_id.clone(),
                                agent_id: Some(id),
                            })
                            .await;
                    } else {
                        warn!(%id, "Agent not found for chat message");
                    }
                } else if let Some((id, handle)) = active_agents.iter().next() {
                    // Default to first available agent
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::ChatMessage {
                            content: content.clone(),
                            provider: provider.clone(),
                            model_id: model_id.clone(),
                            agent_id: Some(*id),
                        })
                        .await;
                } else {
                    // No agents running, respond directly (one-off)
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
                                Uuid::nil(),
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
            Some(AgentCommand::AnswerQuestion { agent_id, response }) => {
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.get(&id) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::AnswerQuestion {
                                agent_id: Some(id),
                                response,
                            })
                            .await;
                    }
                } else if let Some((id, handle)) = active_agents.iter().next() {
                    // Fallback to first
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::AnswerQuestion {
                            agent_id: Some(*id),
                            response,
                        })
                        .await;
                }
            }
            Some(AgentCommand::ApprovePermission {
                agent_id,
                request_id,
                approved,
                remember,
            }) => {
                if let Some(id) = agent_id {
                    if let Some(handle) = active_agents.get(&id) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::ApprovePermission {
                                agent_id: Some(id),
                                request_id,
                                approved,
                                remember,
                            })
                            .await;
                    }
                } else if let Some((id, handle)) = active_agents.iter().next() {
                    let _ = handle
                        .cmd_tx
                        .send(AgentCommand::ApprovePermission {
                            agent_id: Some(*id),
                            request_id,
                            approved,
                            remember,
                        })
                        .await;
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
                // Broadcast to all agents or route intelligently?
                // For now, if agents are running, broadcast. If not, one-off.
                if !active_agents.is_empty() {
                    for (_, handle) in &active_agents {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::ChannelMessage {
                                id: id.clone(),
                                channel: channel.clone(),
                                sender_id: sender_id.clone(),
                                sender_name: sender_name.clone(),
                                text: text.clone(),
                                chat_id: chat_id.clone(),
                                media: media.clone(),
                            })
                            .await;
                    }
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
                            .complete(Uuid::nil(), &text, None, None, None, Some(state_tx_inner))
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
                // Route to target agent
                if let Ok(to_uuid) = Uuid::parse_str(&to_agent_id) {
                    if let Some(handle) = active_agents.get(&to_uuid) {
                        let _ = handle
                            .cmd_tx
                            .send(AgentCommand::InternalMessage {
                                from_agent_id,
                                to_agent_id: to_agent_id.clone(),
                                content,
                            })
                            .await;
                    }
                }
            }
            None => {
                warn!("Agent command channel closed");
                break;
            }
        }
    }
}
