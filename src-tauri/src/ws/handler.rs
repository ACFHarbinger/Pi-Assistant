//! Per-connection WebSocket message handler.

use axum::extract::ws::{Message, WebSocket};
use futures::{stream::SplitSink, SinkExt, StreamExt};
use pi_core::agent_types::{AgentCommand, AgentState};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Incoming message from mobile client.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsClientMessage {
    #[serde(rename = "command")]
    Command(ClientCommand),
    #[serde(rename = "answer")]
    Answer { response: String },
    #[serde(rename = "permission_response")]
    PermissionResponse {
        request_id: Uuid,
        approved: bool,
        remember: bool,
    },
    #[serde(rename = "ping")]
    Ping,
}

/// Command from client.
#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub enum ClientCommand {
    #[serde(rename = "start")]
    Start {
        task: String,
        max_iterations: Option<u32>,
    },
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "pause")]
    Pause,
    #[serde(rename = "resume")]
    Resume,
}

/// Outgoing message to mobile client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsServerMessage {
    #[serde(rename = "agent_state")]
    AgentState(AgentState),
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "pong")]
    Pong,
}

/// WebSocket connection handler.
pub struct WsHandler {
    socket: WebSocket,
    agent_state_rx: watch::Receiver<AgentState>,
    agent_cmd_tx: mpsc::Sender<AgentCommand>,
}

impl WsHandler {
    pub fn new(
        socket: WebSocket,
        agent_state_rx: watch::Receiver<AgentState>,
        agent_cmd_tx: mpsc::Sender<AgentCommand>,
    ) -> Self {
        Self {
            socket,
            agent_state_rx,
            agent_cmd_tx,
        }
    }

    /// Run the handler loop.
    pub async fn run(self) -> anyhow::Result<()> {
        let (mut tx, mut rx) = self.socket.split();
        let mut state_rx = self.agent_state_rx.clone();
        let cmd_tx = self.agent_cmd_tx.clone();

        info!("Mobile client connected");

        // Send initial state
        let initial_state = state_rx.borrow().clone();
        send_message(&mut tx, WsServerMessage::AgentState(initial_state)).await?;

        loop {
            tokio::select! {
                // State updates from agent
                result = state_rx.changed() => {
                    if result.is_err() {
                        break; // Channel closed
                    }
                    let state = state_rx.borrow().clone();
                    if let Err(e) = send_message(&mut tx, WsServerMessage::AgentState(state)).await {
                        warn!(error = %e, "Failed to send state update");
                        break;
                    }
                }

                // Messages from client
                msg = rx.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            debug!(message = %text, "Received message");
                            match serde_json::from_str::<WsClientMessage>(&text) {
                                Ok(client_msg) => {
                                    if let Err(e) = handle_client_message(client_msg, &cmd_tx, &mut tx).await {
                                        let _ = send_message(&mut tx, WsServerMessage::Error {
                                            message: e.to_string(),
                                        }).await;
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "Failed to parse message");
                                    let _ = send_message(&mut tx, WsServerMessage::Error {
                                        message: format!("Invalid message: {}", e),
                                    }).await;
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("Client disconnected");
                            break;
                        }
                        Some(Ok(Message::Ping(data))) => {
                            let _ = tx.send(Message::Pong(data)).await;
                        }
                        Some(Err(e)) => {
                            warn!(error = %e, "WebSocket error");
                            break;
                        }
                        None => break,
                        _ => {}
                    }
                }
            }
        }

        info!("Mobile client disconnected");
        Ok(())
    }
}

async fn send_message(
    tx: &mut SplitSink<WebSocket, Message>,
    msg: WsServerMessage,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(&msg)?;
    tx.send(Message::Text(json.into())).await?;
    Ok(())
}

async fn handle_client_message(
    msg: WsClientMessage,
    cmd_tx: &mpsc::Sender<AgentCommand>,
    tx: &mut SplitSink<WebSocket, Message>,
) -> anyhow::Result<()> {
    match msg {
        WsClientMessage::Command(cmd) => {
            let agent_cmd = match cmd {
                ClientCommand::Start {
                    task,
                    max_iterations,
                } => AgentCommand::Start {
                    task,
                    max_iterations,
                    provider: None,
                    model_id: None,
                },
                ClientCommand::Stop => AgentCommand::Stop,
                ClientCommand::Pause => AgentCommand::Pause,
                ClientCommand::Resume => AgentCommand::Resume,
            };
            cmd_tx.send(agent_cmd).await?;
        }
        WsClientMessage::Answer { response } => {
            cmd_tx
                .send(AgentCommand::AnswerQuestion { response })
                .await?;
        }
        WsClientMessage::PermissionResponse {
            request_id,
            approved,
            remember,
        } => {
            cmd_tx
                .send(AgentCommand::ApprovePermission {
                    request_id,
                    approved,
                    remember,
                })
                .await?;
        }
        WsClientMessage::Ping => {
            send_message(tx, WsServerMessage::Pong).await?;
        }
    }
    Ok(())
}
