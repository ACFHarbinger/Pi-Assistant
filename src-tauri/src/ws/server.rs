//! Axum WebSocket server for mobile clients.

use super::handler::WsHandler;
use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use pi_core::agent_types::{AgentCommand, AgentState};
use serde::Deserialize;
use std::net::SocketAddr;
use tokio::sync::{mpsc, watch};
use tracing::{error, info};

#[derive(Deserialize)]
pub struct WebhookPayload {
    pub task: String,
    pub provider: Option<String>,
    pub model_id: Option<String>,
    #[serde(default)]
    pub is_chat: bool,
}

/// WebSocket server state.
pub struct WebSocketServer {
    port: u16,
    agent_state_rx: watch::Receiver<AgentState>,
    agent_cmd_tx: mpsc::Sender<AgentCommand>,
}

/// Shared state for axum handlers.
#[derive(Clone)]
pub struct WsState {
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
}

impl WebSocketServer {
    /// Create a new WebSocket server.
    pub fn new(
        port: u16,
        agent_state_rx: watch::Receiver<AgentState>,
        agent_cmd_tx: mpsc::Sender<AgentCommand>,
    ) -> Self {
        Self {
            port,
            agent_state_rx,
            agent_cmd_tx,
        }
    }

    /// Start the WebSocket server.
    pub async fn start(self) -> anyhow::Result<()> {
        let state = WsState {
            agent_state_rx: self.agent_state_rx,
            agent_cmd_tx: self.agent_cmd_tx,
        };

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .route("/health", get(health_handler))
            .route("/webhook", post(webhook_handler))
            .with_state(state);

        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        info!(port = self.port, "Starting WebSocket server");

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Spawn the server as a background task.
    pub fn spawn(self) -> tokio::task::JoinHandle<anyhow::Result<()>> {
        tokio::spawn(async move { self.start().await })
    }
}

/// Health check endpoint.
async fn health_handler() -> impl IntoResponse {
    "OK"
}

/// WebSocket upgrade handler.
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<WsState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle a WebSocket connection.
async fn handle_socket(socket: WebSocket, state: WsState) {
    let handler = WsHandler::new(socket, state.agent_state_rx, state.agent_cmd_tx);

    if let Err(e) = handler.run().await {
        error!(error = %e, "WebSocket handler error");
    }
}

/// Webhook handler to trigger agent commands via HTTP POST.
async fn webhook_handler(
    State(state): State<WsState>,
    Json(payload): Json<WebhookPayload>,
) -> impl IntoResponse {
    let cmd = if payload.is_chat {
        AgentCommand::ChatMessage {
            content: payload.task,
            provider: payload.provider,
            model_id: payload.model_id,
        }
    } else {
        AgentCommand::Start {
            task: payload.task,
            max_iterations: None,
            provider: payload.provider,
            model_id: payload.model_id,
        }
    };

    if let Err(e) = state.agent_cmd_tx.send(cmd).await {
        error!("Failed to send agent command from webhook: {}", e);
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to trigger agent",
        )
            .into_response();
    }

    (axum::http::StatusCode::OK, "Agent triggered").into_response()
}
