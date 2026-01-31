//! Axum WebSocket server for mobile clients.

use super::handler::WsHandler;
use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use pi_core::agent_types::{AgentCommand, AgentState};
use std::net::SocketAddr;
use tokio::sync::{mpsc, watch};
use tracing::{error, info};

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
