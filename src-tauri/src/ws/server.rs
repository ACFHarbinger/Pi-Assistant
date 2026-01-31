//! Axum WebSocket server for mobile clients.

use super::handler::WsHandler;
use axum::{
    body::Bytes,
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use hmac::{Hmac, Mac};
use pi_core::agent_types::{AgentCommand, AgentState};
use serde::Deserialize;
use sha2::Sha256;
use std::net::SocketAddr;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

type HmacSha256 = Hmac<Sha256>;

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
    webhook_secret: Option<String>,
}

/// Shared state for axum handlers.
#[derive(Clone)]
pub struct WsState {
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    /// Optional webhook secret for signature verification
    pub webhook_secret: Option<String>,
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
            webhook_secret: None,
        }
    }

    /// Set the webhook secret for signature verification.
    pub fn with_webhook_secret(mut self, secret: Option<String>) -> Self {
        self.webhook_secret = secret;
        self
    }

    /// Start the WebSocket server.
    pub async fn start(self) -> anyhow::Result<()> {
        let state = WsState {
            agent_state_rx: self.agent_state_rx,
            agent_cmd_tx: self.agent_cmd_tx,
            webhook_secret: self.webhook_secret,
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

/// Verify HMAC-SHA256 signature.
fn verify_signature(secret: &str, body: &[u8], signature: &str) -> bool {
    // Parse the signature (expecting format: "sha256=<hex>")
    let expected_hex = signature.strip_prefix("sha256=").unwrap_or(signature);

    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(body);

    // Convert signature from hex
    let expected_bytes = match hex::decode(expected_hex) {
        Ok(b) => b,
        Err(_) => return false,
    };

    mac.verify_slice(&expected_bytes).is_ok()
}

/// Webhook handler to trigger agent commands via HTTP POST.
async fn webhook_handler(
    State(state): State<WsState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Verify signature if webhook_secret is configured
    if let Some(ref secret) = state.webhook_secret {
        let signature = headers
            .get("X-Signature-256")
            .or_else(|| headers.get("x-hub-signature-256"))
            .and_then(|v| v.to_str().ok());

        match signature {
            Some(sig) => {
                if !verify_signature(secret, &body, sig) {
                    warn!("Webhook signature verification failed");
                    return (StatusCode::UNAUTHORIZED, "Invalid signature").into_response();
                }
            }
            None => {
                warn!("Webhook signature header missing");
                return (StatusCode::UNAUTHORIZED, "Missing signature").into_response();
            }
        }
    }

    // Parse the payload
    let payload: WebhookPayload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to parse webhook payload: {}", e);
            return (StatusCode::BAD_REQUEST, "Invalid JSON payload").into_response();
        }
    };

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
