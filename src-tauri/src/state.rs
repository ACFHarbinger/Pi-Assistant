//! Application state management.

use pi_core::agent_types::{AgentCommand, AgentState};
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};

/// Shared application state.
pub struct AppState {
    pub agent_state_tx: watch::Sender<AgentState>,
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    pub agent_cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
}

impl AppState {
    pub fn new() -> Self {
        let (agent_state_tx, agent_state_rx) = watch::channel(AgentState::Idle);
        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel(32);

        Self {
            agent_state_tx,
            agent_state_rx,
            agent_cmd_tx,
            agent_cmd_rx: Arc::new(Mutex::new(agent_cmd_rx)),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
