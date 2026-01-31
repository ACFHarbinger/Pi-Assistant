//! Application state management.

use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::PermissionEngine;
use crate::tools::ToolRegistry;
use pi_core::agent_types::{AgentCommand, AgentState};
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};

/// Shared application state.
pub struct AppState {
    pub agent_state_tx: watch::Sender<AgentState>,
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    pub agent_cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    pub tool_registry: Arc<ToolRegistry>,
    pub permissions: Arc<Mutex<PermissionEngine>>,
    pub memory: Arc<MemoryManager>,
    pub sidecar: Arc<Mutex<SidecarHandle>>,
}

impl AppState {
    pub fn new() -> Self {
        let (agent_state_tx, agent_state_rx) = watch::channel(AgentState::Idle);
        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel(32);

        let memory = MemoryManager::new(None).expect("Failed to initialize memory");

        Self {
            agent_state_tx,
            agent_state_rx,
            agent_cmd_tx,
            agent_cmd_rx: Arc::new(Mutex::new(agent_cmd_rx)),
            tool_registry: Arc::new(ToolRegistry::new()),
            permissions: Arc::new(Mutex::new(PermissionEngine::new())),
            memory: Arc::new(memory),
            sidecar: Arc::new(Mutex::new(SidecarHandle::new())),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
