//! Application state management.

use crate::channels::ChannelManager;
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
    pub channel_manager: Arc<ChannelManager>,
}

impl AppState {
    pub async fn new() -> Self {
        let (agent_state_tx, agent_state_rx) = watch::channel(AgentState::Idle);
        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel(32);

        let memory = MemoryManager::new(None).expect("Failed to initialize memory");

        let sidecar = Arc::new(Mutex::new(SidecarHandle::new()));

        let mut tool_registry = ToolRegistry::new(sidecar.clone());
        if let Err(e) = tool_registry.load_mcp_tools().await {
            tracing::warn!("Failed to load MCP tools: {}", e);
        }

        Self {
            agent_state_tx,
            agent_state_rx,
            agent_cmd_tx: agent_cmd_tx.clone(),
            agent_cmd_rx: Arc::new(Mutex::new(agent_cmd_rx)),
            tool_registry: Arc::new(tool_registry),
            permissions: Arc::new(Mutex::new(PermissionEngine::new())),
            memory: Arc::new(memory),
            sidecar,
            channel_manager: Arc::new(ChannelManager::new()),
        }
    }
}
