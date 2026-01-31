//! Agent pool for managing multiple agent instances.

use crate::agent::r#loop::{spawn_agent_loop, AgentLoopHandle, AgentTask};
use crate::channels::ChannelManager;
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

/// Configuration for channel-to-agent routing.
#[derive(Debug, Clone, Default)]
pub struct RoutingConfig {
    /// Map channel name -> agent name
    pub channel_routes: HashMap<String, String>,
    /// Default agent for unrouted channels
    pub default_agent: Option<String>,
}

/// A single agent instance with its own loop and state.
pub struct AgentInstance {
    pub name: String,
    pub loop_handle: Option<AgentLoopHandle>,
    pub state_tx: watch::Sender<AgentState>,
    pub state_rx: watch::Receiver<AgentState>,
    pub cmd_tx: mpsc::Sender<AgentCommand>,
    pub cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
}

impl AgentInstance {
    pub fn new(name: String) -> Self {
        let (state_tx, state_rx) = watch::channel(AgentState::Idle);
        let (cmd_tx, cmd_rx) = mpsc::channel(32);

        Self {
            name,
            loop_handle: None,
            state_tx,
            state_rx,
            cmd_tx,
            cmd_rx: Arc::new(Mutex::new(cmd_rx)),
        }
    }
}

/// Pool of agent instances with routing capability.
pub struct AgentPool {
    agents: RwLock<HashMap<String, AgentInstance>>,
    routing: RwLock<RoutingConfig>,
    // Shared resources
    tool_registry: Arc<RwLock<ToolRegistry>>,
    memory: Arc<MemoryManager>,
    sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
    channel_manager: Arc<ChannelManager>,
}

impl AgentPool {
    pub fn new(
        tool_registry: Arc<RwLock<ToolRegistry>>,
        memory: Arc<MemoryManager>,
        sidecar: Arc<Mutex<SidecarHandle>>,
        permission_engine: Arc<Mutex<PermissionEngine>>,
        channel_manager: Arc<ChannelManager>,
    ) -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            routing: RwLock::new(RoutingConfig::default()),
            tool_registry,
            memory,
            sidecar,
            permission_engine,
            channel_manager,
        }
    }

    /// Create a new agent instance.
    pub async fn create_agent(&self, name: &str) -> anyhow::Result<()> {
        let mut agents = self.agents.write().await;
        if agents.contains_key(name) {
            return Err(anyhow::anyhow!("Agent '{}' already exists", name));
        }

        let instance = AgentInstance::new(name.to_string());
        info!(agent = %name, "Created new agent instance");
        agents.insert(name.to_string(), instance);
        Ok(())
    }

    /// Remove an agent instance.
    pub async fn remove_agent(&self, name: &str) -> anyhow::Result<()> {
        let mut agents = self.agents.write().await;
        if let Some(mut instance) = agents.remove(name) {
            // Stop the loop if running
            if let Some(handle) = instance.loop_handle.take() {
                handle.cancel_token.cancel();
                let _ = handle.cmd_tx.send(AgentCommand::Stop).await;
            }
            info!(agent = %name, "Removed agent instance");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent '{}' not found", name))
        }
    }

    /// List all agent names.
    pub async fn list_agents(&self) -> Vec<String> {
        self.agents.read().await.keys().cloned().collect()
    }

    /// Get the state of a specific agent.
    pub async fn get_agent_state(&self, name: &str) -> Option<AgentState> {
        let agents = self.agents.read().await;
        agents.get(name).map(|a| a.state_rx.borrow().clone())
    }

    /// Set channel routing configuration.
    pub async fn set_channel_route(&self, channel: &str, agent_name: &str) {
        let mut routing = self.routing.write().await;
        routing
            .channel_routes
            .insert(channel.to_string(), agent_name.to_string());
        info!(channel = %channel, agent = %agent_name, "Set channel route");
    }

    /// Get the agent name for a channel.
    pub async fn get_route_for_channel(&self, channel: &str) -> Option<String> {
        let routing = self.routing.read().await;
        routing
            .channel_routes
            .get(channel)
            .cloned()
            .or_else(|| routing.default_agent.clone())
    }

    /// Route a command to the appropriate agent.
    pub async fn route_command(&self, channel: &str, command: AgentCommand) -> anyhow::Result<()> {
        let agent_name = self
            .get_route_for_channel(channel)
            .await
            .ok_or_else(|| anyhow::anyhow!("No route configured for channel '{}'", channel))?;

        let agents = self.agents.read().await;
        let agent = agents
            .get(&agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent '{}' not found", agent_name))?;

        agent
            .cmd_tx
            .send(command)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send command to agent: {}", e))
    }

    /// Start an agent task.
    pub async fn start_agent_task(&self, agent_name: &str, task: AgentTask) -> anyhow::Result<()> {
        let mut agents = self.agents.write().await;
        let agent = agents
            .get_mut(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent '{}' not found", agent_name))?;

        // Check if already running
        if let Some(ref handle) = agent.loop_handle {
            if !handle.join_handle.is_finished() {
                return Err(anyhow::anyhow!(
                    "Agent '{}' is already running a task",
                    agent_name
                ));
            }
        }

        let tool_registry = self.tool_registry.read().await;
        let handle = spawn_agent_loop(
            task,
            agent.state_tx.clone(),
            Arc::new((*tool_registry).clone()),
            self.memory.clone(),
            self.sidecar.clone(),
            self.permission_engine.clone(),
        );

        agent.loop_handle = Some(handle);
        info!(agent = %agent_name, "Started agent task");
        Ok(())
    }

    /// Stop an agent's current task.
    pub async fn stop_agent_task(&self, agent_name: &str) -> anyhow::Result<()> {
        let mut agents = self.agents.write().await;
        let agent = agents
            .get_mut(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent '{}' not found", agent_name))?;

        if let Some(handle) = agent.loop_handle.take() {
            handle.cancel_token.cancel();
            let _ = handle.cmd_tx.send(AgentCommand::Stop).await;
            info!(agent = %agent_name, "Stopped agent task");
        }
        Ok(())
    }
}
