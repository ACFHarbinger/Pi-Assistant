//! Application state management.

use crate::channels::ChannelManager;
use crate::cron::CronManager;
use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::PermissionEngine;
use crate::skills::SkillManager;
use crate::tools::ToolRegistry;
use pi_core::agent_types::{AgentCommand, AgentState};
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex, RwLock};

/// Shared application state.
pub struct AppState {
    pub agent_state_tx: watch::Sender<AgentState>,
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    pub agent_cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    pub tool_registry: Arc<RwLock<ToolRegistry>>,
    pub permissions: Arc<Mutex<PermissionEngine>>,
    pub memory: Arc<MemoryManager>,
    pub sidecar: Arc<Mutex<SidecarHandle>>,
    pub channel_manager: Arc<ChannelManager>,
    pub cron_manager: Arc<CronManager>,
    pub voice_manager: Arc<Mutex<crate::voice::VoiceManager>>,
    pub skill_manager: Arc<tokio::sync::RwLock<SkillManager>>,
}

impl AppState {
    pub async fn new() -> Self {
        let (agent_state_tx, agent_state_rx) = watch::channel(AgentState::Idle);
        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel(32);

        let config_dir = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".pi-assistant");

        let memory = MemoryManager::new(None).expect("Failed to initialize memory");

        let sidecar = Arc::new(Mutex::new(SidecarHandle::new()));

        let cron_manager = CronManager::new(&config_dir, agent_cmd_tx.clone())
            .await
            .expect("Failed to initialize cron manager");
        let cron_manager_arc = Arc::new(cron_manager);

        let mut tool_registry = ToolRegistry::new(sidecar.clone(), cron_manager_arc.clone());
        if let Err(e) = tool_registry.load_mcp_tools().await {
            tracing::warn!("Failed to load MCP tools: {}", e);
        }

        let mut voice_manager = crate::voice::VoiceManager::new(agent_cmd_tx.clone());
        let model_path = config_dir.join("voice").join("vosk-model-small-en-us-0.15");
        if let Err(e) = voice_manager.init_detector(model_path, 16000.0).await {
            tracing::warn!("Failed to initialize voice detector: {}", e);
        }
        let voice_manager_arc = Arc::new(Mutex::new(voice_manager));

        // Load skills from workspace and global paths
        let mut skill_manager = SkillManager::new();
        let skill_paths = vec![
            std::env::current_dir()
                .unwrap_or_default()
                .join(".agent")
                .join("skills"),
            config_dir.join("skills"),
        ];
        if let Err(e) = skill_manager.load_from_paths(&skill_paths).await {
            tracing::warn!("Failed to load skills: {}", e);
        }
        let skill_manager_arc = Arc::new(tokio::sync::RwLock::new(skill_manager));

        Self {
            agent_state_tx,
            agent_state_rx,
            agent_cmd_tx: agent_cmd_tx.clone(),
            agent_cmd_rx: Arc::new(Mutex::new(agent_cmd_rx)),
            tool_registry: Arc::new(RwLock::new(tool_registry)),
            permissions: Arc::new(Mutex::new(PermissionEngine::new())),
            memory: Arc::new(memory),
            sidecar,
            channel_manager: Arc::new(ChannelManager::new()),
            cron_manager: cron_manager_arc,
            voice_manager: voice_manager_arc,
            skill_manager: skill_manager_arc,
        }
    }
}
