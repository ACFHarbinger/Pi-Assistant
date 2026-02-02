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
use uuid::Uuid;

/// Shared application state.
pub struct AppState {
    pub agent_state_tx: watch::Sender<AgentState>,
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    pub agent_cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    pub tool_registry: Arc<RwLock<ToolRegistry>>,
    pub permissions: Arc<Mutex<PermissionEngine>>,
    pub memory: Arc<MemoryManager>,
    pub ml_sidecar: Arc<Mutex<SidecarHandle>>,
    pub logic_sidecar: Arc<Mutex<SidecarHandle>>,
    pub channel_manager: Arc<ChannelManager>,
    pub cron_manager: Arc<CronManager>,
    pub voice_manager: Arc<Mutex<crate::voice::VoiceManager>>,
    pub system_tool: Arc<crate::tools::system::SystemTool>,
    pub skill_manager: Arc<tokio::sync::RwLock<SkillManager>>,
    pub chat_session_id: Arc<RwLock<Uuid>>,
}

impl AppState {
    pub async fn new() -> Self {
        let (agent_state_tx, agent_state_rx) = watch::channel(AgentState::Idle);
        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel(32);

        let config_dir = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".pi-assistant");

        let memory = MemoryManager::new(None).expect("Failed to initialize memory");

        let ml_sidecar = Arc::new(Mutex::new(
            SidecarHandle::new()
                .with_sidecar_dir("ml")
                .with_sidecar_module("ml_sidecar_main"),
        ));
        let logic_sidecar = Arc::new(Mutex::new(
            SidecarHandle::new()
                .with_sidecar_dir("ml")
                .with_sidecar_module("logic_main"),
        ));

        let cron_manager = CronManager::new(&config_dir, agent_cmd_tx.clone())
            .await
            .expect("Failed to initialize cron manager");
        let cron_manager_arc = Arc::new(cron_manager);

        let system_tool = Arc::new(crate::tools::system::SystemTool::new());

        let mut tool_registry = ToolRegistry::new(
            ml_sidecar.clone(),
            logic_sidecar.clone(),
            cron_manager_arc.clone(),
        );
        tool_registry.register(system_tool.clone());
        if let Err(e) = tool_registry.load_mcp_tools().await {
            tracing::warn!("Failed to load MCP tools: {}", e);
        }

        // Wrap registry in Arc<RwLock> early so TrainingTool can reference it
        // for auto-registering deployed models.
        let tool_registry = Arc::new(RwLock::new(tool_registry));
        {
            let training_tool = Arc::new(crate::tools::training::TrainingTool::new(
                ml_sidecar.clone(),
                tool_registry.clone(),
            ));
            tool_registry.write().await.register(training_tool);
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

        let chat_session_id = Arc::new(RwLock::new(
            memory
                .create_session(Some("Chat"))
                .expect("Failed to create chat session"),
        ));

        Self {
            agent_state_tx,
            agent_state_rx,
            agent_cmd_tx: agent_cmd_tx.clone(),
            agent_cmd_rx: Arc::new(Mutex::new(agent_cmd_rx)),
            tool_registry,
            permissions: Arc::new(Mutex::new(PermissionEngine::new())),
            memory: Arc::new(memory),
            ml_sidecar,
            logic_sidecar,
            channel_manager: Arc::new(ChannelManager::new()),
            cron_manager: cron_manager_arc,
            voice_manager: voice_manager_arc,
            system_tool,
            skill_manager: skill_manager_arc,
            chat_session_id,
        }
    }

    pub async fn spawn_sidecar_listeners(&self, app: tauri::AppHandle) {
        let ml_sidecar = self.ml_sidecar.clone();
        let mut rx = ml_sidecar
            .lock()
            .await
            .take_progress_rx()
            .expect("ML progress receiver already taken");

        tokio::spawn(async move {
            use tauri::Emitter;
            while let Some(update) = rx.recv().await {
                // Emit progress to frontend
                let _ = app.emit("sidecar-progress", update);
            }
        });
    }

    pub async fn spawn_resource_monitor(&self, app: tauri::AppHandle) {
        let system_tool = self.system_tool.clone();
        let ml_sidecar = self.ml_sidecar.clone();

        tokio::spawn(async move {
            use tauri::Emitter;
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));

            loop {
                interval.tick().await;

                // 1. Get CPU/RAM from SystemTool
                let mut status = system_tool.get_system_status_snapshot().await;

                // 2. Get GPU from ML sidecar
                let gpu_res = {
                    let mut ml = ml_sidecar.lock().await;
                    ml.request("device.refresh", serde_json::json!({})).await
                };

                if let Ok(gpu_data) = gpu_res {
                    if let Some(obj) = status.as_object_mut() {
                        obj.insert("gpu".to_string(), gpu_data);
                    }
                }

                // 3. Emit update
                let _ = app.emit("resource-update", status);
            }
        });
    }
}
