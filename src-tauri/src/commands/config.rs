use crate::mcp::config::McpServerConfig;
use crate::mcp::McpConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;

// ── Shared Config Types ──────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ToolsConfig {
    pub enabled_tools: HashMap<String, bool>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ModelsConfig {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,     // "anthropic", "gemini", "local", etc.
    pub path: Option<String>, // Local path or HF ID
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct McpMarketplaceItem {
    pub name: String,
    pub description: String,
    pub command: String,
    pub args: Vec<String>,
    pub env_vars: Vec<String>, // Names of env vars to prompt for
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResetOptions {
    pub memory: bool,
    pub mcp_config: bool,
    pub tools_config: bool,
    pub models_config: bool,
    pub personality: bool,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct TelegramConfig {
    pub token: Option<String>,
    pub enabled: bool,
    pub allowed_users: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct DiscordConfig {
    pub token: Option<String>,
    pub enabled: bool,
}

// ── Helper: Config Paths ─────────────────────────────────────────────

fn get_config_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("No home dir"))?;
    let dir = home.join(".pi-assistant");
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

// ── MCP Commands ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_mcp_config() -> Result<McpConfig, String> {
    McpConfig::load().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_mcp_server(name: String, config: McpServerConfig) -> Result<McpConfig, String> {
    let mut current = McpConfig::load().await.map_err(|e| e.to_string())?;
    current.mcp_servers.insert(name, config);

    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("mcp_config.json");
    let json = serde_json::to_string_pretty(&current).map_err(|e| e.to_string())?;
    fs::write(&path, json)
        .await
        .map_err(|e: std::io::Error| e.to_string())?;

    Ok(current)
}

#[tauri::command]
pub async fn remove_mcp_server(name: String) -> Result<McpConfig, String> {
    let mut current = McpConfig::load().await.map_err(|e| e.to_string())?;
    current.mcp_servers.remove(&name);

    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("mcp_config.json");
    let json = serde_json::to_string_pretty(&current).map_err(|e| e.to_string())?;
    fs::write(&path, json)
        .await
        .map_err(|e: std::io::Error| e.to_string())?;

    Ok(current)
}

#[tauri::command]
pub async fn get_mcp_marketplace() -> Result<Vec<McpMarketplaceItem>, String> {
    Ok(vec![
        McpMarketplaceItem {
            name: "GitHub".to_string(),
            description: "Access GitHub repositories, issues, and pull requests.".to_string(),
            command: "npx".to_string(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-github".into()],
            env_vars: vec!["GITHUB_PERSONAL_ACCESS_TOKEN".into()],
        },
        McpMarketplaceItem {
            name: "Filesystem".to_string(),
            description: "Read and write files on your local system (restricted paths)."
                .to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-filesystem".into(),
                "/home/pkhunter/Repositories".into(),
            ],
            env_vars: vec![],
        },
        McpMarketplaceItem {
            name: "Brave Search".to_string(),
            description: "Search the web using Brave Search API.".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-brave-search".into(),
            ],
            env_vars: vec!["BRAVE_API_KEY".into()],
        },
        McpMarketplaceItem {
            name: "SQLite".to_string(),
            description: "Query SQLite databases.".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-sqlite".into(),
                "test.db".into(),
            ],
            env_vars: vec![],
        },
        McpMarketplaceItem {
            name: "Memory".to_string(),
            description: "Ephemeral knowledge graph memory.".to_string(),
            command: "npx".to_string(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-memory".into()],
            env_vars: vec![],
        },
        McpMarketplaceItem {
            name: "Google Maps".to_string(),
            description: "Search places and get directions.".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-google-maps".into(),
            ],
            env_vars: vec!["GOOGLE_MAPS_API_KEY".into()],
        },
    ])
}

// ── Tool Commands ────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_tools_config() -> Result<ToolsConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("tools_config.json");
    if !path.exists() {
        return Ok(ToolsConfig::default());
    }
    let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn toggle_tool(name: String, enabled: bool) -> Result<ToolsConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("tools_config.json");
    let mut config = if path.exists() {
        let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        ToolsConfig::default()
    };

    config.enabled_tools.insert(name, enabled);

    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json)
        .await
        .map_err(|e: std::io::Error| e.to_string())?;

    Ok(config)
}

// ── Model Commands ───────────────────────────────────────────────────

#[tauri::command]
pub async fn get_models_config() -> Result<ModelsConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("models.json");
    if !path.exists() {
        return Ok(ModelsConfig {
            models: vec![
                ModelInfo {
                    id: "gemini-3-pro".into(),
                    provider: "google".into(),
                    path: None,
                    description: Some("Gemini 3 Pro".into()),
                },
                ModelInfo {
                    id: "gemini-3-flash".into(),
                    provider: "google".into(),
                    path: None,
                    description: Some("Gemini 3 Flash".into()),
                },
                ModelInfo {
                    id: "claude-4-5-sonnet-latest".into(),
                    provider: "google".into(),
                    path: None,
                    description: Some("Claude Sonnet 4.5".into()),
                },
                ModelInfo {
                    id: "claude-4-5-opus-latest".into(),
                    provider: "google".into(),
                    path: None,
                    description: Some("Claude Opus 4.5".into()),
                },
                ModelInfo {
                    id: "gpt-oss-120b".into(),
                    provider: "google".into(),
                    path: None,
                    description: Some("GPT-OSS 120B".into()),
                },
            ],
        });
    }
    let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_model(model: ModelInfo) -> Result<ModelsConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("models.json");
    let mut config = if path.exists() {
        let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        ModelsConfig::default()
    };

    // Replace if exists, else add
    if let Some(idx) = config.models.iter().position(|m| m.id == model.id) {
        config.models[idx] = model;
    } else {
        config.models.push(model);
    }

    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json)
        .await
        .map_err(|e: std::io::Error| e.to_string())?;

    Ok(config)
}

#[tauri::command]
pub async fn load_model(
    state: tauri::State<'_, crate::state::AppState>,
    model_id: String,
    backend: Option<String>,
) -> Result<(), String> {
    let mut sidecar = state.ml_sidecar.lock().await;
    // Request sidecar to load model
    let _response = sidecar
        .request(
            "inference.load_model",
            serde_json::json!({
                "model_id": model_id,
                "backend": backend
            }),
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}
#[tauri::command]
pub async fn save_api_key(provider: String, key: String) -> Result<(), String> {
    let config_dir = get_config_dir().map_err(|e| e.to_string())?;
    let path = config_dir.join("secrets.json");

    let mut secrets: HashMap<String, String> = if path.exists() {
        let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        HashMap::new()
    };

    secrets.insert(provider, key);

    let json = serde_json::to_string_pretty(&secrets).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub async fn reset_agent(options: ResetOptions) -> Result<(), String> {
    let config_dir = get_config_dir().map_err(|e| e.to_string())?;

    if options.memory {
        let path = config_dir.join("memory.db");
        if path.exists() {
            fs::remove_file(path).await.map_err(|e| e.to_string())?;
        }
    }

    if options.mcp_config {
        let path = config_dir.join("mcp_config.json");
        if path.exists() {
            fs::remove_file(path).await.map_err(|e| e.to_string())?;
        }
    }

    if options.tools_config {
        let path = config_dir.join("tools_config.json");
        if path.exists() {
            fs::remove_file(path).await.map_err(|e| e.to_string())?;
        }
    }

    if options.models_config {
        let path = config_dir.join("models.json");
        if path.exists() {
            fs::remove_file(path).await.map_err(|e| e.to_string())?;
        }
    }

    // Personality reset involves soul.md and potentially clearing sidecar cache
    // For now, we'll just indicate personality reset is handled
    // Actually, soul.md might be in the workspace root, not config_dir
    if options.personality {
        // We don't delete soul.md as it's a template, but we can clear the hatching flag
        // Hatching flag is in localStorage (frontend), so we'll handle that there
    }

    Ok(())
}

#[tauri::command]
pub async fn save_agent_name(name: String) -> Result<(), String> {
    let config_dir = get_config_dir().map_err(|e| e.to_string())?;
    let path = config_dir.join("agent_config.json");

    let mut config: HashMap<String, String> = if path.exists() {
        let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        HashMap::new()
    };

    config.insert("agent_name".to_string(), name);

    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub async fn get_current_model() -> Result<Option<String>, String> {
    let config_dir = get_config_dir().map_err(|e| e.to_string())?;
    let path = config_dir.join("agent_config.json");

    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    let config: HashMap<String, String> = serde_json::from_str(&content).unwrap_or_default();

    Ok(config.get("current_model").cloned())
}

#[tauri::command]
pub async fn save_current_model(model_id: String) -> Result<(), String> {
    let config_dir = get_config_dir().map_err(|e| e.to_string())?;
    let path = config_dir.join("agent_config.json");

    let mut config: HashMap<String, String> = if path.exists() {
        let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        HashMap::new()
    };

    config.insert("current_model".to_string(), model_id);

    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub async fn get_telegram_config() -> Result<TelegramConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("telegram.json");
    if !path.exists() {
        return Ok(TelegramConfig::default());
    }
    let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_telegram_config(
    state: tauri::State<'_, crate::state::AppState>,
    config: TelegramConfig,
) -> Result<(), String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("telegram.json");
    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;

    // If enabled, start/restart the channel
    if config.enabled {
        if let Some(token) = &config.token {
            // We need to recreate the channel with the new token
            let channel = Box::new(crate::channels::telegram::TelegramChannel::new(
                token.clone(),
                state.agent_cmd_tx.clone(),
                state.logic_sidecar.clone(),
            ));
            // Set allowed users
            for user in &config.allowed_users {
                channel.allow_user(*user).await;
            }

            state.channel_manager.add_channel(channel).await;
            state
                .channel_manager
                .start_channel("telegram")
                .await
                .map_err(|e| e.to_string())?;
        }
    } else {
        state
            .channel_manager
            .stop_channel("telegram")
            .await
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tauri::command]
pub async fn get_discord_config() -> Result<DiscordConfig, String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("discord.json");
    if !path.exists() {
        return Ok(DiscordConfig::default());
    }
    let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_discord_config(
    state: tauri::State<'_, crate::state::AppState>,
    config: DiscordConfig,
) -> Result<(), String> {
    let path = get_config_dir()
        .map_err(|e| e.to_string())?
        .join("discord.json");
    let json = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, json).await.map_err(|e| e.to_string())?;

    // If enabled, start/restart the channel
    if config.enabled {
        if let Some(token) = &config.token {
            let channel = Box::new(crate::channels::discord::DiscordChannel::new(
                token.clone(),
                state.agent_cmd_tx.clone(),
                state.agent_state_rx.clone(),
            ));

            state.channel_manager.add_channel(channel).await;
            state
                .channel_manager
                .start_channel("discord")
                .await
                .map_err(|e| e.to_string())?;
        }
    } else {
        state
            .channel_manager
            .stop_channel("discord")
            .await
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}
