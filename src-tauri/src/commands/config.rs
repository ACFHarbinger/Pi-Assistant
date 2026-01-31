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

// ── Helper: Config Paths ─────────────────────────────────────────────

fn get_config_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("No home dir"))?;
    let dir = home.join("pi-assistant");
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
            models: vec![ModelInfo {
                id: "gpt2".into(),
                path: None,
                description: Some("Default small model".into()),
            }],
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
) -> Result<(), String> {
    let mut sidecar = state.sidecar.lock().await;
    // Request sidecar to load model
    let _response = sidecar
        .request(
            "inference.load_model",
            serde_json::json!({
                "model_id": model_id
            }),
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}
