use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct McpConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfig>,
}

impl McpConfig {
    pub async fn load() -> Result<Self> {
        let home_dir =
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let config_dir = home_dir.join(".pi-assistant");
        let config_path = config_dir.join("mcp_config.json");

        if !config_path.exists() {
            // Create default config if it doesn't exist
            if !config_dir.exists() {
                fs::create_dir_all(&config_dir).await?;
            }

            let default_config = McpConfig::default();
            let json = serde_json::to_string_pretty(&default_config)?;
            fs::write(&config_path, json).await?;
            return Ok(default_config);
        }

        let content = fs::read_to_string(&config_path).await?;
        let config: McpConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}
