//! Session tool for agent management.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::agent::pool::AgentPool;
use crate::tools::{PermissionTier, Tool, ToolResult};

/// Tool for managing agent sessions.
pub struct SessionTool {
    pool: Arc<RwLock<AgentPool>>,
}

impl SessionTool {
    pub fn new(pool: Arc<RwLock<AgentPool>>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Tool for SessionTool {
    fn name(&self) -> &str {
        "sessions"
    }

    fn description(&self) -> &str {
        "Manage agent sessions. Actions: list, create, remove, route."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "remove", "route"],
                    "description": "Action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Agent name (for create/remove)"
                },
                "channel": {
                    "type": "string",
                    "description": "Channel name (for route)"
                },
                "agent": {
                    "type": "string",
                    "description": "Target agent name (for route)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: Value) -> anyhow::Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing action parameter"))?;

        let pool = self.pool.read().await;

        match action {
            "list" => {
                let agents = pool.list_agents().await;
                Ok(ToolResult::success(format!(
                    "Active agents: {}",
                    if agents.is_empty() {
                        "none".to_string()
                    } else {
                        agents.join(", ")
                    }
                )))
            }
            "create" => {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing name for create action"))?;

                drop(pool);
                let pool = self.pool.write().await;
                pool.create_agent(name).await?;

                Ok(ToolResult::success(format!("Created agent '{}'", name)))
            }
            "remove" => {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing name for remove action"))?;

                drop(pool);
                let pool = self.pool.write().await;
                pool.remove_agent(name).await?;

                Ok(ToolResult::success(format!("Removed agent '{}'", name)))
            }
            "route" => {
                let channel = params
                    .get("channel")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing channel for route action"))?;

                let agent = params
                    .get("agent")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing agent for route action"))?;

                drop(pool);
                let pool = self.pool.write().await;
                pool.set_channel_route(channel, agent).await;

                Ok(ToolResult::success(format!(
                    "Routed channel '{}' to agent '{}'",
                    channel, agent
                )))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
