use super::{PermissionTier, Tool, ToolResult};
use crate::mcp::client::McpToolInfo;
use crate::mcp::McpClient;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

pub struct McpToolWrapper {
    client: Arc<McpClient>,
    info: McpToolInfo,
}

impl McpToolWrapper {
    pub fn new(client: Arc<McpClient>, info: McpToolInfo) -> Self {
        Self { client, info }
    }
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> &str {
        self.info.description.as_deref().unwrap_or("MCP Tool")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.info.input_schema.clone()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        match self.client.call_tool(&self.info.name, params).await {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(e) => Ok(ToolResult::error(format!("{}", e))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        // Defaulting all MCP tools to Medium for now.
        // In the future, we could configure this in mcp_config.json
        PermissionTier::Medium
    }
}
