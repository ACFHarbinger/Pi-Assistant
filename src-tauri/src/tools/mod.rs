//! Tool system: registry, trait, and implementations.

pub mod browser;
pub mod canvas;
pub mod code;
pub mod cron;
pub mod sessions;
pub mod shell;
pub mod training;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// Re-export ToolCall from pi-core
pub use pi_core::agent_types::ToolCall;

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            data: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        let msg = message.into();
        Self {
            success: false,
            output: String::new(),
            error: Some(msg),
            data: None,
        }
    }

    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Permission tier for tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionTier {
    /// Always allowed (read-only operations).
    Low,
    /// Needs user approval (write operations, network).
    Medium,
    /// Dangerous (system changes, deletions).
    High,
}

/// Trait that all tools must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used in tool_calls).
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// JSON Schema for parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with given parameters.
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult>;

    /// Permission tier for this tool.
    fn permission_tier(&self) -> PermissionTier;
}

/// Registry of available tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new registry with default tools.
    pub fn new(
        ml_sidecar: Arc<Mutex<crate::ipc::SidecarHandle>>,
        _logic_sidecar: Arc<Mutex<crate::ipc::SidecarHandle>>,
        cron_manager: Arc<crate::cron::CronManager>,
    ) -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };

        // Register default tools
        registry.register(Arc::new(shell::ShellTool::new()));
        registry.register(Arc::new(code::CodeTool::new()));
        registry.register(Arc::new(training::TrainingTool::new(ml_sidecar)));
        registry.register(Arc::new(cron::CronTool::new(cron_manager)));

        registry
    }

    /// Register the canvas tool (requires AppHandle).
    pub fn register_canvas_tool(&mut self, app_handle: tauri::AppHandle) {
        self.register(Arc::new(canvas::CanvasTool::new(app_handle)));
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Execute a tool call.
    pub async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        let tool = self
            .tools
            .get(&call.tool_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", call.tool_name))?;

        tool.execute(call.parameters.clone()).await
    }

    /// List all tools with their schemas.
    pub fn list_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters_schema(),
                })
            })
            .collect()
    }

    /// Load tools from MCP configuration.
    pub async fn load_mcp_tools(&mut self) -> Result<()> {
        use crate::mcp::McpConfig;
        use crate::tools::mcp::McpToolWrapper;
        use tracing::info;

        // Load config from ~/.pi-assistant/mcp_config.json
        let config = McpConfig::load().await?;

        for (server_name, server_config) in config.mcp_servers {
            match crate::mcp::McpClient::new(&server_name, &server_config).await {
                Ok(client) => {
                    let client: Arc<crate::mcp::McpClient> = Arc::new(client);
                    match client.list_tools().await {
                        Ok(tools) => {
                            for tool_info in tools {
                                info!(tool = %tool_info.name, server = %server_name, "Registering MCP tool");
                                let tool = Arc::new(McpToolWrapper::new(client.clone(), tool_info));
                                self.register(tool);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(server = %server_name, error = %e, "Failed to list MCP tools");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(server = %server_name, error = %e, "Failed to start MCP server");
                }
            }
        }
        Ok(())
    }
}

// ToolRegistry no longer implements Default as it requires a SidecarHandle
pub mod mcp;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct MockTool;
    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            "mock"
        }
        fn description(&self) -> &str {
            "A mock tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({})
        }
        fn permission_tier(&self) -> PermissionTier {
            PermissionTier::Low
        }
        async fn execute(&self, _params: serde_json::Value) -> Result<ToolResult> {
            Ok(ToolResult::success("mock executed"))
        }
    }

    #[tokio::test]
    async fn test_registry_register_and_get() {
        let ml_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let logic_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let (tx, _) = tokio::sync::mpsc::channel(32);
        let cron_manager = Arc::new(
            crate::cron::CronManager::new(&std::env::temp_dir(), tx)
                .await
                .unwrap(),
        );
        let mut registry = ToolRegistry::new(ml_sidecar, logic_sidecar, cron_manager);
        registry.register(Arc::new(MockTool));
        assert!(registry.get("mock").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_registry_execute() {
        let ml_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let logic_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let (tx, _) = tokio::sync::mpsc::channel(32);
        let cron_manager = Arc::new(
            crate::cron::CronManager::new(&std::env::temp_dir(), tx)
                .await
                .unwrap(),
        );
        let mut registry = ToolRegistry::new(ml_sidecar, logic_sidecar, cron_manager);
        registry.register(Arc::new(MockTool));

        let call = ToolCall {
            tool_name: "mock".into(),
            parameters: json!({}),
        };

        let result = registry.execute(&call).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "mock executed");
    }

    #[tokio::test]
    async fn test_default_tools_exist() {
        let ml_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let logic_sidecar = Arc::new(Mutex::new(crate::ipc::SidecarHandle::new()));
        let (tx, _) = tokio::sync::mpsc::channel(32);
        let cron_manager = Arc::new(
            crate::cron::CronManager::new(&std::env::temp_dir(), tx)
                .await
                .unwrap(),
        );
        let registry = ToolRegistry::new(ml_sidecar, logic_sidecar, cron_manager);
        assert!(registry.get("shell").is_some());
        assert!(registry.get("code").is_some());
        assert!(registry.get("train").is_some());
        assert!(registry.get("cron").is_some());
    }
}
