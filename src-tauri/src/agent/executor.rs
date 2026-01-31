//! Agent executor: tool dispatch and result collection.

use crate::tools::{ToolCall, ToolRegistry, ToolResult};
use anyhow::Result;
use std::sync::Arc;
use tracing::info;

/// Executor for agent tool calls.
pub struct AgentExecutor {
    tool_registry: Arc<ToolRegistry>,
}

impl AgentExecutor {
    /// Create a new executor.
    pub fn new(tool_registry: Arc<ToolRegistry>) -> Self {
        Self { tool_registry }
    }

    /// Execute a single tool call.
    pub async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        info!(tool = %call.tool_name, "Executing tool");
        self.tool_registry.execute(call).await
    }

    /// Execute multiple tool calls in sequence.
    pub async fn execute_all(&self, calls: &[ToolCall]) -> Vec<Result<ToolResult>> {
        let mut results = Vec::with_capacity(calls.len());

        for call in calls {
            results.push(self.execute(call).await);
        }

        results
    }
}
