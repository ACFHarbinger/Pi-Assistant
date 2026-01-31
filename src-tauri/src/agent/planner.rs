//! Agent planner: LLM-driven task planning via Python sidecar.

use crate::ipc::SidecarHandle;
use crate::tools::ToolRegistry;
use anyhow::Result;
use pi_core::agent_types::ToolCall;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Plan returned by the LLM planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPlan {
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub question: Option<String>,
    pub is_complete: bool,
}

/// Agent planner that uses the Python sidecar for LLM inference.
pub struct AgentPlanner {
    sidecar: Arc<Mutex<SidecarHandle>>,
    tool_registry: Arc<ToolRegistry>,
}

impl AgentPlanner {
    /// Create a new planner.
    pub fn new(sidecar: Arc<Mutex<SidecarHandle>>, tool_registry: Arc<ToolRegistry>) -> Self {
        Self {
            sidecar,
            tool_registry,
        }
    }

    /// Generate a plan for the next iteration.
    pub async fn plan(
        &self,
        task: &str,
        iteration: u32,
        context: Vec<serde_json::Value>,
        provider: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<AgentPlan> {
        info!(task = task, iteration = iteration, "Generating plan");

        let tools = self.tool_registry.list_tools();

        let mut sidecar = self.sidecar.lock().await;
        let response = sidecar
            .request(
                "inference.plan",
                serde_json::json!({
                    "task": task,
                    "iteration": iteration,
                    "context": context,
                    "tools": tools,
                    "provider": provider.unwrap_or("local"),
                    "model_id": model_id,
                }),
            )
            .await?;

        let plan: AgentPlan = serde_json::from_value(response)?;

        info!(
            reasoning_len = plan.reasoning.len(),
            tool_calls = plan.tool_calls.len(),
            is_complete = plan.is_complete,
            "Plan generated"
        );

        Ok(plan)
    }

    /// Generate a simple completion (for chat responses).
    pub async fn complete(
        &self,
        prompt: &str,
        provider: Option<&str>,
        model_id: Option<&str>,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        let mut sidecar = self.sidecar.lock().await;
        let response = sidecar
            .request(
                "inference.complete",
                serde_json::json!({
                    "prompt": prompt,
                    "provider": provider.unwrap_or("local"),
                    "model_id": model_id,
                    "max_tokens": max_tokens.unwrap_or(1024),
                }),
            )
            .await?;

        let text = response
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(text)
    }
}
