//! Agent planner: LLM-driven task planning via Python sidecar.

use crate::ipc::SidecarHandle;
use crate::tools::ToolRegistry;
use anyhow::Result;
use pi_core::agent_types::AgentState;
use pi_core::agent_types::ToolCall;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{watch, Mutex, RwLock};
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
    ml_sidecar: Arc<Mutex<SidecarHandle>>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
}

impl AgentPlanner {
    /// Create a new planner.
    pub fn new(
        ml_sidecar: Arc<Mutex<SidecarHandle>>,
        tool_registry: Arc<RwLock<ToolRegistry>>,
    ) -> Self {
        Self {
            ml_sidecar,
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

        let tools = self.tool_registry.read().await.list_tools();

        let mut sidecar = self.ml_sidecar.lock().await;
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

    /// Generate a simple completion (for chat responses) with streaming.
    pub async fn complete(
        &self,
        agent_id: uuid::Uuid,
        prompt: &str,
        provider: Option<&str>,
        model_id: Option<&str>,
        max_tokens: Option<u32>,
        state_tx: Option<watch::Sender<AgentState>>,
    ) -> Result<String> {
        let (ptx, mut prx) = tokio::sync::mpsc::channel(100);
        let mut sidecar = self.ml_sidecar.lock().await;

        let params = serde_json::json!({
            "prompt": prompt,
            "provider": provider.unwrap_or("local"),
            "model_id": model_id,
            "max_tokens": max_tokens.unwrap_or(1024),
            "stream": true,
        });

        // Send initial empty message to signal 'thinking' state
        if let Some(ref tx) = state_tx {
            let _ = tx.send(AgentState::AssistantMessage {
                agent_id,
                content: String::new(),
                is_streaming: true,
            });
        }

        // Start request with progress listener
        let sidecar_task = sidecar.request_with_progress("inference.complete", params, Some(ptx));

        // Consume progress updates (tokens)
        let state_tx_clone = state_tx.clone();
        let progress_task = tokio::spawn(async move {
            let mut accumulated_text = String::new();
            while let Some(progress) = prx.recv().await {
                if let Some(token) = progress.get("token").and_then(|v| v.as_str()) {
                    accumulated_text.push_str(token);
                    if let Some(ref tx) = state_tx_clone {
                        let _ = tx.send(AgentState::AssistantMessage {
                            agent_id,
                            content: accumulated_text.clone(),
                            is_streaming: true,
                        });
                    }
                }
            }
        });

        // Wait for final response
        let final_response = sidecar_task.await?;
        let _ = progress_task.abort(); // Ensure progress task stops

        let text = final_response
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Broadcast final message (non-streaming)
        if let Some(tx) = state_tx {
            let _ = tx.send(AgentState::AssistantMessage {
                agent_id,
                content: text.clone(),
                is_streaming: false,
            });
        }

        Ok(text)
    }
}
