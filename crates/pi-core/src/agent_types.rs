//! Agent types shared across the application.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tsify::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

/// Agent state machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(tag = "status", content = "data")]
pub enum AgentState {
    #[default]
    Idle,
    Running {
        agent_id: Uuid,
        task_id: Uuid,
        iteration: u32,
        #[serde(default)]
        task_tree: Vec<Subtask>,
        #[serde(default)]
        active_subtask_id: Option<Uuid>,
        #[serde(default)]
        consecutive_errors: u32,
        #[serde(default)]
        cost_stats: Option<TokenUsage>,
    },
    Paused {
        agent_id: Uuid,
        task_id: Uuid,
        question: Option<String>,
        awaiting_permission: Option<PermissionRequest>,
    },
    Stopped {
        agent_id: Uuid,
        task_id: Uuid,
        reason: StopReason,
    },
    AssistantMessage {
        agent_id: Uuid,
        content: String,
        #[serde(default)]
        is_streaming: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Subtask {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub result: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum TaskStatus {
    Pending,
    Running,
    Blocked,
    Completed,
    Failed,
}

/// Reason the agent stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum StopReason {
    Completed,
    ManualStop,
    Error(String),
    IterationLimit,
}

/// Permission request from the agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct PermissionRequest {
    pub id: Uuid,
    pub tool_name: String,
    pub command: String,
    pub tier: String,
    pub description: String,
}

/// Commands sent TO the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentCommand {
    Start {
        task: String,
        max_iterations: Option<u32>,
        provider: Option<String>,
        model_id: Option<String>,
        #[serde(default)]
        cost_config: Option<CostConfig>,
    },
    Stop {
        agent_id: Option<Uuid>,
    },
    Pause {
        agent_id: Option<Uuid>,
    },
    Resume {
        agent_id: Option<Uuid>,
    },
    AnswerQuestion {
        agent_id: Option<Uuid>,
        response: String,
    },
    ApprovePermission {
        agent_id: Option<Uuid>,
        request_id: Uuid,
        approved: bool,
        remember: bool,
    },
    ChatMessage {
        agent_id: Option<Uuid>,
        content: String,
        provider: Option<String>,
        model_id: Option<String>,
    },
    ChannelMessage {
        id: String,
        channel: String,
        sender_id: String,
        sender_name: Option<String>,
        text: String,
        chat_id: String,
        /// Media attachment file paths with metadata (type, file_name)
        #[serde(default)]
        media: Vec<HashMap<String, String>>,
    },
    /// Internal message from another agent
    InternalMessage {
        from_agent_id: String,
        to_agent_id: String,
        content: String,
    },
}

/// Plan returned by the LLM planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPlan {
    pub tool_calls: Vec<ToolCall>,
    pub is_complete: bool,
    pub question: Option<String>,
    pub reasoning: String,
    #[serde(default)]
    pub reflection: Option<String>,
}

/// A single tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub parameters: serde_json::Value,
}

impl ToolCall {
    /// Display a human-readable command string.
    pub fn display_command(&self) -> String {
        if self.tool_name == "shell" {
            self.parameters
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>")
                .to_string()
        } else {
            format!("{}({})", self.tool_name, self.parameters)
        }
    }

    /// Describe the tool call.
    pub fn describe(&self) -> String {
        format!("Execute {} tool", self.tool_name)
    }

    /// Pattern key for permission caching.
    pub fn pattern_key(&self) -> String {
        self.display_command()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CostConfig {
    pub max_tokens_per_session: Option<u32>,
    pub max_cost_per_session: Option<f64>,
}
