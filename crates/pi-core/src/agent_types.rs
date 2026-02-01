//! Agent types shared across the application.

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
        task_id: Uuid,
        iteration: u32,
    },
    Paused {
        task_id: Uuid,
        question: Option<String>,
        awaiting_permission: Option<PermissionRequest>,
    },
    Stopped {
        task_id: Uuid,
        reason: StopReason,
    },
    AssistantMessage {
        content: String,
        #[serde(default)]
        is_streaming: bool,
    },
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
    },
    Stop,
    Pause,
    Resume,
    AnswerQuestion {
        response: String,
    },
    ApprovePermission {
        request_id: Uuid,
        approved: bool,
        remember: bool,
    },
    ChatMessage {
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
