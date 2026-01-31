//! Protocol types for Rust <-> Python IPC.

use serde::{Deserialize, Serialize};

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

/// Permission check result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PermissionResult {
    Allowed,
    NeedsApproval,
    Denied(String),
}
