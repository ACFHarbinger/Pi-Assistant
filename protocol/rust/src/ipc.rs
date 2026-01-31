//! IPC message types for Rust <-> Python NDJSON communication.

use serde::{Deserialize, Serialize};

/// Request sent from Rust to Python.
#[derive(Debug, Clone, Serialize)]
pub struct IpcRequest {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
}

/// Response/message from Python to Rust.
#[derive(Debug, Clone, Deserialize)]
pub struct IpcMessage {
    pub id: String,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<IpcError>,
    #[serde(default)]
    pub progress: Option<serde_json::Value>,
}

/// Error response from Python.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IpcError {
    pub code: String,
    pub message: String,
}

/// Progress update for long-running operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub request_id: String,
    pub data: serde_json::Value,
}
