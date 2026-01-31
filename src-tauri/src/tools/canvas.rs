//! Canvas tool for agent to push content to the Live Canvas.
//!
//! Supports state persistence: canvas content is saved to disk
//! and restored on application startup.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::sync::RwLock;

use crate::tools::{PermissionTier, Tool, ToolResult};

/// Manages canvas state persistence.
pub struct CanvasStateManager {
    state_file: PathBuf,
    current_content: RwLock<Option<String>>,
}

impl CanvasStateManager {
    /// Create a new state manager.
    pub fn new(config_dir: &std::path::Path) -> Self {
        Self {
            state_file: config_dir.join("canvas_state.json"),
            current_content: RwLock::new(None),
        }
    }

    /// Load persisted state from disk.
    pub async fn load(&self) -> Option<String> {
        if self.state_file.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&self.state_file).await {
                if let Ok(state) = serde_json::from_str::<Value>(&content) {
                    let html = state
                        .get("html")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    if let Some(ref h) = html {
                        *self.current_content.write().await = Some(h.clone());
                    }
                    return html;
                }
            }
        }
        None
    }

    /// Save current state to disk.
    pub async fn save(&self, content: &str) -> anyhow::Result<()> {
        *self.current_content.write().await = Some(content.to_string());
        let state = json!({ "html": content });
        tokio::fs::write(&self.state_file, serde_json::to_string_pretty(&state)?).await?;
        Ok(())
    }

    /// Clear persisted state.
    pub async fn clear(&self) -> anyhow::Result<()> {
        *self.current_content.write().await = None;
        if self.state_file.exists() {
            tokio::fs::remove_file(&self.state_file).await?;
        }
        Ok(())
    }

    /// Get current content without loading from disk.
    pub async fn get_current(&self) -> Option<String> {
        self.current_content.read().await.clone()
    }
}

/// Tool that allows the agent to push HTML/React to the canvas.
pub struct CanvasTool {
    app_handle: AppHandle,
    state_manager: Arc<CanvasStateManager>,
}

impl CanvasTool {
    pub fn new(app_handle: AppHandle, state_manager: Arc<CanvasStateManager>) -> Self {
        Self {
            app_handle,
            state_manager,
        }
    }
}

#[async_trait]
impl Tool for CanvasTool {
    fn name(&self) -> &str {
        "canvas"
    }

    fn description(&self) -> &str {
        "Push HTML content to the Live Canvas for visual output. Actions: push, clear, eval."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["push", "clear", "eval"],
                    "description": "Action to perform on the canvas"
                },
                "content": {
                    "type": "string",
                    "description": "HTML content to push (required for 'push' action)"
                },
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute (required for 'eval' action)"
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

        match action {
            "push" => {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing content for push action"))?;

                self.app_handle.emit("canvas-push", content)?;
                self.state_manager.save(content).await?;

                Ok(ToolResult::success(format!(
                    "Pushed {} bytes to canvas",
                    content.len()
                )))
            }
            "clear" => {
                self.app_handle.emit("canvas-clear", ())?;
                self.state_manager.clear().await?;

                Ok(ToolResult::success("Canvas cleared"))
            }
            "eval" => {
                let code = params
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing code for eval action"))?;

                self.app_handle.emit("canvas-eval", code)?;

                Ok(ToolResult::success(format!(
                    "Executed {} bytes of JavaScript in canvas",
                    code.len()
                )))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
