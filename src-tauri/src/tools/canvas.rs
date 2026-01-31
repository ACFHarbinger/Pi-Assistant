//! Canvas tool for agent to push content to the Live Canvas.

use async_trait::async_trait;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

use crate::tools::{PermissionTier, Tool, ToolResult};

/// Tool that allows the agent to push HTML/React to the canvas.
pub struct CanvasTool {
    app_handle: AppHandle,
}

impl CanvasTool {
    pub fn new(app_handle: AppHandle) -> Self {
        Self { app_handle }
    }
}

#[async_trait]
impl Tool for CanvasTool {
    fn name(&self) -> &str {
        "canvas"
    }

    fn description(&self) -> &str {
        "Push HTML content to the Live Canvas for visual output. Actions: push, clear, snapshot."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["push", "clear"],
                    "description": "Action to perform on the canvas"
                },
                "content": {
                    "type": "string",
                    "description": "HTML content to push (required for 'push' action)"
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

                Ok(ToolResult::success(format!(
                    "Pushed {} bytes to canvas",
                    content.len()
                )))
            }
            "clear" => {
                self.app_handle.emit("canvas-clear", ())?;

                Ok(ToolResult::success("Canvas cleared"))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
