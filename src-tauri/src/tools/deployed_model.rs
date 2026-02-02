use crate::ipc::SidecarHandle;
use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Dynamic tool wrapper for a deployed trained model.
///
/// Each deployed model becomes a callable tool that the agent can invoke
/// for inference. Created by `TrainingTool` when a model is deployed.
pub struct DeployedModelTool {
    tool_name: String,
    description: String,
    task_type: String,
    sidecar: Arc<Mutex<SidecarHandle>>,
}

impl DeployedModelTool {
    pub fn new(tool_name: String, task_type: String, sidecar: Arc<Mutex<SidecarHandle>>) -> Self {
        let description = format!(
            "Run {} inference using deployed model '{}'.",
            task_type, tool_name,
        );
        Self {
            tool_name,
            description,
            task_type,
            sidecar,
        }
    }
}

#[async_trait]
impl Tool for DeployedModelTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "description": "Input data for the model. Array of numbers or nested arrays.",
                    "oneOf": [
                        { "type": "array", "items": { "type": "number" } },
                        {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "number" }
                            }
                        }
                    ]
                }
            },
            "required": ["input"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolResult> {
        let input = &params["input"];
        if input.is_null() {
            return Ok(ToolResult::error("Missing 'input' parameter"));
        }

        let mut sidecar = self.sidecar.lock().await;
        let res = sidecar
            .request(
                "training.predict",
                json!({
                    "tool_name": self.tool_name,
                    "input_data": input,
                }),
            )
            .await?;

        let summary = if self.task_type == "classification" {
            format!(
                "Prediction: class {} (confidence: {:.2}%)",
                res["prediction"],
                res["confidence"].as_f64().unwrap_or(0.0) * 100.0
            )
        } else {
            format!("Prediction result from '{}'", self.tool_name)
        };

        Ok(ToolResult::success(summary).with_data(res))
    }

    fn permission_tier(&self) -> PermissionTier {
        // Read-only inference — auto-approved
        PermissionTier::Low
    }
}
