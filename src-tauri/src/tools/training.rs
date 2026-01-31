use crate::ipc::SidecarHandle;
use crate::tools::{PermissionTier, Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Tool for managing ML/RL training pipelines via the sidecar.
pub struct TrainingTool {
    sidecar: Arc<Mutex<SidecarHandle>>,
}

impl TrainingTool {
    pub fn new(sidecar: Arc<Mutex<SidecarHandle>>) -> Self {
        Self { sidecar }
    }
}

#[async_trait]
impl Tool for TrainingTool {
    fn name(&self) -> &str {
        "train"
    }

    fn description(&self) -> &str {
        "Manage ML/RL training pipeline: start, stop, check status, or list runs."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'start', 'stop', 'status', 'list'",
                    "enum": ["start", "stop", "status", "list"]
                },
                "run_id": {
                    "type": "string",
                    "description": "ID of the training run (required for 'stop' and 'status')"
                },
                "config": {
                    "type": "object",
                    "description": "Training configuration (required for 'start'). Should include 'backbone', 'head', 'backbone_config', 'head_config', 'training', 'data'."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let action = params["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        let mut sidecar = self.sidecar.lock().await;

        match action {
            "start" => {
                let config = params["config"].clone();
                if config.is_null() {
                    return Ok(ToolResult::error("Missing 'config' for 'start' action"));
                }
                let res = sidecar.request("training.start", config).await?;
                Ok(
                    ToolResult::success(format!("Started training run: {}", res["run_id"]))
                        .with_data(res),
                )
            }
            "stop" => {
                let run_id = params["run_id"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing run_id"))?;
                let res = sidecar
                    .request("training.stop", json!({ "run_id": run_id }))
                    .await?;
                Ok(
                    ToolResult::success(format!("Stop command sent for run: {}", run_id))
                        .with_data(res),
                )
            }
            "status" => {
                let run_id = params["run_id"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing run_id"))?;
                let res = sidecar
                    .request("training.status", json!({ "run_id": run_id }))
                    .await?;
                Ok(ToolResult::success(format!("Status for run {}", run_id)).with_data(res))
            }
            "list" => {
                let res = sidecar.request("training.list", json!({})).await?;
                Ok(ToolResult::success("List of all training runs").with_data(res))
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
