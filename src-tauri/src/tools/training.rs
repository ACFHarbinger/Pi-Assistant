use crate::ipc::SidecarHandle;
use crate::tools::deployed_model::DeployedModelTool;
use crate::tools::{PermissionTier, Tool, ToolContext, ToolRegistry, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Tool for managing ML/RL training pipelines via the sidecar.
///
/// Supports training, deploying trained models as inference tools,
/// and running predictions on deployed models.
pub struct TrainingTool {
    sidecar: Arc<Mutex<SidecarHandle>>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
}

impl TrainingTool {
    pub fn new(
        sidecar: Arc<Mutex<SidecarHandle>>,
        tool_registry: Arc<RwLock<ToolRegistry>>,
    ) -> Self {
        Self {
            sidecar,
            tool_registry,
        }
    }
}

#[async_trait]
impl Tool for TrainingTool {
    fn name(&self) -> &str {
        "train"
    }

    fn description(&self) -> &str {
        "Manage ML/RL training pipeline: start, stop, check status, list runs, deploy trained models as tools, and run predictions."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["start", "stop", "status", "list", "deploy", "predict", "list_deployed"]
                },
                "run_id": {
                    "type": "string",
                    "description": "ID of the training run (required for 'stop', 'status', 'deploy')"
                },
                "config": {
                    "type": "object",
                    "description": "Training configuration (required for 'start'). Should include 'backbone', 'head', 'backbone_config', 'head_config', 'training', 'data'."
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name for the deployed model tool (required for 'deploy', 'predict')"
                },
                "device": {
                    "type": "string",
                    "description": "Target device for deployment (optional, e.g. 'cpu', 'cuda:0')"
                },
                "input": {
                    "description": "Input data for prediction (required for 'predict')"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolResult> {
        let action = params["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        match action {
            "start" => {
                let config = params["config"].clone();
                if config.is_null() {
                    return Ok(ToolResult::error("Missing 'config' for 'start' action"));
                }
                let mut sidecar = self.sidecar.lock().await;
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
                let mut sidecar = self.sidecar.lock().await;
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
                let mut sidecar = self.sidecar.lock().await;
                let res = sidecar
                    .request("training.status", json!({ "run_id": run_id }))
                    .await?;
                Ok(ToolResult::success(format!("Status for run {}", run_id)).with_data(res))
            }
            "list" => {
                let mut sidecar = self.sidecar.lock().await;
                let res = sidecar.request("training.list", json!({})).await?;
                Ok(ToolResult::success("List of all training runs").with_data(res))
            }
            "deploy" => {
                let run_id = params["run_id"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing run_id for deploy"))?;
                let tool_name = params["tool_name"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_name for deploy"))?;
                let device = params["device"].as_str();

                // Call sidecar to deploy (load checkpoint + register in Python registry)
                let deploy_params = json!({
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "device": device,
                });

                let res = {
                    let mut sidecar = self.sidecar.lock().await;
                    sidecar.request("training.deploy", deploy_params).await?
                };
                // Sidecar lock is dropped here before acquiring registry write lock

                let task_type = res["task_type"]
                    .as_str()
                    .unwrap_or("classification")
                    .to_string();

                // Auto-register the deployed model as a callable tool in the Rust registry
                let deployed_tool = Arc::new(DeployedModelTool::new(
                    tool_name.to_string(),
                    task_type,
                    self.sidecar.clone(),
                ));

                self.tool_registry.write().await.register(deployed_tool);

                Ok(ToolResult::success(format!(
                    "Deployed run {} as tool '{}' on {}",
                    run_id,
                    tool_name,
                    res["device"].as_str().unwrap_or("unknown")
                ))
                .with_data(res))
            }
            "predict" => {
                let tool_name = params["tool_name"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_name for predict"))?;
                let input = &params["input"];
                if input.is_null() {
                    return Ok(ToolResult::error("Missing 'input' for predict"));
                }

                let mut sidecar = self.sidecar.lock().await;
                let res = sidecar
                    .request(
                        "training.predict",
                        json!({
                            "tool_name": tool_name,
                            "input_data": input,
                        }),
                    )
                    .await?;

                Ok(ToolResult::success(format!("Prediction from '{}'", tool_name)).with_data(res))
            }
            "list_deployed" => {
                let mut sidecar = self.sidecar.lock().await;
                let res = sidecar.request("training.list_deployed", json!({})).await?;
                Ok(ToolResult::success("List of deployed models").with_data(res))
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
