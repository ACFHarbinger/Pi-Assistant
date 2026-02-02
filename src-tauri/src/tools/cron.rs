use crate::cron::CronManager;
use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

pub struct CronTool {
    cron_manager: Arc<CronManager>,
}

impl CronTool {
    pub fn new(cron_manager: Arc<CronManager>) -> Self {
        Self { cron_manager }
    }
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn description(&self) -> &str {
        "Schedule or manage periodic tasks. Use standard cron syntax (e.g., '0 */5 * * * *' for every 5 minutes). Actions: add, remove, list."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list"]
                },
                "schedule": {
                    "type": "string",
                    "description": "Cron expression (required for 'add')"
                },
                "task_description": {
                    "type": "string",
                    "description": "Description of what the agent should do when triggered (required for 'add')"
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Defaults to UTC if not specified."
                },
                "id": {
                    "type": "string",
                    "description": "Job UUID (required for 'remove')"
                }
            },
            "required": ["action"]
        })
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        match action {
            "add" => {
                let schedule = params
                    .get("schedule")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing schedule"))?;
                let task_desc = params
                    .get("task_description")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing task_description"))?;
                let timezone = params
                    .get("timezone")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let id = self
                    .cron_manager
                    .add_job(schedule.to_string(), task_desc.to_string(), timezone)
                    .await?;
                Ok(
                    ToolResult::success(format!("Scheduled task '{}' with ID: {}", task_desc, id))
                        .with_data(json!({ "id": id })),
                )
            }
            "list" => {
                let jobs = self.cron_manager.list_jobs().await;
                Ok(ToolResult::success(format!("Current jobs: {}", jobs.len()))
                    .with_data(json!({ "jobs": jobs })))
            }
            "remove" => {
                let id_str = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing job ID"))?;
                let id = id_str
                    .parse::<uuid::Uuid>()
                    .map_err(|_| anyhow::anyhow!("Invalid job ID format"))?;
                self.cron_manager.remove_job(id).await?;
                Ok(ToolResult::success(format!(
                    "Job {} successfully removed",
                    id
                )))
            }
            _ => Err(anyhow::anyhow!("Unsupported action: {}", action)),
        }
    }
}
