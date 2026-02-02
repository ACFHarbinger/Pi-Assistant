use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use sysinfo::System;
use tokio::sync::Mutex;

use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};

/// Tool for monitoring system resources (CPU, Memory, Disk, Network).
pub struct SystemTool {
    sys: Arc<Mutex<System>>,
}

impl SystemTool {
    pub fn new() -> Self {
        Self {
            sys: Arc::new(Mutex::new(System::new_all())),
        }
    }

    async fn get_status(&self) -> ToolResult {
        let data = self.get_system_status_snapshot().await;
        ToolResult::success("System status retrieved".to_string()).with_data(data)
    }

    pub async fn get_system_status_snapshot(&self) -> Value {
        let mut sys = self.sys.lock().await;
        sys.refresh_all();

        // CPU
        let cpu_usage = sys.global_cpu_usage();

        // Memory
        let total_mem = sys.total_memory();
        let used_mem = sys.used_memory();

        // Swap
        let total_swap = sys.total_swap();
        let used_swap = sys.used_swap();

        // Uptime
        let uptime = System::uptime();

        // Load Avg
        let load_avg = System::load_average();

        json!({
            "cpu_usage_percent": cpu_usage,
            "memory": {
                "total": total_mem,
                "used": used_mem,
                "free": sys.free_memory(),
                "percent": (used_mem as f64 / total_mem as f64) * 100.0,
            },
            "swap": {
                "total": total_swap,
                "used": used_swap,
                "percent": if total_swap > 0 { (used_swap as f64 / total_swap as f64) * 100.0 } else { 0.0 },
            },
            "uptime_secs": uptime,
            "load_average": {
                "one": load_avg.one,
                "five": load_avg.five,
                "fifteen": load_avg.fifteen,
            },
            "os_name": System::name(),
            "os_version": System::os_version(),
            "host_name": System::host_name(),
            "kernel_version": System::kernel_version(),
        })
    }

    async fn list_processes(&self, limit: usize, sort_by_mem: bool) -> ToolResult {
        let mut sys = self.sys.lock().await;
        sys.refresh_all();

        let mut procs: Vec<_> = sys.processes().values().collect();

        if sort_by_mem {
            procs.sort_by(|a, b| b.memory().cmp(&a.memory()));
        } else {
            // Sort by CPU
            procs.sort_by(|a, b| {
                b.cpu_usage()
                    .partial_cmp(&a.cpu_usage())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let top_procs: Vec<_> = procs
            .into_iter()
            .take(limit)
            .map(|p| {
                json!({
                    "pid": p.pid().as_u32(),
                    "name": p.name(),
                    "cpu_usage": p.cpu_usage(),
                    "memory": p.memory(),
                    "status": format!("{:?}", p.status()),
                })
            })
            .collect();

        ToolResult::success(format!("Top {} processes retrieved", limit))
            .with_data(json!(top_procs))
    }
}

#[async_trait]
impl Tool for SystemTool {
    fn name(&self) -> &str {
        "system"
    }

    fn description(&self) -> &str {
        "Monitor system resources and processes. Actions: status, processes."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "processes"],
                    "description": "Action to perform"
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit for process list (default: 10)",
                    "minimum": 1
                },
                "sort_by_memory": {
                    "type": "boolean",
                    "description": "Sort processes by memory instead of CPU (default: false)"
                }
            }
        })
    }

    async fn execute(&self, params: Value, _context: ToolContext) -> anyhow::Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "status" => Ok(self.get_status().await),
            "processes" => {
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let sort_mem = params
                    .get("sort_by_memory")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Ok(self.list_processes(limit, sort_mem).await)
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium // Getting process info is generally safe provided we don't kill them (action not exposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_tool_status() {
        let tool = SystemTool::new();
        // First refresh might take a clearer snapshot if done twice, but once is enough for structure check
        let res = tool
            .execute(
                json!({"action": "status"}),
                ToolContext {
                    transactions: None,
                    memory: None,
                    session_id: uuid::Uuid::nil(),
                },
            )
            .await
            .unwrap();
        assert!(res.success);
        let data = res.data.unwrap();
        assert!(data.get("cpu_usage_percent").is_some());
        assert!(data.get("memory").is_some());
        assert!(data.get("uptime_secs").is_some());
    }

    #[tokio::test]
    async fn test_system_tool_processes() {
        let tool = SystemTool::new();
        let res = tool
            .execute(
                json!({
                    "action": "processes",
                    "limit": 5,
                    "sort_by_memory": true
                }),
                ToolContext {
                    transactions: None,
                    memory: None,
                    session_id: uuid::Uuid::nil(),
                },
            )
            .await
            .unwrap();

        assert!(res.success);
        let data = res.data.unwrap();
        let procs = data.as_array().unwrap();
        assert_eq!(procs.len(), 5);
        assert!(procs[0].get("pid").is_some());
        assert!(procs[0].get("cpu_usage").is_some());
    }
}
