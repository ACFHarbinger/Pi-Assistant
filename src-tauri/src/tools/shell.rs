//! Shell tool: execute shell commands.

use super::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;
use tokio::process::Command;
use tracing::{info, warn};

/// Shell command execution tool.
pub struct ShellTool {
    default_timeout: Duration,
}

impl ShellTool {
    pub fn new() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute shell commands. Use for running scripts, git, npm, cargo, etc."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory (optional)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)"
                },
                "sandbox": {
                    "type": "boolean",
                    "description": "Run the command in a containerized sandbox (Docker) for safety (default: false)"
                },
                "image": {
                    "type": "string",
                    "description": "Docker image to use for sandbox (default: ubuntu:latest)"
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolResult> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;

        let working_dir = params.get("working_dir").and_then(|v| v.as_str());
        let sandbox = params
            .get("sandbox")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let image = params
            .get("image")
            .and_then(|v| v.as_str())
            .unwrap_or("ubuntu:latest");

        let timeout_secs = params
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_timeout.as_secs());

        info!(command = %command, working_dir = ?working_dir, sandbox = sandbox, "Executing shell command");

        let mut cmd = if sandbox {
            let mut d_cmd = Command::new("docker");
            d_cmd.arg("run").arg("--rm");

            // Map working directory
            let host_dir = if let Some(dir) = working_dir {
                std::path::Path::new(dir)
                    .canonicalize()?
                    .to_string_lossy()
                    .to_string()
            } else {
                std::env::current_dir()?
                    .canonicalize()?
                    .to_string_lossy()
                    .to_string()
            };

            d_cmd.arg("-v").arg(format!("{}:/workspace", host_dir));
            d_cmd.arg("-w").arg("/workspace");

            // Map user if on Unix
            #[cfg(unix)]
            {
                let uid = unsafe { libc::getuid() };
                let gid = unsafe { libc::getgid() };
                d_cmd.arg("--user").arg(format!("{}:{}", uid, gid));
            }

            d_cmd.arg(image).arg("sh").arg("-c").arg(command);
            d_cmd
        } else {
            let mut s_cmd = Command::new("sh");
            s_cmd.arg("-c").arg(command);
            if let Some(dir) = working_dir {
                s_cmd.current_dir(dir);
            }
            s_cmd
        };

        cmd.env("PAGER", "cat"); // Disable paging

        let start = std::time::Instant::now();

        let output = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

        let duration = start.elapsed();

        match output {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);

                let combined = if stderr.is_empty() {
                    stdout.clone()
                } else {
                    format!("{}\n[stderr]\n{}", stdout, stderr)
                };

                if output.status.success() {
                    Ok(ToolResult::success(combined).with_data(serde_json::json!({
                        "exit_code": exit_code,
                        "duration_ms": duration.as_millis() as u64,
                        "stdout": stdout,
                        "stderr": stderr,
                    })))
                } else {
                    warn!(exit_code = exit_code, "Command failed");
                    Ok(ToolResult {
                        success: false,
                        output: combined,
                        error: Some(format!("Exit code: {}", exit_code)),
                        data: Some(serde_json::json!({
                            "exit_code": exit_code,
                            "duration_ms": duration.as_millis() as u64,
                        })),
                    })
                }
            }
            Ok(Err(e)) => Ok(ToolResult::error(format!(
                "Failed to execute command: {}",
                e
            ))),
            Err(_) => Ok(ToolResult::error(format!(
                "Command timed out after {}s",
                timeout_secs
            ))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}
