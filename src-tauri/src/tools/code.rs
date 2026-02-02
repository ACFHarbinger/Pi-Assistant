//! Code tool: file read/write/patch operations.

use super::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::info;

/// Code file operations tool.
pub struct CodeTool {
    blocked_paths: Vec<PathBuf>,
}

impl CodeTool {
    pub fn new() -> Self {
        Self {
            blocked_paths: vec![
                PathBuf::from("/etc"),
                PathBuf::from("/sys"),
                PathBuf::from("/proc"),
                PathBuf::from("/boot"),
                dirs::home_dir().map(|h| h.join(".ssh")).unwrap_or_default(),
                dirs::home_dir()
                    .map(|h| h.join(".gnupg"))
                    .unwrap_or_default(),
                dirs::home_dir().map(|h| h.join(".aws")).unwrap_or_default(),
            ],
        }
    }

    fn is_path_blocked(&self, path: &Path) -> bool {
        // Block path traversal
        if path.to_string_lossy().contains("..") {
            return true;
        }

        // Check against blocked paths
        for blocked in &self.blocked_paths {
            if !blocked.as_os_str().is_empty() && path.starts_with(blocked) {
                return true;
            }
        }

        false
    }
}

#[async_trait]
impl Tool for CodeTool {
    fn name(&self) -> &str {
        "code"
    }

    fn description(&self) -> &str {
        "Read, write, and patch files. Actions: read, write, patch, list, create_dir."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["action", "path"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "patch", "list", "create_dir"],
                    "description": "The file operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content for write action"
                },
                "old": {
                    "type": "string",
                    "description": "Text to find for patch action"
                },
                "new": {
                    "type": "string",
                    "description": "Replacement text for patch action"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for list action"
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        let path_str = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        let path = PathBuf::from(path_str);

        // Security check
        if self.is_path_blocked(&path) {
            return Ok(ToolResult::error(format!("Access denied: {}", path_str)));
        }

        match action {
            "read" => self.read_file(&path).await,
            "write" => {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'content' parameter"))?;
                self.write_file(&path, content).await
            }
            "patch" => {
                let old = params
                    .get("old")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'old' parameter"))?;
                let new = params
                    .get("new")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'new' parameter"))?;
                self.patch_file(&path, old, new).await
            }
            "list" => {
                let pattern = params.get("pattern").and_then(|v| v.as_str());
                self.list_dir(&path, pattern).await
            }
            "create_dir" => self.create_dir(&path).await,
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}

impl CodeTool {
    async fn read_file(&self, path: &Path) -> Result<ToolResult> {
        info!(path = %path.display(), "Reading file");

        match fs::read_to_string(path).await {
            Ok(content) => Ok(ToolResult::success(content)),
            Err(e) => Ok(ToolResult::error(format!("Failed to read file: {}", e))),
        }
    }

    async fn write_file(&self, path: &Path, content: &str) -> Result<ToolResult> {
        info!(path = %path.display(), "Writing file");

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        match fs::write(path, content).await {
            Ok(_) => Ok(ToolResult::success(format!(
                "Wrote {} bytes to {}",
                content.len(),
                path.display()
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
    }

    async fn patch_file(&self, path: &Path, old: &str, new: &str) -> Result<ToolResult> {
        info!(path = %path.display(), "Patching file");

        let content = match fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read file: {}", e))),
        };

        if !content.contains(old) {
            return Ok(ToolResult::error("Target string not found in file"));
        }

        let new_content = content.replace(old, new);
        match fs::write(path, &new_content).await {
            Ok(_) => Ok(ToolResult::success(format!("Patched {}", path.display()))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
    }

    async fn list_dir(&self, path: &Path, _pattern: Option<&str>) -> Result<ToolResult> {
        info!(path = %path.display(), "Listing directory");

        let mut entries = match fs::read_dir(path).await {
            Ok(e) => e,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to read directory: {}",
                    e
                )))
            }
        };

        let mut items = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            items.push(format!("{}{}", name, if is_dir { "/" } else { "" }));
        }

        items.sort();
        Ok(ToolResult::success(items.join("\n")))
    }

    async fn create_dir(&self, path: &Path) -> Result<ToolResult> {
        info!(path = %path.display(), "Creating directory");

        match fs::create_dir_all(path).await {
            Ok(_) => Ok(ToolResult::success(format!("Created {}", path.display()))),
            Err(e) => Ok(ToolResult::error(format!(
                "Failed to create directory: {}",
                e
            ))),
        }
    }
}

impl Default for CodeTool {
    fn default() -> Self {
        Self::new()
    }
}
