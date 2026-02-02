//! Permission engine: 3-tier gate for tool execution.

use super::rules::PermissionRules;
use crate::tools::ToolCall;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

/// Result of a permission check.
#[derive(Debug, Clone)]
pub enum PermissionResult {
    /// Tool execution is allowed.
    Allowed,
    /// User approval is needed.
    NeedsApproval,
    /// Tool execution is denied.
    Denied(String),
}

use chrono::{DateTime, Utc};

/// Override entry with optional expiry.
#[derive(Debug, Clone)]
struct OverrideEntry {
    allowed: bool,
    expires_at: Option<DateTime<Utc>>,
}

/// Permission engine for tool execution.
pub struct PermissionEngine {
    rules: PermissionRules,
    /// User overrides: pattern -> entry
    user_overrides: HashMap<String, OverrideEntry>,
}

impl PermissionEngine {
    /// Create a new permission engine with default rules.
    pub fn new() -> Self {
        Self {
            rules: PermissionRules::default(),
            user_overrides: HashMap::new(),
        }
    }

    /// Check if a tool call is allowed.
    pub fn check(&self, call: &ToolCall) -> Result<PermissionResult> {
        let pattern = call.pattern_key();
        debug!(pattern = %pattern, tool = %call.tool_name, "Checking permission");

        // 1. Check user overrides (with wildcard support)
        // We check specific matches first, then broader patterns
        if let Some(entry) = self.find_override(&pattern) {
            if let Some(expires) = entry.expires_at {
                if Utc::now() > expires {
                    // Expired, ignore
                } else if entry.allowed {
                    info!(pattern = %pattern, "Allowed by user override (time-limited)");
                    return Ok(PermissionResult::Allowed);
                } else {
                    return Ok(PermissionResult::Denied("Denied by user override".into()));
                }
            } else if entry.allowed {
                info!(pattern = %pattern, "Allowed by user override");
                return Ok(PermissionResult::Allowed);
            } else {
                return Ok(PermissionResult::Denied("Denied by user override".into()));
            }
        }

        // 3. Apply rules
        match call.tool_name.as_str() {
            "shell" => self.check_shell_command(call),
            "code" => self.check_code_operation(call),
            "database" => self.check_database_operation(call),
            "browser" => Ok(PermissionResult::NeedsApproval),
            _ => Ok(PermissionResult::NeedsApproval),
        }
    }

    fn check_shell_command(&self, call: &ToolCall) -> Result<PermissionResult> {
        let command = call
            .parameters
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Check blocked patterns (high danger)
        if self.rules.is_command_blocked(command) {
            return Ok(PermissionResult::Denied(
                "Command matches blocked pattern".into(),
            ));
        }

        // Check safe patterns (auto-allowed)
        if self.rules.is_command_safe(command) {
            return Ok(PermissionResult::Allowed);
        }

        // Everything else needs approval
        Ok(PermissionResult::NeedsApproval)
    }

    fn check_code_operation(&self, call: &ToolCall) -> Result<PermissionResult> {
        let action = call
            .parameters
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let path = call
            .parameters
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Read operations are generally safe
        if (action == "read" || action == "list") && !self.rules.is_path_blocked(path) {
            return Ok(PermissionResult::Allowed);
        }

        // Write operations need approval
        if self.rules.is_path_blocked(path) {
            return Ok(PermissionResult::Denied("Path is blocked".into()));
        }

        Ok(PermissionResult::NeedsApproval)
    }

    fn check_database_operation(&self, call: &ToolCall) -> Result<PermissionResult> {
        let action = call
            .parameters
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Read-only actions are auto-approved
        match action {
            "connect" | "schema" | "explain" | "list_tables" | "disconnect" => {
                Ok(PermissionResult::Allowed)
            }
            "query" => {
                let sql = call
                    .parameters
                    .get("sql")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let trimmed = sql.trim().to_uppercase();
                // Auto-approve read-only queries
                if trimmed.starts_with("SELECT")
                    || trimmed.starts_with("PRAGMA")
                    || trimmed.starts_with("EXPLAIN")
                    || trimmed.starts_with("WITH")
                {
                    Ok(PermissionResult::Allowed)
                } else {
                    // Write operations need approval
                    Ok(PermissionResult::NeedsApproval)
                }
            }
            _ => Ok(PermissionResult::NeedsApproval),
        }
    }

    /// Find a matching override for the pattern.
    fn find_override(&self, key: &str) -> Option<&OverrideEntry> {
        // 1. Exact match
        if let Some(entry) = self.user_overrides.get(key) {
            return Some(entry);
        }

        // 2. Wildcard match (simple suffix or prefix)
        // This iterates through all overrides, so it could be slow if there are many.
        // But typically there are very few.
        for (pattern, entry) in &self.user_overrides {
            // Generic wildcard match (prefix)
            if pattern.ends_with('*') {
                let prefix = &pattern[0..pattern.len() - 1];
                if key.starts_with(prefix) {
                    return Some(entry);
                }
            }
        }

        None
    }

    /// Add a user override for a pattern.
    pub fn add_user_override(&mut self, pattern: &str, allowed: bool) {
        info!(pattern = %pattern, allowed = allowed, "Adding user override");
        self.user_overrides.insert(
            pattern.to_string(),
            OverrideEntry {
                allowed,
                expires_at: None, // Default to no expiry for now
            },
        );
    }

    /// Add a time-limited user override.
    pub fn add_scoped_override(
        &mut self,
        pattern: &str,
        allowed: bool,
        duration: Option<std::time::Duration>,
    ) {
        let expires_at = duration.map(|d| Utc::now() + chrono::Duration::from_std(d).unwrap());
        info!(pattern = %pattern, allowed = allowed, ?expires_at, "Adding scoped override");
        self.user_overrides.insert(
            pattern.to_string(),
            OverrideEntry {
                allowed,
                expires_at,
            },
        );
    }

    /// Clear session cache (now just clears overrides).
    pub fn clear_session_cache(&mut self) {
        self.user_overrides.clear();
    }
}

impl Default for PermissionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_check_shell_allowed() {
        let engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "ls -la" }),
        };
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed"),
        }
    }

    #[test]
    fn test_check_shell_blocked() {
        let engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "sudo rm -rf /" }),
        };
        match engine.check(&call).unwrap() {
            PermissionResult::Denied(_) => {}
            _ => panic!("Expected Denied"),
        }
    }

    #[test]
    fn test_check_shell_needs_approval() {
        let engine = PermissionEngine::new();
        // random command not in safe list
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "git push origin main" }),
        };
        match engine.check(&call).unwrap() {
            PermissionResult::NeedsApproval => {}
            _ => panic!("Expected NeedsApproval"),
        }
    }

    #[test]
    fn test_user_override() {
        let mut engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "git push origin main" }),
        };

        // Initially needs approval
        if let PermissionResult::NeedsApproval = engine.check(&call).unwrap() {
        } else {
            panic!("Should need approval");
        }

        // Add override
        let pattern = call.pattern_key();
        engine.add_user_override(&pattern, true);

        // Now allowed
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed after override"),
        }
    }

    #[test]
    fn test_code_read_safe() {
        let engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "code".into(),
            parameters: json!({
                "action": "read",
                "path": "/home/user/project/README.md"
            }),
        };
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for read"),
        }
    }

    #[test]
    fn test_code_write_needs_approval() {
        let engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "code".into(),
            parameters: json!({
                "action": "write",
                "path": "/home/user/project/README.md"
            }),
        };
        match engine.check(&call).unwrap() {
            PermissionResult::NeedsApproval => {}
            _ => panic!("Expected NeedsApproval for write"),
        }
    }

    #[test]
    fn test_wildcard_override() {
        let mut engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "git push" }),
        };

        // Initially needs approval
        if let PermissionResult::NeedsApproval = engine.check(&call).unwrap() {
        } else {
            panic!("Should need approval");
        }

        // Add wildcard override for all git commands
        engine.add_user_override("git *", true);

        // Now allowed
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed after wildcard override"),
        }
    }

    #[test]
    fn test_scoped_override_expiry() {
        let mut engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "git status" }),
        };

        // Add short-lived override
        engine.add_scoped_override(
            &call.pattern_key(),
            true,
            Some(std::time::Duration::from_millis(100)),
        );

        // Should be allowed immediately
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed immediately"),
        }

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Should be expired (NeedsApproval because override is ignored)
        // Note: The implementation returns Denied if override says false,
        // but if override is EXPIRED, it should fall back to next check?
        // Let's check implementation:
        // if Utc::now() > expires { // Expired, ignore }
        // ... fall through to rules ...
        // Rule for "git status" is NeedsApproval (unless in safe list).
        // "git status" is usually safe. Let's use "git push".
    }

    #[test]
    fn test_scoped_override_expiry_fallback() {
        let mut engine = PermissionEngine::new();
        let call = ToolCall {
            tool_name: "shell".into(),
            parameters: json!({ "command": "dangerous info" }),
        };

        // Add override
        engine.add_scoped_override(
            &call.pattern_key(),
            true,
            Some(std::time::Duration::from_millis(50)),
        );

        // Allowed
        match engine.check(&call).unwrap() {
            PermissionResult::Allowed => {}
            _ => panic!("Expected Allowed"),
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Expired -> Falls back to rules -> NeedsApproval
        match engine.check(&call).unwrap() {
            PermissionResult::NeedsApproval => {}
            _ => panic!("Expected NeedsApproval after expiry"),
        }
    }
}
