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

/// Permission engine for tool execution.
pub struct PermissionEngine {
    rules: PermissionRules,
    /// User overrides: pattern -> allowed
    user_overrides: HashMap<String, bool>,
    /// Cached decisions for this session
    session_cache: HashMap<String, bool>,
}

impl PermissionEngine {
    /// Create a new permission engine with default rules.
    pub fn new() -> Self {
        Self {
            rules: PermissionRules::default(),
            user_overrides: HashMap::new(),
            session_cache: HashMap::new(),
        }
    }

    /// Check if a tool call is allowed.
    pub fn check(&self, call: &ToolCall) -> Result<PermissionResult> {
        let pattern = call.pattern_key();
        debug!(pattern = %pattern, tool = %call.tool_name, "Checking permission");

        // 1. Check user overrides first
        if let Some(&allowed) = self.user_overrides.get(&pattern) {
            if allowed {
                info!(pattern = %pattern, "Allowed by user override");
                return Ok(PermissionResult::Allowed);
            } else {
                return Ok(PermissionResult::Denied("Denied by user override".into()));
            }
        }

        // 2. Check session cache
        if let Some(&allowed) = self.session_cache.get(&pattern) {
            if allowed {
                return Ok(PermissionResult::Allowed);
            } else {
                return Ok(PermissionResult::NeedsApproval);
            }
        }

        // 3. Apply rules
        match call.tool_name.as_str() {
            "shell" => self.check_shell_command(call),
            "code" => self.check_code_operation(call),
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
        if action == "read" || action == "list" {
            if !self.rules.is_path_blocked(path) {
                return Ok(PermissionResult::Allowed);
            }
        }

        // Write operations need approval
        if self.rules.is_path_blocked(path) {
            return Ok(PermissionResult::Denied("Path is blocked".into()));
        }

        Ok(PermissionResult::NeedsApproval)
    }

    /// Add a user override for a pattern.
    pub fn add_user_override(&mut self, pattern: &str, allowed: bool) {
        info!(pattern = %pattern, allowed = allowed, "Adding user override");
        self.user_overrides.insert(pattern.to_string(), allowed);
    }

    /// Cache a decision for the current session.
    pub fn cache_decision(&mut self, pattern: &str, allowed: bool) {
        self.session_cache.insert(pattern.to_string(), allowed);
    }

    /// Clear session cache.
    pub fn clear_session_cache(&mut self) {
        self.session_cache.clear();
    }
}

impl Default for PermissionEngine {
    fn default() -> Self {
        Self::new()
    }
}
