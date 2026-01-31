//! Permission rules: pattern matching for commands and paths.

use regex::Regex;
use std::path::Path;

/// Permission rules for tool execution.
pub struct PermissionRules {
    /// Commands that are always blocked.
    blocked_commands: Vec<Regex>,
    /// Commands that are always safe.
    safe_commands: Vec<Regex>,
    /// Paths that are always blocked.
    blocked_paths: Vec<String>,
}

impl PermissionRules {
    /// Create new rules with defaults.
    pub fn new() -> Self {
        Self {
            blocked_commands: vec![
                // System destructive commands
                Regex::new(r"^\s*rm\s+(-[rf]+\s+)?/").unwrap(),
                Regex::new(r"^\s*rm\s+-rf\s+~").unwrap(),
                Regex::new(r"^\s*dd\s+").unwrap(),
                Regex::new(r"^\s*mkfs").unwrap(),
                Regex::new(r"^\s*:[()\s]*\{\s*:\|:\s*&\s*\}").unwrap(), // fork bomb
                Regex::new(r">\s*/dev/sd[a-z]").unwrap(),
                Regex::new(r"curl.*\|\s*(sudo\s+)?sh").unwrap(),
                Regex::new(r"wget.*\|\s*(sudo\s+)?sh").unwrap(),
                // Privilege escalation
                Regex::new(r"^\s*sudo\s+").unwrap(),
                Regex::new(r"^\s*su\s+").unwrap(),
                Regex::new(r"^\s*pkexec\s+").unwrap(),
                // System modification
                Regex::new(r"^\s*systemctl\s+(start|stop|enable|disable|restart)").unwrap(),
                Regex::new(r"^\s*apt\s+(install|remove|purge)").unwrap(),
                Regex::new(r"^\s*apt-get\s+(install|remove|purge)").unwrap(),
                Regex::new(r"^\s*dnf\s+(install|remove)").unwrap(),
                Regex::new(r"^\s*yum\s+(install|remove)").unwrap(),
                Regex::new(r"^\s*pacman\s+-[SRU]").unwrap(),
            ],
            safe_commands: vec![
                // Read-only commands
                Regex::new(r"^\s*ls\s").unwrap(),
                Regex::new(r"^\s*ls$").unwrap(),
                Regex::new(r"^\s*cat\s").unwrap(),
                Regex::new(r"^\s*head\s").unwrap(),
                Regex::new(r"^\s*tail\s").unwrap(),
                Regex::new(r"^\s*less\s").unwrap(),
                Regex::new(r"^\s*more\s").unwrap(),
                Regex::new(r"^\s*wc\s").unwrap(),
                Regex::new(r"^\s*grep\s").unwrap(),
                Regex::new(r"^\s*find\s").unwrap(),
                Regex::new(r"^\s*which\s").unwrap(),
                Regex::new(r"^\s*type\s").unwrap(),
                Regex::new(r"^\s*pwd$").unwrap(),
                Regex::new(r"^\s*echo\s").unwrap(),
                Regex::new(r"^\s*date$").unwrap(),
                Regex::new(r"^\s*whoami$").unwrap(),
                Regex::new(r"^\s*id$").unwrap(),
                Regex::new(r"^\s*env$").unwrap(),
                Regex::new(r"^\s*printenv").unwrap(),
                // Development tools (read)
                Regex::new(r"^\s*git\s+(status|log|diff|branch|remote|show)").unwrap(),
                Regex::new(r"^\s*cargo\s+(check|clippy|test|build|doc)").unwrap(),
                Regex::new(r"^\s*npm\s+(list|ls|outdated|audit)").unwrap(),
                Regex::new(r"^\s*node\s+--version").unwrap(),
                Regex::new(r"^\s*python3?\s+--version").unwrap(),
                Regex::new(r"^\s*rustc\s+--version").unwrap(),
            ],
            blocked_paths: vec![
                "/etc".to_string(),
                "/sys".to_string(),
                "/proc".to_string(),
                "/boot".to_string(),
                "/root".to_string(),
                "~/.ssh".to_string(),
                "~/.gnupg".to_string(),
                "~/.aws".to_string(),
                "~/.config/gcloud".to_string(),
            ],
        }
    }

    /// Check if a command matches a blocked pattern.
    pub fn is_command_blocked(&self, command: &str) -> bool {
        self.blocked_commands.iter().any(|r| r.is_match(command))
    }

    /// Check if a command matches a safe pattern.
    pub fn is_command_safe(&self, command: &str) -> bool {
        self.safe_commands.iter().any(|r| r.is_match(command))
    }

    /// Check if a path is blocked.
    pub fn is_path_blocked(&self, path: &str) -> bool {
        let expanded = if path.starts_with('~') {
            dirs::home_dir()
                .map(|h| path.replacen('~', &h.to_string_lossy(), 1))
                .unwrap_or_else(|| path.to_string())
        } else {
            path.to_string()
        };

        let path_obj = Path::new(&expanded);

        for blocked in &self.blocked_paths {
            let blocked_expanded = if blocked.starts_with('~') {
                dirs::home_dir()
                    .map(|h| blocked.replacen('~', &h.to_string_lossy(), 1))
                    .unwrap_or_else(|| blocked.to_string())
            } else {
                blocked.to_string()
            };

            if path_obj.starts_with(&blocked_expanded) {
                return true;
            }
        }

        // Block path traversal
        if path.contains("..") {
            return true;
        }

        false
    }
}

impl Default for PermissionRules {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_commands() {
        let rules = PermissionRules::new();
        assert!(rules.is_command_blocked("rm -rf /"));
        assert!(rules.is_command_blocked("sudo apt install foo"));
        assert!(!rules.is_command_blocked("ls -la"));
    }

    #[test]
    fn test_safe_commands() {
        let rules = PermissionRules::new();
        assert!(rules.is_command_safe("ls -la"));
        assert!(rules.is_command_safe("git status"));
        assert!(rules.is_command_safe("cargo build"));
        assert!(!rules.is_command_safe("rm file.txt"));
    }

    #[test]
    fn test_blocked_paths() {
        let rules = PermissionRules::new();
        assert!(rules.is_path_blocked("/etc/passwd"));
        assert!(rules.is_path_blocked("/proc/self"));
        assert!(!rules.is_path_blocked("/home/user/project"));
    }
}
