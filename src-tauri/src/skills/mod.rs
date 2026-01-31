//! Skills platform: loading and managing SKILL.md files.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{info, warn};

/// Represents a skill loaded from a SKILL.md file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name from frontmatter or directory name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Semantic version.
    pub version: String,
    /// Path to the skill directory.
    pub path: PathBuf,
    /// Full instruction content (markdown body).
    pub instructions: String,
}

/// Frontmatter parsed from SKILL.md YAML header.
#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    version: Option<String>,
}

/// Manager for loading and accessing skills.
pub struct SkillManager {
    skills: HashMap<String, Skill>,
}

impl SkillManager {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Load skills from multiple directories.
    pub async fn load_from_paths(&mut self, paths: &[PathBuf]) -> Result<()> {
        for path in paths {
            if path.exists() {
                self.load_skills_dir(path).await?;
            }
        }
        Ok(())
    }

    /// Load all skills from a directory.
    async fn load_skills_dir(&mut self, dir: &Path) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let skill_file = path.join("SKILL.md");
                if skill_file.exists() {
                    match self.load_skill(&path, &skill_file).await {
                        Ok(skill) => {
                            info!(skill = %skill.name, path = ?path, "Loaded skill");
                            self.skills.insert(skill.name.clone(), skill);
                        }
                        Err(e) => {
                            warn!(path = ?skill_file, error = %e, "Failed to load skill");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Load a single skill from SKILL.md.
    async fn load_skill(&self, dir: &Path, skill_file: &Path) -> Result<Skill> {
        let content = fs::read_to_string(skill_file).await?;

        // Parse YAML frontmatter
        let (frontmatter, body) = parse_frontmatter(&content)?;

        let name = frontmatter.name.unwrap_or_else(|| {
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        Ok(Skill {
            name,
            description: frontmatter.description.unwrap_or_default(),
            version: frontmatter.version.unwrap_or_else(|| "0.1.0".to_string()),
            path: dir.to_path_buf(),
            instructions: body,
        })
    }

    /// Get a skill by name.
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// List all loaded skills.
    pub fn list(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Get skill instructions for injection into prompts.
    pub fn get_instructions(&self, name: &str) -> Option<&str> {
        self.skills.get(name).map(|s| s.instructions.as_str())
    }
}

/// Parse YAML frontmatter from markdown content.
fn parse_frontmatter(content: &str) -> Result<(SkillFrontmatter, String)> {
    let content = content.trim();

    if let Some(content) = content.strip_prefix("---") {
        if let Some(end) = content.find("---") {
            let yaml = content[..end].trim();
            let body = content[end + 3..].trim().to_string();

            let frontmatter: SkillFrontmatter =
                serde_yaml::from_str(yaml).unwrap_or(SkillFrontmatter {
                    name: None,
                    description: None,
                    version: None,
                });

            return Ok((frontmatter, body));
        }
    }

    // No frontmatter
    Ok((
        SkillFrontmatter {
            name: None,
            description: None,
            version: None,
        },
        content.to_string(),
    ))
}

impl Default for SkillManager {
    fn default() -> Self {
        Self::new()
    }
}
