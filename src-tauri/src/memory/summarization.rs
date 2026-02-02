//! Episode summarization for completed tasks.

/// Generate a narrative summary of a completed task.
///
/// This creates a human-readable summary based on:
/// - The original task description
/// - Tool calls that were executed
/// - The final outcome (success/failure)
///
/// Currently uses a template-based approach. Future versions may
/// use an LLM for more nuanced summaries.
pub fn generate_task_summary(
    task_description: &str,
    tool_calls: &[(String, bool)], // (tool_name, success)
    outcome: TaskOutcome,
) -> String {
    let tool_count = tool_calls.len();
    let success_count = tool_calls.iter().filter(|(_, s)| *s).count();
    let failure_count = tool_count - success_count;

    let tool_summary = if tool_count == 0 {
        "No tools were used.".to_string()
    } else {
        let tool_names: Vec<&str> = tool_calls
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let tools_str = tool_names.join(", ");
        format!(
            "Used {} tool calls ({} succeeded, {} failed) involving: {}.",
            tool_count, success_count, failure_count, tools_str
        )
    };

    let outcome_str = match outcome {
        TaskOutcome::Completed => "Task completed successfully.",
        TaskOutcome::Failed(ref reason) => &format!("Task failed: {}", reason),
        TaskOutcome::Stopped(ref reason) => &format!("Task stopped: {}", reason),
    };

    format!(
        "**Task**: {}\n\n**Actions**: {}\n\n**Outcome**: {}",
        task_description, tool_summary, outcome_str
    )
}

/// The outcome of a task execution.
#[derive(Debug, Clone)]
pub enum TaskOutcome {
    Completed,
    Failed(String),
    Stopped(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_summary_success() {
        let summary = generate_task_summary(
            "Build a simple web page",
            &[
                ("code".to_string(), true),
                ("shell".to_string(), true),
                ("browser".to_string(), true),
            ],
            TaskOutcome::Completed,
        );

        assert!(summary.contains("Build a simple web page"));
        assert!(summary.contains("3 tool calls"));
        assert!(summary.contains("3 succeeded"));
        assert!(summary.contains("completed successfully"));
    }

    #[test]
    fn test_generate_summary_no_tools() {
        let summary = generate_task_summary("Answer a question", &[], TaskOutcome::Completed);

        assert!(summary.contains("No tools were used"));
    }

    #[test]
    fn test_generate_summary_failure() {
        let summary = generate_task_summary(
            "Deploy to production",
            &[("shell".to_string(), false)],
            TaskOutcome::Failed("Permission denied".to_string()),
        );

        assert!(summary.contains("1 failed"));
        assert!(summary.contains("Permission denied"));
    }
}
