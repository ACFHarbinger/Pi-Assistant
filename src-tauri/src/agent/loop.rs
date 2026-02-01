//! Core agent loop: plan-execute-observe cycle.

use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::{PermissionEngine, PermissionResult};
use crate::tools::{ToolCall, ToolRegistry};
use pi_core::agent_types::{AgentCommand, AgentState, PermissionRequest, StopReason};
use pi_core::task_manager::TaskManager;

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

/// A task for the agent to execute.
#[derive(Clone, Debug)]
pub struct AgentTask {
    pub id: Uuid,
    pub description: String,
    pub max_iterations: u32,
    pub session_id: Uuid,
    pub provider: String,
    pub model_id: Option<String>,
}

/// Handle for a running agent loop.
pub struct AgentLoopHandle {
    pub cancel_token: CancellationToken,
    pub join_handle: tokio::task::JoinHandle<Result<StopReason>>,
    pub cmd_tx: mpsc::Sender<AgentCommand>,
}

/// Plan from the LLM planner.
#[derive(Debug, Deserialize)]
pub struct AgentPlan {
    pub tool_calls: Vec<ToolCall>,
    pub is_complete: bool,
    pub question: Option<String>,
    pub reasoning: String,
}

/// Resources shared by the agent loop.
pub struct AgentResources {
    pub tool_registry: Arc<RwLock<ToolRegistry>>,
    pub memory: Arc<MemoryManager>,
    pub ml_sidecar: Arc<Mutex<SidecarHandle>>,
    pub permission_engine: Arc<Mutex<PermissionEngine>>,
}

/// Spawn the agent loop as a background Tokio task.
pub fn spawn_agent_loop(
    task: AgentTask,
    state_tx: watch::Sender<AgentState>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
    memory: Arc<MemoryManager>,
    ml_sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
) -> AgentLoopHandle {
    let cancel_token = CancellationToken::new();
    let token = cancel_token.clone();
    let (cmd_tx, cmd_rx) = mpsc::channel(32);

    let resources = AgentResources {
        tool_registry,
        memory,
        ml_sidecar,
        permission_engine,
    };

    let join_handle = tokio::spawn(async move {
        match agent_loop(task.clone(), state_tx.clone(), cmd_rx, resources, token).await {
            Ok(reason) => {
                info!(task_id = %task.id, ?reason, "Agent loop finished naturally");
                Ok(reason)
            }
            Err(e) => {
                warn!(task_id = %task.id, error = %e, "Agent loop failed");
                let _ = state_tx.send(AgentState::Stopped {
                    task_id: task.id,
                    reason: pi_core::agent_types::StopReason::Error(e.to_string()),
                });
                Err(e)
            }
        }
    });

    AgentLoopHandle {
        cancel_token,
        join_handle,
        cmd_tx,
    }
}

async fn agent_loop(
    task: AgentTask,
    state_tx: watch::Sender<AgentState>,
    mut cmd_rx: mpsc::Receiver<AgentCommand>,
    resources: AgentResources,
    cancel_token: CancellationToken,
) -> Result<StopReason> {
    info!(task_id = %task.id, "Agent loop started: {}", task.description);

    let mut iteration: u32 = 0;
    let mut task_manager = TaskManager::new();

    loop {
        // ── Check cancellation ───────────────────────────────────────
        if cancel_token.is_cancelled() {
            let _ = state_tx.send(AgentState::Stopped {
                task_id: task.id,
                reason: StopReason::ManualStop,
            });
            return Ok(StopReason::ManualStop);
        }

        // ── Check iteration limit ────────────────────────────────────
        if iteration >= task.max_iterations {
            let _ = state_tx.send(AgentState::Stopped {
                task_id: task.id,
                reason: StopReason::IterationLimit,
            });
            return Ok(StopReason::IterationLimit);
        }

        // ── Broadcast current iteration ──────────────────────────────
        let _ = state_tx.send(AgentState::Running {
            task_id: task.id,
            iteration,
            task_tree: task_manager.get_tree(),
            active_subtask_id: task_manager.get_active_subtask(),
        });

        // ── Check for incoming commands (non-blocking) ───────────────
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                AgentCommand::Stop => {
                    let _ = state_tx.send(AgentState::Stopped {
                        task_id: task.id,
                        reason: StopReason::ManualStop,
                    });
                    return Ok(StopReason::ManualStop);
                }
                AgentCommand::Pause => {
                    let _ = state_tx.send(AgentState::Paused {
                        task_id: task.id,
                        question: None,
                        awaiting_permission: None,
                    });

                    // Block until Resume or Stop
                    loop {
                        let cmd = cmd_rx.recv().await;
                        match cmd {
                            Some(AgentCommand::Resume) => break,
                            Some(AgentCommand::Stop) => {
                                let _ = state_tx.send(AgentState::Stopped {
                                    task_id: task.id,
                                    reason: StopReason::ManualStop,
                                });
                                return Ok(StopReason::ManualStop);
                            }
                            _ => continue,
                        }
                    }

                    let _ = state_tx.send(AgentState::Running {
                        task_id: task.id,
                        iteration,
                        task_tree: task_manager.get_tree(),
                        active_subtask_id: task_manager.get_active_subtask(),
                    });
                }
                AgentCommand::ChannelMessage { text, .. } => {
                    // Store in memory so next iteration sees it
                    let _ = resources
                        .memory
                        .store_message(&task.session_id, "user", &text)
                        .await;
                }
                _ => {}
            }
        }

        // ── 1. Retrieve relevant context from memory ─────────────────
        let context = resources
            .memory
            .retrieve_context(&task.description, &task.session_id, 10)
            .await?;

        // ── 2. Plan next step (LLM call via sidecar) ─────────────────
        let tools = resources.tool_registry.read().await.list_tools();
        let plan = {
            let mut sidecar = resources.ml_sidecar.lock().await;

            // Query device capabilities so the planner knows what hardware is available
            let device_info = sidecar
                .request("device.info", serde_json::json!({}))
                .await
                .ok();

            let response = sidecar
                .request(
                    "inference.plan",
                    serde_json::json!({
                        "task": task.description,
                        "iteration": iteration,
                        "context": context,
                        "task_tree": task_manager.get_tree(),
                        "active_subtask_id": task_manager.get_active_subtask(),
                        "provider": task.provider,
                        "model_id": task.model_id,
                        "tools": tools,
                        "devices": device_info,
                    }),
                )
                .await?;
            serde_json::from_value::<AgentPlan>(response)?
        };

        info!(
            iteration = iteration,
            tool_calls = plan.tool_calls.len(),
            is_complete = plan.is_complete,
            "Plan received"
        );

        // ── 3. Human-in-the-loop: agent asks a question ─────────────
        if let Some(ref question) = plan.question {
            let _ = state_tx.send(AgentState::Paused {
                task_id: task.id,
                question: Some(question.clone()),
                awaiting_permission: None,
            });

            let answer = wait_for_answer(&mut cmd_rx, &cancel_token).await?;
            resources
                .memory
                .store_message(&task.session_id, "user", &answer)
                .await?;

            let _ = state_tx.send(AgentState::Running {
                task_id: task.id,
                iteration,
                task_tree: task_manager.get_tree(),
                active_subtask_id: task_manager.get_active_subtask(),
            });
        }

        // ── 4. Execute each tool call with permission checks ─────────
        for tool_call in &plan.tool_calls {
            let permission = resources.permission_engine.lock().await.check(tool_call)?;

            match permission {
                PermissionResult::Allowed => {}
                PermissionResult::NeedsApproval => {
                    let req = PermissionRequest {
                        id: Uuid::new_v4(),
                        tool_name: tool_call.tool_name.clone(),
                        command: tool_call.display_command(),
                        tier: "medium".into(),
                        description: tool_call.describe(),
                    };

                    let _ = state_tx.send(AgentState::Paused {
                        task_id: task.id,
                        question: None,
                        awaiting_permission: Some(req.clone()),
                    });

                    let (approved, remember) =
                        wait_for_permission(&mut cmd_rx, &cancel_token, req.id).await?;

                    if remember {
                        resources
                            .permission_engine
                            .lock()
                            .await
                            .add_user_override(&tool_call.pattern_key(), approved);
                    }

                    let _ = state_tx.send(AgentState::Running {
                        task_id: task.id,
                        iteration,
                        task_tree: task_manager.get_tree(),
                        active_subtask_id: task_manager.get_active_subtask(),
                    });

                    if !approved {
                        warn!(tool = %tool_call.tool_name, "Permission denied by user");
                        continue;
                    }
                }
                PermissionResult::Denied(reason) => {
                    warn!(tool = %tool_call.tool_name, %reason, "Permission denied by rule");
                    continue;
                }
            }

            // ── Handle internal subtask management ───────────────────
            if tool_call.tool_name == "manage_subtasks" {
                let action = tool_call.parameters.get("action").and_then(|v| v.as_str());
                match action {
                    Some("create") => {
                        if let Some(subtasks) = tool_call
                            .parameters
                            .get("subtasks")
                            .and_then(|v| v.as_array())
                        {
                            for st in subtasks {
                                let title = st
                                    .get("title")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("Untitled")
                                    .to_string();
                                let desc = st
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let parent_id = st
                                    .get("parent_id")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| Uuid::parse_str(s).ok());
                                task_manager.add_subtask(title, desc, parent_id);
                            }
                        }
                    }
                    Some("update") => {
                        let id_str = tool_call
                            .parameters
                            .get("subtask_id")
                            .and_then(|v| v.as_str());
                        let status_str =
                            tool_call.parameters.get("status").and_then(|v| v.as_str());
                        if let (Some(id_s), Some(status_s)) = (id_str, status_str) {
                            if let Ok(id) = Uuid::parse_str(id_s) {
                                let status = match status_s {
                                    "running" => pi_core::agent_types::TaskStatus::Running,
                                    "completed" => pi_core::agent_types::TaskStatus::Completed,
                                    "failed" => pi_core::agent_types::TaskStatus::Failed,
                                    "blocked" => pi_core::agent_types::TaskStatus::Blocked,
                                    _ => pi_core::agent_types::TaskStatus::Pending,
                                };
                                task_manager.update_status(id, status);
                            }
                        }
                    }
                    _ => warn!("Unknown subtask management action: {:?}", action),
                }

                // Persist updated tree
                let _ = resources
                    .memory
                    .store_subtasks(&task.id, &task_manager.get_tree())
                    .await;
                continue;
            }

            // Clone the tool Arc and drop the read lock before executing.
            // This prevents deadlock if a tool (e.g. TrainingTool.deploy)
            // needs to write-lock the registry to register a new tool.
            let tool = resources
                .tool_registry
                .read()
                .await
                .get(&tool_call.tool_name)
                .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", tool_call.tool_name))?
                .clone();
            let result = tool.execute(tool_call.parameters.clone()).await?;
            resources
                .memory
                .store_tool_result(&task.id, tool_call, &result)
                .await?;
        }

        // ── 5. Check completion ──────────────────────────────────────
        if plan.is_complete {
            info!(task_id = %task.id, iterations = iteration, "Task completed");
            let _ = state_tx.send(AgentState::Stopped {
                task_id: task.id,
                reason: StopReason::Completed,
            });
            return Ok(StopReason::Completed);
        }

        iteration += 1;
    }
}

/// Block until the user answers a question.
async fn wait_for_answer(
    cmd_rx: &mut mpsc::Receiver<AgentCommand>,
    cancel_token: &CancellationToken,
) -> Result<String> {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!("Cancelled while waiting for user answer");
            }
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(AgentCommand::AnswerQuestion { response }) => return Ok(response),
                    Some(AgentCommand::ChannelMessage { text, .. }) => return Ok(text),
                    Some(AgentCommand::Stop) => anyhow::bail!("Stopped by user"),
                    None => anyhow::bail!("Command channel closed"),
                    _ => continue,
                }
            }
        }
    }
}

/// Block until the user approves/denies a permission request.
async fn wait_for_permission(
    cmd_rx: &mut mpsc::Receiver<AgentCommand>,
    cancel_token: &CancellationToken,
    _request_id: Uuid,
) -> Result<(bool, bool)> {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!("Cancelled while waiting for permission");
            }
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(AgentCommand::ApprovePermission { approved, remember, .. }) => {
                        return Ok((approved, remember));
                    }
                    Some(AgentCommand::Stop) => anyhow::bail!("Stopped by user"),
                    None => anyhow::bail!("Command channel closed"),
                    _ => continue,
                }
            }
        }
    }
}
