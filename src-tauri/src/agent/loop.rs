//! Core agent loop: plan-execute-observe cycle.

use crate::ipc::SidecarHandle;
use crate::memory::MemoryManager;
use crate::safety::{PermissionEngine, PermissionResult};
use crate::tools::{ToolCall, ToolRegistry};
use pi_core::agent_types::{AgentCommand, AgentState, PermissionRequest, StopReason};

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

/// A task for the agent to execute.
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

/// Spawn the agent loop as a background Tokio task.
pub fn spawn_agent_loop(
    task: AgentTask,
    state_tx: watch::Sender<AgentState>,
    tool_registry: Arc<ToolRegistry>,
    memory: Arc<MemoryManager>,
    sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
) -> AgentLoopHandle {
    let cancel_token = CancellationToken::new();
    let token = cancel_token.clone();
    let (cmd_tx, cmd_rx) = mpsc::channel(32);

    let join_handle = tokio::spawn(async move {
        agent_loop(
            task,
            state_tx,
            cmd_rx,
            tool_registry,
            memory,
            sidecar,
            permission_engine,
            token,
        )
        .await
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
    tool_registry: Arc<ToolRegistry>,
    memory: Arc<MemoryManager>,
    sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
    cancel_token: CancellationToken,
) -> Result<StopReason> {
    info!(task_id = %task.id, "Agent loop started: {}", task.description);

    let mut iteration: u32 = 0;

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
                    });
                }
                AgentCommand::ChannelMessage { text, .. } => {
                    // Store in memory so next iteration sees it
                    let _ = memory.store_message(&task.session_id, "user", &text).await;
                }
                _ => {}
            }
        }

        // ── 1. Retrieve relevant context from memory ─────────────────
        let context = memory
            .retrieve_context(&task.description, &task.session_id, 10)
            .await?;

        // ── 2. Plan next step (LLM call via sidecar) ─────────────────
        let plan = {
            let mut sidecar = sidecar.lock().await;
            let response = sidecar
                .request(
                    "inference.plan",
                    serde_json::json!({
                        "task": task.description,
                        "iteration": iteration,
                        "context": context,
                        "provider": task.provider,
                        "model_id": task.model_id,
                        "tools": tool_registry.list_tools(),
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
            memory
                .store_message(&task.session_id, "user", &answer)
                .await?;

            let _ = state_tx.send(AgentState::Running {
                task_id: task.id,
                iteration,
            });
        }

        // ── 4. Execute each tool call with permission checks ─────────
        for tool_call in &plan.tool_calls {
            let permission = permission_engine.lock().await.check(tool_call)?;

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
                        permission_engine
                            .lock()
                            .await
                            .add_user_override(&tool_call.pattern_key(), approved);
                    }

                    let _ = state_tx.send(AgentState::Running {
                        task_id: task.id,
                        iteration,
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

            let result = tool_registry.execute(tool_call).await?;
            memory
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
                    _ => continue,
                }
            }
        }
    }
}
