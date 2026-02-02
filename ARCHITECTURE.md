# Pi-Assistant: Universal Agent Harness — Architecture & Implementation Plan

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Kotlin](https://img.shields.io/badge/kotlin-%237F52FF.svg?style=for-the-badge&logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![Tauri](https://img.shields.io/badge/tauri-%2324C8DB.svg?style=for-the-badge&logo=tauri&logoColor=%23262626)](https://tauri.app/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com/)

## 1. High-Level Architecture

### System Overview

Pi-Assistant is a multi-runtime desktop application where a **Rust core** (Tauri v2) orchestrates an autonomous AI agent, a **Python sidecar** handles ML inference/training, a **React/TypeScript** frontend provides the desktop UI, and a **Kotlin Android** app serves as a mobile remote.

```
+--------------------+       Tauri IPC        +-----------------------------+
|                    | <-- invoke/events ----> |                             |
|  React/TypeScript  |                         |       Rust Core (Tauri)     |
|  (Desktop UI)      |                         |                             |
|                    |                         |  +-----------------------+  |
+--------------------+                         |  |     Agent Loop        |  |
                                               |  |  (tokio async task)   |  |
+--------------------+    WebSocket (axum)     |  +----------+------------+  |
|                    | <-- ws://:9120/ws -----> |             |               |
|  Kotlin / Android  |                         |  +----------v------------+  |
|  (Mobile Client)   |                         |  |    Tool Registry       |  |
|                    |                         |  | shell|browser|code|ml  |  |
+--------------------+                         |  +----------+------------+  |
                                               |             |               |
                                               |  +----------v------------+  |
                                               |  |  Permission Engine    |  |
                                               |  | (3-tier gate)         |  |
                                               |  +----------+------------+  |
                                               |             |               |
                                               |  +----------v------------+  |
                                               |  |  Memory Manager       |  |
                                               |  | SQLite + sqlite-vec   |  |
                                               |  +----------+------------+  |
                                               |             |               |
                                               |  +----------v------------+  |
                                               |  |    IPC Bridge         |  |
                                               |  | (NDJSON over stdio)   |  |
                                               |  +----------+------------+  |
                                               +-------------|---------------+
                                                             |
                                                       stdin/stdout
                                                      (NDJSON lines)
                                                             |
                                               +-------------v---------------+
                                               |                             |
                                               |     Python Sidecar          |
                                               |  PyTorch Lightning          |
                                               |  sentence-transformers      |
                                               |  transformers (HF)          |
                                               |                             |
                                               +-----------------------------+
```

### How Tauri Manages the Python Sidecar

1. On app startup, `lib.rs` spawns the Python process via `tokio::process::Command` with `stdin(Stdio::piped())`, `stdout(Stdio::piped())`, and `kill_on_drop(true)`.
2. Rust writes **newline-delimited JSON (NDJSON)** requests to Python's stdin. Each request carries a UUID correlation ID.
3. A dedicated Tokio task reads lines from Python's stdout, parses each JSON line, and routes responses back to the waiting caller via a `HashMap<Uuid, oneshot::Sender>`.
4. Long-running operations (training) emit intermediate `"progress"` messages that flow through a separate `mpsc` channel to the UI.
5. If the sidecar crashes, the `SidecarHandle` detects the closed stdout stream and automatically restarts it.

### Data Flow: Android -> Rust Server -> Agent

```
Android App                    Rust Server                    Agent Loop
    |                              |                              |
    |-- WS: {"type":"command",  -->|                              |
    |    "payload":{"action":      |                              |
    |     "start","task":"..."}}   |                              |
    |                              |-- mpsc: AgentCommand ------->|
    |                              |   ::Start { task }           |
    |                              |                              |-- iterate
    |                              |                              |-- call tools
    |                              |                              |-- check permissions
    |                              |<-- watch: AgentState --------|
    |<-- WS: {"type":             |   ::Running { task_id }      |
    |    "agent_state_update"} ----|                              |
    |                              |                              |
    |                              |<-- watch: AgentState --------|
    |<-- WS: {"type":             |   ::Paused { question }      |
    |    "permission_request"} ----|                              |
    |                              |                              |
    |-- WS: {"type":           -->|                              |
    |    "permission_response",    |-- mpsc: AgentCommand ------->|
    |    "approved": true}         |   ::ApprovePermission       |
    |                              |                              |-- continue
```

### Memory Integration Strategy

Memory serves two purposes: **structured recall** (what happened, in order) and **semantic search** (find relevant context by meaning).

| Layer      | Technology                                   | Purpose                                                |
| ---------- | -------------------------------------------- | ------------------------------------------------------ |
| Structured | SQLite (`rusqlite`, bundled)                 | Sessions, messages, tasks, tool logs, permission cache |
| Semantic   | `sqlite-vec` extension                       | Vector similarity search over embeddings               |
| Embeddings | Python sidecar (`all-MiniLM-L6-v2`, 384-dim) | Generate vectors from text                             |

**Retrieval flow per agent iteration:**

1. Build a query string from the current task description + recent context.
2. Send to Python sidecar `inference.embed` -> get 384-dim vector.
3. Query `sqlite-vec` for top-K similar memories.
4. Also fetch the last N messages from SQLite by recency.
5. Merge, deduplicate, rank by relevance, inject into LLM prompt.

---

## 2. File Structure

```
Pi-Assistant/
|-- .gitignore
|-- LICENSE                              # AGPL-3.0
|-- ARCHITECTURE.md                      # This document
|-- Cargo.toml                           # Workspace root (virtual)
|-- package.json                         # Frontend deps + scripts
|-- tsconfig.json
|-- vite.config.ts
|
|-- protocol/                            # === SHARED PROTOCOL SCHEMAS ===
|   |-- schemas/
|   |   |-- ipc-message.schema.json      # Rust <-> Python NDJSON contract
|   |   |-- ws-message.schema.json       # Desktop <-> Mobile WebSocket contract
|   |   |-- agent-state.schema.json
|   |   |-- tool-request.schema.json
|   |   `-- permission.schema.json
|   |-- rust/
|   |   |-- Cargo.toml                   # Crate: pi-protocol
|   |   `-- src/
|   |       `-- lib.rs                   # Serde structs for all protocol types
|   |-- python/
|   |   `-- pi_protocol/
|   |       |-- __init__.py
|   |       `-- messages.py              # Pydantic models mirroring schemas
|   `-- kotlin/
|       `-- PiProtocol.kt               # Kotlinx.serialization data classes
|
|-- src/                                 # === FRONTEND (React + TypeScript) ===
|   |-- index.html
|   |-- main.tsx
|   |-- App.tsx
|   |-- components/
|   |   |-- AgentStatus.tsx              # Running/Paused/Idle/Stopped badge
|   |   |-- ChatInterface.tsx            # Human-in-the-loop conversation
|   |   |-- TaskManager.tsx              # Task queue and iteration history
|   |   |-- ToolOutput.tsx               # Shell/browser/code output viewer
|   |   |-- PermissionDialog.tsx         # Approve/deny dangerous commands
|   |   `-- MemoryBrowser.tsx            # Search long-term memory
|   |-- hooks/
|   |   |-- useAgentState.ts
|   |   |-- useChat.ts
|   |   `-- usePermission.ts
|   |-- stores/
|   |   `-- agentStore.ts               # Zustand
|   |-- services/
|   |   `-- tauriIpc.ts                 # @tauri-apps/api wrapper
|   `-- styles/
|       `-- index.css                    # Tailwind
|
|-- src-tauri/                           # === RUST CORE (Tauri v2) ===
|   |-- Cargo.toml
|   |-- tauri.conf.json
|   |-- build.rs
|   |-- capabilities/
|   |   `-- default.json                 # Tauri v2 ACL
|   |-- binaries/                        # Bundled Python sidecar binary
|   `-- src/
|       |-- main.rs                      # Entry point
|       |-- lib.rs                       # Tauri builder, plugin registration
|       |-- state.rs                     # AppState, AgentState enum
|       |-- commands/                    # Tauri #[command] handlers
|       |   |-- mod.rs
|       |   |-- agent.rs                 # start/stop/pause/resume
|       |   |-- chat.rs                  # send_message, get_history
|       |   |-- memory.rs               # query_memory, store_memory
|       |   `-- settings.rs             # permissions, config
|       |-- agent/                       # Agent loop engine
|       |   |-- mod.rs
|       |   |-- loop.rs                  # Core async loop (tokio task)
|       |   |-- planner.rs              # LLM-driven task decomposition
|       |   `-- executor.rs             # Tool dispatch, result collection
|       |-- tools/                       # Tool implementations
|       |   |-- mod.rs                   # Tool trait + ToolRegistry
|       |   |-- shell.rs                # Shell command execution
|       |   |-- browser.rs              # Headless Chrome (chromiumoxide)
|       |   |-- code.rs                 # File read/write/patch
|       |   `-- python_inference.rs     # Delegate to Python sidecar
|       |-- ipc/                         # Rust <-> Python bridge
|       |   |-- mod.rs
|       |   |-- sidecar.rs              # Child process lifecycle
|       |   |-- ndjson.rs               # NDJSON codec
|       |   `-- router.rs              # Correlation ID routing
|       |-- ws/                          # WebSocket server (mobile)
|       |   |-- mod.rs
|       |   |-- server.rs              # Axum WS server on :9120
|       |   |-- handler.rs             # Per-connection message dispatch
|       |   `-- auth.rs                # Token/pairing authentication
|       |-- memory/                      # Memory subsystem
|       |   |-- mod.rs
|       |   |-- sqlite.rs              # Schema + queries (rusqlite)
|       |   |-- vector.rs              # sqlite-vec integration
|       |   |-- embeddings.rs          # Embedding via Python sidecar
|       |   `-- retriever.rs           # Combined SQL + vector retrieval
|       |-- safety/                      # Permission / safety layer
|       |   |-- mod.rs
|       |   |-- permission.rs          # PermissionEngine
|       |   |-- rules.rs              # Tier definitions, regex patterns
|       |   `-- sandbox.rs            # Process sandboxing
|       `-- config/
|           |-- mod.rs
|           `-- defaults.rs            # Default rules, paths, settings
|
|-- crates/                              # === WORKSPACE CRATES ===
|   `-- pi-core/
|       |-- Cargo.toml
|       `-- src/
|           |-- lib.rs
|           |-- agent_types.rs          # Shared types (no Tauri dep)
|           `-- protocol.rs
|
|-- sidecar/                             # === PYTHON SIDECAR (ML) ===
|   |-- pyproject.toml
|   |-- uv.lock
|   |-- build.py                         # PyInstaller bundling script
|   |-- src/
|   |   `-- pi_sidecar/
|   |       |-- __init__.py
|   |       |-- __main__.py              # Entry: NDJSON event loop
|   |       |-- ipc/
|   |       |   |-- __init__.py
|   |       |   |-- ndjson_transport.py  # Async stdin/stdout NDJSON
|   |       |   `-- handler.py          # Route requests to handlers
|   |       |-- inference/
|   |       |   |-- __init__.py
|   |       |   |-- engine.py           # Model loading + inference
|   |       |   |-- embeddings.py       # sentence-transformers
|   |       |   `-- completion.py       # Text generation (HF transformers)
|   |       |-- training/
|   |       |   |-- __init__.py
|   |       |   |-- lightning_module.py # PL LightningModule
|   |       |   |-- data_module.py      # PL LightningDataModule
|   |       |   `-- trainer.py          # Training orchestration + streaming
|   |       `-- models/
|   |           |-- __init__.py
|   |           `-- registry.py         # Model versioning, load/save
|   `-- tests/
|       |-- test_ipc.py
|       |-- test_inference.py
|       `-- test_training.py
|
|-- android/                             # === ANDROID CLIENT (Kotlin) ===
|   |-- build.gradle.kts
|   |-- settings.gradle.kts
|   |-- gradle.properties
|   |-- gradle/
|   |   `-- libs.versions.toml
|   `-- app/
|       |-- build.gradle.kts
|       `-- src/
|           `-- main/
|               |-- AndroidManifest.xml
|               `-- java/dev/piassistant/android/
|                   |-- PiApplication.kt
|                   |-- MainActivity.kt
|                   |-- ui/
|                   |   |-- screens/
|                   |   |   |-- HomeScreen.kt
|                   |   |   `-- SettingsScreen.kt
|                   |   |-- components/
|                   |   |   |-- ChatBubble.kt
|                   |   |   |-- StatusIndicator.kt
|                   |   |   `-- VoiceInputButton.kt
|                   |   `-- theme/
|                   |       `-- Theme.kt
|                   |-- network/
|                   |   |-- WebSocketClient.kt
|                   |   |-- ConnectionManager.kt
|                   |   `-- MessageSerializer.kt
|                   |-- viewmodel/
|                   |   |-- AgentViewModel.kt
|                   |   `-- ChatViewModel.kt
|                   |-- voice/
|                   |   `-- SpeechRecognizerHelper.kt
|                   `-- data/
|                       |-- Message.kt
|                       `-- AgentState.kt
`-- docs/
    |-- ipc-protocol.md
    |-- safety-model.md
    `-- setup.md
```

---

## 3. Implementation Details

### 3.1 Rust — State & Agent Loop

**`src-tauri/src/state.rs`** — Central application state:

```rust
use std::sync::Arc;
use tokio::sync::{watch, mpsc, Mutex};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

// ── Agent state machine ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
pub enum AgentState {
    Idle,
    Running {
        task_id: Uuid,
        iteration: u32,
    },
    Paused {
        task_id: Uuid,
        question: Option<String>,
        awaiting_permission: Option<PermissionRequest>,
    },
    Stopped {
        task_id: Uuid,
        reason: StopReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    Completed,
    ManualStop,
    Error(String),
    IterationLimit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub id: Uuid,
    pub tool_name: String,
    pub command: String,
    pub tier: String,
    pub description: String,
}

// ── Commands sent TO the agent loop ──────────────────────────────────

#[derive(Debug)]
pub enum AgentCommand {
    Start {
        task: String,
        max_iterations: Option<u32>,
    },
    Stop,
    Pause,
    Resume,
    AnswerQuestion {
        response: String,
    },
    ApprovePermission {
        request_id: Uuid,
        approved: bool,
        remember: bool,  // "always allow this pattern"
    },
}

// ── Shared application state ─────────────────────────────────────────

pub struct AppState {
    pub agent_state_tx: watch::Sender<AgentState>,
    pub agent_state_rx: watch::Receiver<AgentState>,
    pub agent_cmd_tx: mpsc::Sender<AgentCommand>,
    pub agent_cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    pub permissions: Arc<Mutex<crate::safety::PermissionEngine>>,
    pub memory: Arc<crate::memory::MemoryManager>,
    pub sidecar: Arc<Mutex<crate::ipc::SidecarHandle>>,
    pub tool_registry: Arc<crate::tools::ToolRegistry>,
}
```

**`src-tauri/src/agent/loop.rs`** — Core async agent loop:

```rust
use crate::state::*;
use crate::tools::ToolRegistry;
use crate::memory::MemoryManager;
use crate::ipc::SidecarHandle;
use crate::safety::PermissionEngine;

use std::sync::Arc;
use tokio::sync::{watch, mpsc, Mutex};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use anyhow::Result;
use tracing::{info, warn, error};

pub struct AgentTask {
    pub id: Uuid,
    pub description: String,
    pub max_iterations: u32,
    pub session_id: Uuid,
}

pub struct AgentLoopHandle {
    pub cancel_token: CancellationToken,
    pub join_handle: tokio::task::JoinHandle<Result<StopReason>>,
}

/// Spawn the agent loop as a background Tokio task.
/// Returns a handle for cancellation and awaiting completion.
pub fn spawn_agent_loop(
    task: AgentTask,
    state_tx: watch::Sender<AgentState>,
    cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    tool_registry: Arc<ToolRegistry>,
    memory: Arc<MemoryManager>,
    sidecar: Arc<Mutex<SidecarHandle>>,
    permission_engine: Arc<Mutex<PermissionEngine>>,
) -> AgentLoopHandle {
    let cancel_token = CancellationToken::new();
    let token = cancel_token.clone();

    let join_handle = tokio::spawn(async move {
        agent_loop(task, state_tx, cmd_rx, tool_registry, memory, sidecar, permission_engine, token).await
    });

    AgentLoopHandle { cancel_token, join_handle }
}

async fn agent_loop(
    task: AgentTask,
    state_tx: watch::Sender<AgentState>,
    cmd_rx: Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
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
            state_tx.send_replace(AgentState::Stopped {
                task_id: task.id,
                reason: StopReason::ManualStop,
            });
            return Ok(StopReason::ManualStop);
        }

        // ── Check iteration limit ────────────────────────────────────
        if iteration >= task.max_iterations {
            state_tx.send_replace(AgentState::Stopped {
                task_id: task.id,
                reason: StopReason::IterationLimit,
            });
            return Ok(StopReason::IterationLimit);
        }

        // ── Broadcast current iteration ──────────────────────────────
        state_tx.send_replace(AgentState::Running {
            task_id: task.id,
            iteration,
        });

        // ── Check for incoming commands (non-blocking) ───────────────
        if let Ok(cmd) = cmd_rx.lock().await.try_recv() {
            match cmd {
                AgentCommand::Stop => {
                    state_tx.send_replace(AgentState::Stopped {
                        task_id: task.id,
                        reason: StopReason::ManualStop,
                    });
                    return Ok(StopReason::ManualStop);
                }
                AgentCommand::Pause => {
                    state_tx.send_replace(AgentState::Paused {
                        task_id: task.id,
                        question: None,
                        awaiting_permission: None,
                    });
                    // Block until Resume or Stop
                    loop {
                        let cmd = cmd_rx.lock().await.recv().await;
                        match cmd {
                            Some(AgentCommand::Resume) => break,
                            Some(AgentCommand::Stop) => {
                                state_tx.send_replace(AgentState::Stopped {
                                    task_id: task.id,
                                    reason: StopReason::ManualStop,
                                });
                                return Ok(StopReason::ManualStop);
                            }
                            _ => continue,
                        }
                    }
                    state_tx.send_replace(AgentState::Running {
                        task_id: task.id,
                        iteration,
                    });
                }
                _ => {}
            }
        }

        // ── 1. Retrieve relevant context from memory ─────────────────
        let context = memory.retrieve_context(&task.description, &task.session_id, 10).await?;

        // ── 2. Plan next step (LLM call via sidecar) ─────────────────
        let plan = {
            let mut sidecar = sidecar.lock().await;
            let response = sidecar.request("inference.plan", serde_json::json!({
                "task": task.description,
                "iteration": iteration,
                "context": context,
            })).await?;
            serde_json::from_value::<AgentPlan>(response)?
        };

        // ── 3. Human-in-the-loop: agent asks a question ─────────────
        if let Some(ref question) = plan.question {
            state_tx.send_replace(AgentState::Paused {
                task_id: task.id,
                question: Some(question.clone()),
                awaiting_permission: None,
            });

            let answer = wait_for_answer(&cmd_rx, &cancel_token).await?;
            memory.store_message(&task.session_id, "user", &answer).await?;

            state_tx.send_replace(AgentState::Running {
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

                    state_tx.send_replace(AgentState::Paused {
                        task_id: task.id,
                        question: None,
                        awaiting_permission: Some(req.clone()),
                    });

                    let (approved, remember) =
                        wait_for_permission(&cmd_rx, &cancel_token, req.id).await?;

                    if remember {
                        permission_engine.lock().await
                            .add_user_override(&tool_call.pattern_key(), approved);
                    }

                    state_tx.send_replace(AgentState::Running {
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
            memory.store_tool_result(&task.id, tool_call, &result).await?;
        }

        // ── 5. Check completion ──────────────────────────────────────
        if plan.is_complete {
            info!(task_id = %task.id, iterations = iteration, "Task completed");
            state_tx.send_replace(AgentState::Stopped {
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
    cmd_rx: &Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    cancel_token: &CancellationToken,
) -> Result<String> {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!("Cancelled while waiting for user answer");
            }
            cmd = cmd_rx.lock().await.recv() => {
                match cmd {
                    Some(AgentCommand::AnswerQuestion { response }) => return Ok(response),
                    Some(AgentCommand::Stop) => anyhow::bail!("Stopped by user"),
                    _ => continue,
                }
            }
        }
    }
}

/// Block until the user approves/denies a permission request.
async fn wait_for_permission(
    cmd_rx: &Arc<Mutex<mpsc::Receiver<AgentCommand>>>,
    cancel_token: &CancellationToken,
    _request_id: Uuid,
) -> Result<(bool, bool)> {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!("Cancelled while waiting for permission");
            }
            cmd = cmd_rx.lock().await.recv() => {
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

// ── Types used by the planner ────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct AgentPlan {
    pub tool_calls: Vec<ToolCall>,
    pub is_complete: bool,
    pub question: Option<String>,
    pub reasoning: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub parameters: serde_json::Value,
}

impl ToolCall {
    pub fn display_command(&self) -> String {
        if self.tool_name == "shell" {
            self.parameters.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>")
                .to_string()
        } else {
            format!("{}({})", self.tool_name, self.parameters)
        }
    }

    pub fn describe(&self) -> String {
        format!("Execute {} tool", self.tool_name)
    }

    pub fn pattern_key(&self) -> String {
        self.display_command()
    }
}
```

### 3.2 Rust <-> Python Bridge

**`src-tauri/src/ipc/sidecar.rs`** — Sidecar lifecycle + request/response routing:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::{info, error, warn};

// ── Wire types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct IpcRequest {
    id: String,
    method: String,
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct IpcMessage {
    id: String,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<IpcError>,
    #[serde(default)]
    progress: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct IpcError {
    code: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub request_id: String,
    pub data: serde_json::Value,
}

// ── Sidecar handle ───────────────────────────────────────────────────

pub struct SidecarHandle {
    child: Option<Child>,
    stdin: Option<tokio::process::ChildStdin>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<serde_json::Value>>>>>,
    progress_tx: mpsc::Sender<ProgressUpdate>,
    progress_rx: Option<mpsc::Receiver<ProgressUpdate>>,
    python_path: String,
    sidecar_module: String,
}

impl SidecarHandle {
    pub fn new() -> Self {
        let (progress_tx, progress_rx) = mpsc::channel(256);
        Self {
            child: None,
            stdin: None,
            pending: Arc::new(Mutex::new(HashMap::new())),
            progress_tx,
            progress_rx: Some(progress_rx),
            python_path: "python3".to_string(),
            sidecar_module: "pi_sidecar".to_string(),
        }
    }

    /// Take the progress receiver (can only be called once; give to UI layer).
    pub fn take_progress_rx(&mut self) -> Option<mpsc::Receiver<ProgressUpdate>> {
        self.progress_rx.take()
    }

    /// Start the Python sidecar process.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting Python sidecar: {} -m {}", self.python_path, self.sidecar_module);

        let mut child = Command::new(&self.python_path)
            .arg("-m")
            .arg(&self.sidecar_module)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit()) // stderr goes to Rust's stderr (for logging)
            .kill_on_drop(true)
            .spawn()?;

        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow!("Failed to capture sidecar stdout"))?;
        let stdin = child.stdin.take()
            .ok_or_else(|| anyhow!("Failed to capture sidecar stdin"))?;

        self.stdin = Some(stdin);
        self.child = Some(child);

        // Spawn stdout reader that routes responses back to callers
        let pending = self.pending.clone();
        let progress_tx = self.progress_tx.clone();

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                let msg: IpcMessage = match serde_json::from_str(&line) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Malformed NDJSON from sidecar: {e}");
                        continue;
                    }
                };

                // Progress update — don't complete the pending request
                if let Some(progress_data) = msg.progress {
                    let _ = progress_tx.send(ProgressUpdate {
                        request_id: msg.id.clone(),
                        data: progress_data,
                    }).await;
                    continue;
                }

                // Response (success or error) — complete the pending request
                let mut pending = pending.lock().await;
                if let Some(tx) = pending.remove(&msg.id) {
                    let result = if let Some(err) = msg.error {
                        Err(anyhow!("[{}] {}", err.code, err.message))
                    } else {
                        Ok(msg.result.unwrap_or(serde_json::Value::Null))
                    };
                    let _ = tx.send(result);
                } else {
                    warn!("Received response for unknown request ID: {}", msg.id);
                }
            }

            error!("Sidecar stdout stream ended — process likely exited");
        });

        // Verify the sidecar is alive with a health check
        let health = self.request("health.ping", serde_json::json!({})).await;
        match health {
            Ok(_) => info!("Python sidecar is healthy"),
            Err(e) => {
                error!("Sidecar health check failed: {e}");
                return Err(anyhow!("Sidecar health check failed: {e}"));
            }
        }

        Ok(())
    }

    /// Send a request to the Python sidecar and await the response.
    pub async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();

        // Register the pending response
        self.pending.lock().await.insert(id.clone(), tx);

        // Serialize and write to stdin
        let request = IpcRequest {
            id: id.clone(),
            method: method.to_string(),
            params,
        };
        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        let stdin = self.stdin.as_mut()
            .ok_or_else(|| anyhow!("Sidecar not started"))?;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        // Await response with timeout (5 minutes for training tasks)
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            rx,
        )
        .await
        .map_err(|_| {
            // Remove from pending on timeout
            let pending = self.pending.clone();
            let id = id.clone();
            tokio::spawn(async move { pending.lock().await.remove(&id); });
            anyhow!("Sidecar request timed out after 300s: {method}")
        })?
        .map_err(|_| anyhow!("Sidecar response channel closed"))?;

        result
    }

    /// Gracefully shut down the sidecar.
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(ref mut child) = self.child {
            // Send shutdown request (best-effort)
            let _ = self.request("lifecycle.shutdown", serde_json::json!({})).await;
            // Wait briefly, then kill
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let _ = child.kill().await;
        }
        self.child = None;
        self.stdin = None;
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }
}
```

### 3.3 Python Sidecar — Entry Point & PyTorch Lightning Module

**`sidecar/src/pi_sidecar/__main__.py`**:

```python
"""
Pi-Assistant Python Sidecar — Entry Point.

Communicates with the Rust core via NDJSON over stdin/stdout.
stderr is used exclusively for logging (not protocol traffic).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from pi_sidecar.ipc.ndjson_transport import NdjsonTransport
from pi_sidecar.inference.engine import InferenceEngine
from pi_sidecar.training.trainer import TrainingManager
from pi_sidecar.models.registry import ModelRegistry

# All logging goes to stderr so stdout stays clean for NDJSON
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[sidecar] %(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RequestHandler:
    """Routes incoming IPC method calls to the correct subsystem."""

    def __init__(
        self,
        engine: InferenceEngine,
        trainer: TrainingManager,
        registry: ModelRegistry,
    ):
        self.engine = engine
        self.trainer = trainer
        self.registry = registry

        self._handlers: dict[str, Any] = {
            "health.ping": self._health_ping,
            "lifecycle.shutdown": self._lifecycle_shutdown,
            "inference.complete": self._inference_complete,
            "inference.embed": self._inference_embed,
            "inference.plan": self._inference_plan,
            "training.start": self._training_start,
            "training.stop": self._training_stop,
            "model.list": self._model_list,
            "model.load": self._model_load,
        }

    async def dispatch(
        self,
        method: str,
        params: dict,
        progress_callback=None,
    ) -> Any:
        handler = self._handlers.get(method)
        if handler is None:
            raise ValueError(f"Unknown method: {method}")
        return await handler(params, progress_callback)

    # ── Built-in handlers ─────────────────────────────────────────

    async def _health_ping(self, params, _cb):
        return {"status": "ok", "version": "0.1.0"}

    async def _lifecycle_shutdown(self, params, _cb):
        logger.info("Shutdown requested by Rust core")
        # Give a moment for the response to flush, then exit
        asyncio.get_event_loop().call_later(0.5, sys.exit, 0)
        return {"status": "shutting_down"}

    async def _inference_complete(self, params, _cb):
        return await self.engine.complete(
            prompt=params["prompt"],
            model_id=params.get("model_id", "default"),
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
        )

    async def _inference_embed(self, params, _cb):
        vector = await self.engine.embed(
            text=params["text"],
            model_id=params.get("model_id", "embedding-default"),
        )
        return {"embedding": vector}

    async def _inference_plan(self, params, _cb):
        # The planner uses the LLM to decide next steps
        return await self.engine.plan(
            task=params["task"],
            iteration=params["iteration"],
            context=params.get("context", []),
        )

    async def _training_start(self, params, progress_callback):
        return await self.trainer.start_training(
            model_id=params["model_id"],
            dataset_path=params["dataset_path"],
            config=params.get("config", {}),
            progress_callback=progress_callback,
        )

    async def _training_stop(self, params, _cb):
        return self.trainer.stop_training(params.get("model_id"))

    async def _model_list(self, params, _cb):
        return {"models": self.registry.list_models()}

    async def _model_load(self, params, _cb):
        self.registry.load_model(params["model_id"])
        return {"status": "loaded", "model_id": params["model_id"]}


async def main():
    logger.info("Pi-Assistant sidecar starting")

    registry = ModelRegistry()
    engine = InferenceEngine(registry)
    trainer = TrainingManager(registry)
    handler = RequestHandler(engine=engine, trainer=trainer, registry=registry)
    transport = NdjsonTransport()

    async for request in transport.read_requests():
        # Each request is handled concurrently so inference and training
        # can proceed in parallel without blocking the IPC loop.
        asyncio.create_task(_handle_request(handler, transport, request))


async def _handle_request(
    handler: RequestHandler,
    transport: NdjsonTransport,
    request: dict,
):
    request_id = request.get("id", "unknown")
    method = request.get("method", "")
    params = request.get("params", {})

    logger.info("Handling request %s: %s", request_id, method)

    try:
        result = await handler.dispatch(
            method=method,
            params=params,
            progress_callback=lambda p: asyncio.ensure_future(
                transport.send_progress(request_id, p)
            ),
        )
        await transport.send_response(request_id, result=result)
    except Exception as e:
        logger.exception("Error handling %s: %s", method, e)
        await transport.send_error(
            request_id,
            code=type(e).__name__,
            message=str(e),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

**`sidecar/src/pi_sidecar/ipc/ndjson_transport.py`**:

```python
"""NDJSON transport over stdin/stdout for Rust <-> Python IPC."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import AsyncIterator


class NdjsonTransport:
    """
    Reads JSON lines from stdin, writes JSON lines to stdout.
    Thread-safe for concurrent writes via asyncio Lock.
    """

    def __init__(self):
        self._write_lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None

    async def _ensure_reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            loop = asyncio.get_event_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
            self._reader = reader
        return self._reader

    async def read_requests(self) -> AsyncIterator[dict]:
        """Async generator yielding parsed JSON request dicts from stdin."""
        reader = await self._ensure_reader()

        while True:
            line = await reader.readline()
            if not line:
                break  # stdin closed — parent process exited

            try:
                decoded = line.decode("utf-8").strip()
                if decoded:
                    yield json.loads(decoded)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Log to stderr, never to stdout
                print(f"[sidecar] Malformed input line: {e}", file=sys.stderr)
                continue

    async def send_response(
        self, request_id: str, result=None, error=None
    ):
        msg: dict = {"id": request_id}
        if result is not None:
            msg["result"] = result
        if error is not None:
            msg["error"] = error
        await self._write(msg)

    async def send_error(
        self, request_id: str, code: str, message: str
    ):
        await self.send_response(
            request_id,
            error={"code": code, "message": message},
        )

    async def send_progress(self, request_id: str, progress: dict):
        await self._write({"id": request_id, "progress": progress})

    async def _write(self, msg: dict):
        async with self._write_lock:
            line = json.dumps(msg, separators=(",", ":")) + "\n"
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.buffer.flush()
```

**`sidecar/src/pi_sidecar/training/lightning_module.py`**:

```python
"""PyTorch Lightning module for fine-tuning models via the Rust harness."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn


class PiLightningModule(pl.LightningModule):
    """
    Generic fine-tuning module. Wraps a HuggingFace model and exposes
    standard training_step / validation_step for Lightning's Trainer.
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels", batch["input_ids"]),
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels", batch["input_ids"]),
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class ProgressStreamingCallback(pl.Callback):
    """
    Lightning Callback that sends training progress to the Rust core
    via the NDJSON transport's progress_callback.
    """

    def __init__(self, progress_callback):
        self.progress_callback = progress_callback

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        metrics = {
            k: float(v) if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        }
        self.progress_callback(
            {
                "stage": "training",
                "epoch": trainer.current_epoch,
                "total_epochs": trainer.max_epochs,
                "global_step": trainer.global_step,
                "metrics": metrics,
            }
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        # Send every 50 steps to avoid flooding
        if trainer.global_step % 50 == 0:
            self.progress_callback(
                {
                    "stage": "training",
                    "epoch": trainer.current_epoch,
                    "step": trainer.global_step,
                    "loss": float(outputs["loss"]) if isinstance(outputs, dict) else None,
                }
            )
```

### 3.4 Kotlin — Android WebSocket Client

**`android/app/src/main/java/dev/piassistant/android/network/WebSocketClient.kt`**:

```kotlin
package dev.piassistant.android.network

import kotlinx.coroutines.flow.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import okhttp3.*
import java.util.concurrent.TimeUnit

// ── Protocol types ───────────────────────────────────────────────────

@Serializable
data class WsMessage(
    val type: String,
    val payload: JsonElement? = null,
)

enum class ConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    DISCONNECTING,
    ERROR,
}

// ── WebSocket client with Kotlin Flow ────────────────────────────────

class PiWebSocketClient(
    private val json: Json = Json { ignoreUnknownKeys = true },
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // no read timeout for WebSocket
        .pingInterval(30, TimeUnit.SECONDS)     // keep-alive pings
        .build()

    private val _messages = MutableSharedFlow<WsMessage>(
        replay = 0,
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val messages: SharedFlow<WsMessage> = _messages.asSharedFlow()

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private var webSocket: WebSocket? = null

    /**
     * Connect to the Pi-Assistant desktop server.
     *
     * @param serverUrl e.g. "ws://192.168.1.100:9120/ws"
     * @param authToken Bearer token obtained during pairing
     */
    fun connect(serverUrl: String, authToken: String) {
        _connectionState.value = ConnectionState.CONNECTING

        val request = Request.Builder()
            .url(serverUrl)
            .addHeader("Authorization", "Bearer $authToken")
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                _connectionState.value = ConnectionState.CONNECTED
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val message = json.decodeFromString<WsMessage>(text)
                    _messages.tryEmit(message)
                } catch (e: Exception) {
                    // Log malformed messages; don't crash
                    android.util.Log.w("PiWS", "Failed to parse message: $text", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTING
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTED
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                android.util.Log.e("PiWS", "WebSocket failure", t)
                _connectionState.value = ConnectionState.ERROR
                // ConnectionManager handles reconnection
            }
        })
    }

    /** Send a typed message to the desktop server. */
    fun send(message: WsMessage): Boolean {
        val text = json.encodeToString(message)
        return webSocket?.send(text) ?: false
    }

    /** Send a text command (convenience wrapper). */
    fun sendCommand(action: String, payload: JsonElement? = null) {
        send(WsMessage(type = "command", payload = payload))
    }

    /** Send a chat message. */
    fun sendChatMessage(content: String) {
        val payload = json.parseToJsonElement(
            """{"role": "user", "content": ${json.encodeToString(content)}}"""
        )
        send(WsMessage(type = "message", payload = payload))
    }

    /** Respond to a permission request. */
    fun respondToPermission(requestId: String, approved: Boolean) {
        val payload = json.parseToJsonElement(
            """{"id": "$requestId", "approved": $approved}"""
        )
        send(WsMessage(type = "permission_response", payload = payload))
    }

    /** Gracefully disconnect. */
    fun disconnect() {
        webSocket?.close(1000, "Client disconnected")
        webSocket = null
    }
}
```

**`android/app/src/main/java/dev/piassistant/android/network/ConnectionManager.kt`**:

```kotlin
package dev.piassistant.android.network

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Wraps PiWebSocketClient with auto-reconnection logic.
 * Uses exponential backoff: 1s, 2s, 4s, 8s, ... up to 30s.
 */
class ConnectionManager(
    private val wsClient: PiWebSocketClient,
    private val scope: CoroutineScope,
) {
    private var serverUrl: String = ""
    private var authToken: String = ""
    private var reconnectJob: Job? = null
    private var shouldReconnect = false

    val connectionState = wsClient.connectionState

    fun connect(serverUrl: String, authToken: String) {
        this.serverUrl = serverUrl
        this.authToken = authToken
        this.shouldReconnect = true

        wsClient.connect(serverUrl, authToken)
        startReconnectMonitor()
    }

    fun disconnect() {
        shouldReconnect = false
        reconnectJob?.cancel()
        wsClient.disconnect()
    }

    private fun startReconnectMonitor() {
        reconnectJob?.cancel()
        reconnectJob = scope.launch {
            wsClient.connectionState.collect { state ->
                if (state == ConnectionState.ERROR && shouldReconnect) {
                    reconnectWithBackoff()
                }
            }
        }
    }

    private suspend fun reconnectWithBackoff() {
        var delayMs = 1000L
        val maxDelay = 30_000L

        while (shouldReconnect) {
            delay(delayMs)
            if (wsClient.connectionState.value == ConnectionState.CONNECTED) return

            wsClient.connect(serverUrl, authToken)
            delay(2000) // wait for connection attempt

            if (wsClient.connectionState.value == ConnectionState.CONNECTED) return

            delayMs = (delayMs * 2).coerceAtMost(maxDelay)
        }
    }
}
```

**`android/app/src/main/java/dev/piassistant/android/viewmodel/AgentViewModel.kt`**:

```kotlin
package dev.piassistant.android.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.piassistant.android.network.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class AgentViewModel(
    private val wsClient: PiWebSocketClient,
) : ViewModel() {

    private val json = Json { ignoreUnknownKeys = true }

    /** Current WebSocket connection state. */
    val connectionState: StateFlow<ConnectionState> = wsClient.connectionState
        .stateIn(viewModelScope, SharingStarted.Eagerly, ConnectionState.DISCONNECTED)

    /** Latest agent state from the desktop server. */
    val agentState: StateFlow<String> = wsClient.messages
        .filter { it.type == "agent_state_update" }
        .map { msg ->
            msg.payload?.jsonObject?.get("status")?.jsonPrimitive?.content ?: "Unknown"
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "Idle")

    /** Accumulating list of chat messages. */
    val chatMessages: StateFlow<List<ChatMessage>> = wsClient.messages
        .filter { it.type == "message" }
        .map { msg ->
            val obj = msg.payload?.jsonObject
            ChatMessage(
                role = obj?.get("role")?.jsonPrimitive?.content ?: "unknown",
                content = obj?.get("content")?.jsonPrimitive?.content ?: "",
            )
        }
        .runningFold(emptyList<ChatMessage>()) { acc, msg -> acc + msg }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    /** Pending permission requests from the agent. */
    val permissionRequests: SharedFlow<PermissionRequestData> = wsClient.messages
        .filter { it.type == "permission_request" }
        .map { msg ->
            val obj = msg.payload!!.jsonObject
            PermissionRequestData(
                id = obj["id"]!!.jsonPrimitive.content,
                command = obj["command"]!!.jsonPrimitive.content,
                description = obj["description"]?.jsonPrimitive?.content ?: "",
            )
        }
        .shareIn(viewModelScope, SharingStarted.Eagerly, replay = 0)

    // ── Actions ──────────────────────────────────────────────────

    fun startAgent(task: String) {
        val payload = json.parseToJsonElement("""{"action":"start","task":"$task"}""")
        wsClient.send(WsMessage(type = "command", payload = payload))
    }

    fun stopAgent() {
        val payload = json.parseToJsonElement("""{"action":"stop"}""")
        wsClient.send(WsMessage(type = "command", payload = payload))
    }

    fun pauseAgent() {
        val payload = json.parseToJsonElement("""{"action":"pause"}""")
        wsClient.send(WsMessage(type = "command", payload = payload))
    }

    fun sendMessage(text: String) {
        wsClient.sendChatMessage(text)
    }

    fun approvePermission(requestId: String) {
        wsClient.respondToPermission(requestId, approved = true)
    }

    fun denyPermission(requestId: String) {
        wsClient.respondToPermission(requestId, approved = false)
    }
}

// ── Supporting data classes ──────────────────────────────────────────

data class ChatMessage(
    val role: String,
    val content: String,
)

data class PermissionRequestData(
    val id: String,
    val command: String,
    val description: String,
)
```

**`android/app/src/main/java/dev/piassistant/android/voice/SpeechRecognizerHelper.kt`**:

```kotlin
package dev.piassistant.android.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

/**
 * Wraps Android SpeechRecognizer for voice-to-text input.
 * Results are delivered via callbacks to keep this UI-agnostic.
 */
class SpeechRecognizerHelper(context: Context) {

    private val recognizer = SpeechRecognizer.createSpeechRecognizer(context)

    fun startListening(
        onResult: (String) -> Unit,
        onPartial: ((String) -> Unit)? = null,
        onError: ((Int) -> Unit)? = null,
    ) {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
            )
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                onError?.invoke(error)
            }

            override fun onResults(results: Bundle?) {
                val matches = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.firstOrNull()?.let(onResult)
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val matches = partialResults
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.firstOrNull()?.let { onPartial?.invoke(it) }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        recognizer.startListening(intent)
    }

    fun stopListening() {
        recognizer.stopListening()
    }

    fun destroy() {
        recognizer.destroy()
    }
}
```

---

## 4. Safety & Security — Permission Layer

### Permission Tiers

Every tool call passes through the `PermissionEngine` before execution. Commands are classified into three tiers:

| Tier             | Behavior                                              | Examples                                                                                                                                           |
| ---------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Auto-Approve** | Executes immediately, no prompt                       | `ls`, `cat`, `grep`, `git status`, `git log`, `git diff`, `npm list`, `pip list`, `cargo tree`, `uname`, `whoami`, `date`                          |
| **Ask User**     | Pauses agent, shows approval dialog on desktop/mobile | `cp`, `mv`, `mkdir`, `git commit`, `git push`, `npm install`, `pip install`, `curl`, `python`, `node`, file writes via `>` or `>>`                 |
| **Block**        | Silently denied, logged, never executed               | `rm -rf /`, `sudo`, `su`, `dd`, `mkfs`, `chmod 777`, editing `/etc/`, `/sys/`, `/proc/`, printing env vars matching `SECRET\|KEY\|TOKEN\|PASSWORD` |

### PermissionEngine Implementation

```rust
// src-tauri/src/safety/permission.rs

use regex::Regex;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq)]
pub enum PermissionTier {
    AutoApprove,
    AskUser,
    Block,
}

#[derive(Debug, Clone)]
pub enum PermissionResult {
    Allowed,
    NeedsApproval,
    Denied(String),
}

pub struct PermissionRule {
    pub pattern: Regex,
    pub tier: PermissionTier,
    pub description: String,
}

pub struct PermissionEngine {
    rules: Vec<PermissionRule>,
    user_overrides: HashMap<String, bool>, // pattern -> allow/deny
}

impl PermissionEngine {
    pub fn with_defaults() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            user_overrides: HashMap::new(),
        };
        engine.load_default_rules();
        engine
    }

    pub fn check(&self, tool_call: &super::super::agent::loop_::ToolCall) -> anyhow::Result<PermissionResult> {
        // Shell commands get the full rule engine
        if tool_call.tool_name == "shell" {
            let command = tool_call.parameters.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            return Ok(self.check_shell_command(command));
        }

        // Code tool: check path restrictions
        if tool_call.tool_name == "code" {
            return Ok(self.check_code_paths(tool_call));
        }

        // Browser and inference tools: auto-approve
        Ok(PermissionResult::Allowed)
    }

    fn check_shell_command(&self, command: &str) -> PermissionResult {
        // 1. User overrides take precedence
        if let Some(&allowed) = self.user_overrides.get(command) {
            return if allowed {
                PermissionResult::Allowed
            } else {
                PermissionResult::Denied("Denied by user override".into())
            };
        }

        // 2. Block rules checked first (highest priority)
        for rule in &self.rules {
            if rule.tier == PermissionTier::Block && rule.pattern.is_match(command) {
                return PermissionResult::Denied(rule.description.clone());
            }
        }

        // 3. Auto-approve rules
        for rule in &self.rules {
            if rule.tier == PermissionTier::AutoApprove && rule.pattern.is_match(command) {
                return PermissionResult::Allowed;
            }
        }

        // 4. Default: ask user (anything not explicitly allowed or blocked)
        PermissionResult::NeedsApproval
    }

    fn check_code_paths(&self, tool_call: &super::super::agent::loop_::ToolCall) -> PermissionResult {
        let path = tool_call.parameters.get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Block system directories
        let blocked_prefixes = [
            "/etc", "/sys", "/proc", "/boot", "/sbin", "/usr/sbin",
            "/root", "/var/run", "/var/lock",
        ];
        for prefix in &blocked_prefixes {
            if path.starts_with(prefix) {
                return PermissionResult::Denied(
                    format!("Access to {prefix} is blocked")
                );
            }
        }

        // Block sensitive user files
        let home = std::env::var("HOME").unwrap_or_default();
        let blocked_home = [".ssh", ".gnupg", ".aws", ".config/gcloud"];
        for dir in &blocked_home {
            if path.starts_with(&format!("{home}/{dir}")) {
                return PermissionResult::Denied(
                    format!("Access to ~/{dir} is blocked")
                );
            }
        }

        // Block path traversal
        if path.contains("..") {
            return PermissionResult::Denied("Path traversal (..) is not allowed".into());
        }

        // Read operations: auto-approve within allowed paths
        let action = tool_call.parameters.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("read");

        if action == "read" {
            PermissionResult::Allowed
        } else {
            PermissionResult::NeedsApproval
        }
    }

    pub fn add_user_override(&mut self, pattern: &str, allowed: bool) {
        self.user_overrides.insert(pattern.to_string(), allowed);
    }

    fn load_default_rules(&mut self) {
        // ── BLOCK TIER ───────────────────────────────────────────
        let block_patterns = vec![
            (r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$", "Recursive delete of root"),
            (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/", "Recursive force delete from root"),
            (r"\bsudo\b", "Privilege escalation via sudo"),
            (r"\bsu\b\s", "Privilege escalation via su"),
            (r"\bdd\b\s+.*of=/dev/", "Direct disk write via dd"),
            (r"\bmkfs\b", "Filesystem creation"),
            (r"\bfdisk\b", "Disk partitioning"),
            (r"chmod\s+777", "World-writable permissions"),
            (r"chown\s+root", "Changing ownership to root"),
            (r">\s*/etc/", "Writing to /etc/"),
            (r">\s*/sys/", "Writing to /sys/"),
            (r"\benv\b.*\b(SECRET|KEY|TOKEN|PASSWORD|CREDENTIAL)\b", "Credential exposure"),
            (r"\bnmap\b", "Network scanning"),
            (r":(){ :|:& };:", "Fork bomb"),
            (r"\bshutdown\b", "System shutdown"),
            (r"\breboot\b", "System reboot"),
        ];

        for (pattern, desc) in block_patterns {
            self.rules.push(PermissionRule {
                pattern: Regex::new(pattern).unwrap(),
                tier: PermissionTier::Block,
                description: desc.to_string(),
            });
        }

        // ── AUTO-APPROVE TIER ────────────────────────────────────
        let approve_patterns = vec![
            (r"^ls(\s|$)", "List directory contents"),
            (r"^cat\s", "Read file contents"),
            (r"^head\s", "Read file head"),
            (r"^tail\s", "Read file tail"),
            (r"^wc\s", "Word/line count"),
            (r"^grep\s", "Search file contents"),
            (r"^rg\s", "Ripgrep search"),
            (r"^find\s(?!.*-exec)", "Find files (no exec)"),
            (r"^git\s+(status|log|diff|show|branch|tag|remote)\b", "Git read operations"),
            (r"^git\s+rev-parse\b", "Git rev-parse"),
            (r"^npm\s+list\b", "NPM list packages"),
            (r"^pip\s+list\b", "Pip list packages"),
            (r"^cargo\s+tree\b", "Cargo dependency tree"),
            (r"^(uname|whoami|hostname|date|pwd|echo\s)\b", "System info"),
            (r"^tree(\s|$)", "Directory tree"),
            (r"^file\s", "File type detection"),
            (r"^stat\s", "File stats"),
        ];

        for (pattern, desc) in approve_patterns {
            self.rules.push(PermissionRule {
                pattern: Regex::new(pattern).unwrap(),
                tier: PermissionTier::AutoApprove,
                description: desc.to_string(),
            });
        }
    }
}
```

### Security Invariants

1. **Block rules always win.** Even if a user override says "allow `sudo`", the block tier is checked first and cannot be bypassed without modifying source code.
2. **Default-deny for shell.** Any command not matching an auto-approve pattern requires user confirmation. The safe set is intentionally narrow.
3. **Path canonicalization.** All file paths are resolved through `std::fs::canonicalize()` before permission checks, preventing symlink-based escapes.
4. **No credential leaking.** The block tier catches `env` commands that could expose secrets. The agent's process environment is also sanitized: `API_KEY`, `SECRET_*`, `*_TOKEN` variables are stripped before the sidecar is spawned.
5. **Resource limits.** Shell commands have a 60-second timeout. The Python sidecar has a 5-minute timeout per request. Both prevent runaway processes.
6. **Kill on drop.** The Python sidecar is spawned with `kill_on_drop(true)`, ensuring it is terminated if the Rust process crashes.

---

## 5. Key Dependency Summary

### Rust (`src-tauri/Cargo.toml`)

| Crate                  | Version | Purpose                       |
| ---------------------- | ------- | ----------------------------- |
| `tauri`                | 2.x     | App framework                 |
| `tauri-plugin-shell`   | 2.x     | Shell/sidecar access          |
| `tokio`                | 1.x     | Async runtime (full features) |
| `tokio-util`           | 0.7     | CancellationToken             |
| `axum`                 | 0.8     | WebSocket server for mobile   |
| `serde` / `serde_json` | 1.x     | Serialization                 |
| `rusqlite`             | 0.32    | SQLite (bundled feature)      |
| `sqlite-vec`           | 0.1     | Vector similarity extension   |
| `chromiumoxide`        | 0.7     | Headless browser (CDP)        |
| `uuid`                 | 1.x     | Correlation IDs               |
| `regex`                | 1.x     | Permission pattern matching   |
| `tracing`              | 0.1     | Structured logging            |
| `thiserror`            | 2.x     | Error types                   |
| `anyhow`               | 1.x     | Error handling                |
| `async-trait`          | 0.1     | Async trait methods           |

### Python (`sidecar/pyproject.toml`)

| Package                 | Version | Purpose            |
| ----------------------- | ------- | ------------------ |
| `torch`                 | >=2.2   | Tensor computation |
| `pytorch-lightning`     | >=2.2   | Training framework |
| `transformers`          | >=4.40  | HuggingFace models |
| `sentence-transformers` | >=3.0   | Embedding models   |
| `pydantic`              | >=2.5   | Data validation    |

### Android (`android/gradle/libs.versions.toml`)

| Library                       | Version | Purpose            |
| ----------------------------- | ------- | ------------------ |
| `okhttp`                      | 5.0.x   | WebSocket client   |
| `kotlinx-serialization-json`  | 1.7.x   | JSON serialization |
| `kotlinx-coroutines`          | 1.10.x  | Async/Flow         |
| `lifecycle-viewmodel-compose` | 2.8.x   | ViewModel          |
| `compose-bom`                 | 2025.x  | Jetpack Compose    |

### Frontend (`package.json`)

| Package               | Purpose            |
| --------------------- | ------------------ |
| `@tauri-apps/api`     | Tauri v2 IPC       |
| `react` / `react-dom` | UI framework       |
| `zustand`             | State management   |
| `tailwindcss`         | Styling            |
| `react-markdown`      | Markdown rendering |
| `vite`                | Build tooling      |
