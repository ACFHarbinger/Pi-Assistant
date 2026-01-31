# Agent System

This document describes how the Pi-Assistant agent works: its execution loop, tool system, planning strategy, and memory integration.

---

## Agent Loop

The agent runs as an asynchronous Tokio task inside the Rust core. It follows a **plan-execute-observe** cycle, iterating until the task is complete, manually stopped, or an iteration limit is reached.

### State Machine

```
           ┌──────────────────────────┐
           │                          │
     ┌─────▼─────┐   Start     ┌─────┴─────┐
     │   Idle     │────────────►│  Running   │◄──── Resume
     └───────────┘             └─────┬─────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                    Pause/Ask    Completed     Error/Stop
                         │            │            │
                   ┌─────▼─────┐ ┌────▼────┐ ┌────▼────┐
                   │  Paused   │ │ Stopped │ │ Stopped │
                   │ (question │ │ (done)  │ │ (error) │
                   │  or perm) │ └─────────┘ └─────────┘
                   └───────────┘
```

| State       | Description                                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Idle**    | No task is running. Waiting for user input.                                                                                                         |
| **Running** | Agent is actively iterating. Broadcasts current iteration number.                                                                                   |
| **Paused**  | Agent is blocked. Either asking the user a question or awaiting permission approval. Contains the question/permission request in its state payload. |
| **Stopped** | Terminal state. Includes a reason: `Completed`, `ManualStop`, `Error`, or `IterationLimit`.                                                         |

### Iteration Cycle

Each iteration of the agent loop performs these steps in order:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration N                                                     │
│                                                                 │
│  1. Check cancellation token                                    │
│  2. Check iteration limit                                       │
│  3. Drain command channel (handle Pause/Stop if received)       │
│  4. Retrieve relevant context from memory                       │
│  5. Call LLM planner (via Python sidecar) → get AgentPlan       │
│  6. If plan.question: pause, wait for user answer               │
│  7. For each tool_call in plan:                                 │
│     a. PermissionEngine.check(tool_call)                        │
│     b. If NeedsApproval → pause, wait for user decision         │
│     c. If Allowed → ToolRegistry.execute(tool_call)             │
│     d. If Denied → skip, log reason                             │
│     e. Store tool result in memory                              │
│  8. If plan.is_complete → transition to Stopped(Completed)      │
│  9. Increment iteration counter                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cancellation

The loop uses `tokio_util::sync::CancellationToken` for cooperative cancellation. When the user clicks "Stop":

1. The frontend sends a `stop_agent` Tauri command.
2. The command handler calls `cancel_token.cancel()`.
3. At the top of the next iteration, the loop checks `cancel_token.is_cancelled()` and exits cleanly.

This ensures in-progress tool executions finish before the loop terminates — no orphaned processes or half-written files.

### Concurrency Model

```
Main Tauri Thread
    │
    ├── tokio::spawn(agent_loop)          ← the agent
    │       │
    │       ├── sidecar.request(...)      ← IPC to Python (awaits response)
    │       ├── tool_registry.execute(...)← tool execution (may spawn child processes)
    │       └── memory.store(...)         ← SQLite writes
    │
    ├── tokio::spawn(ws_server)           ← WebSocket for mobile
    │
    └── tokio::spawn(state_broadcaster)   ← watch channel → Tauri events
```

Only one agent loop runs at a time. Starting a new task while one is running requires stopping the current one first.

---

## Tool System

Tools are the agent's hands. Each tool implements the `Tool` trait and is registered in the `ToolRegistry`.

### Tool Trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult>;
    fn permission_tier(&self) -> PermissionTier;
}
```

### Available Tools

#### Shell (`shell`)

Executes arbitrary shell commands via `tokio::process::Command`.

**Parameters:**

```json
{
  "command": "git status",
  "working_dir": "/home/user/project",
  "timeout_secs": 60
}
```

**Returns:**

```json
{
  "stdout": "On branch main\nnothing to commit...",
  "stderr": "",
  "exit_code": 0,
  "duration_ms": 42
}
```

**Safety:** Every command passes through the `PermissionEngine` before execution. Commands are matched against regex rules to determine the tier. Default timeout: 60 seconds.

#### Browser (`browser`)

Controls a headless Chrome instance via Chrome DevTools Protocol (using the `chromiumoxide` crate).

**Actions:**

| Action         | Description                              |
| -------------- | ---------------------------------------- |
| `navigate`     | Go to a URL, wait for page load          |
| `extract_text` | Get the visible text content of the page |
| `extract_html` | Get the raw HTML                         |
| `screenshot`   | Take a PNG screenshot                    |
| `click`        | Click an element by CSS selector         |
| `fill`         | Fill a form field by CSS selector        |
| `evaluate`     | Execute JavaScript and return the result |

**Parameters (navigate example):**

```json
{
  "action": "navigate",
  "url": "https://docs.rs",
  "wait_for": "networkidle0"
}
```

**Safety:** Domain allowlist restricts which URLs the browser can access. By default, only `localhost` and `127.0.0.1` are allowed. Users configure additional domains explicitly.

#### Code (`code`)

Reads, writes, and patches files on disk.

**Actions:**

| Action       | Parameters           | Description                                |
| ------------ | -------------------- | ------------------------------------------ |
| `read`       | `path`               | Read file contents                         |
| `write`      | `path`, `content`    | Write/overwrite a file                     |
| `patch`      | `path`, `old`, `new` | Replace a string in a file                 |
| `list`       | `path`, `pattern`    | List directory contents with optional glob |
| `create_dir` | `path`               | Create a directory (mkdir -p)              |

**Safety:** Path-based restrictions. Blocked directories: `/etc`, `/sys`, `/proc`, `/boot`, `~/.ssh`, `~/.gnupg`, `~/.aws`. Path traversal (`..`) is rejected. All paths are canonicalized before checking. Read operations are auto-approved; writes require user confirmation.

#### Python Inference (`python_inference`)

Delegates to the Python sidecar for ML tasks. This is a bridge tool — it translates tool calls into IPC requests.

**Actions:**

| Action     | IPC Method           | Description                             |
| ---------- | -------------------- | --------------------------------------- |
| `embed`    | `inference.embed`    | Generate text embeddings (384-dim)      |
| `complete` | `inference.complete` | Text generation via loaded LLM          |
| `plan`     | `inference.plan`     | Agent planning step (structured output) |

**Safety:** Auto-approved. The Python sidecar runs in the same security context as the Rust process.

### Tool Registration

Tools are registered at app startup in `ToolRegistry::new()`. The registry provides:

- **Discovery**: The LLM planner receives tool names, descriptions, and parameter schemas to decide which tools to call.
- **Dispatch**: Given a `ToolCall { tool_name, parameters }`, the registry looks up the tool by name and calls `execute()`.
- **Schema validation**: Tool parameters can be validated against their JSON Schema before execution.

---

## Planning

The agent uses an LLM to decide what to do at each iteration. The planner receives:

1. **Task description** — the user's original request
2. **Iteration number** — how many cycles have elapsed
3. **Retrieved context** — relevant memories (recent messages + vector search results)
4. **Available tools** — names, descriptions, and parameter schemas

The planner returns an `AgentPlan`:

```json
{
  "reasoning": "The user wants to add a login page. I need to first check what framework they're using.",
  "tool_calls": [
    {
      "tool_name": "shell",
      "parameters": { "command": "cat package.json" }
    }
  ],
  "question": null,
  "is_complete": false
}
```

### Plan Fields

| Field         | Type         | Description                                              |
| ------------- | ------------ | -------------------------------------------------------- |
| `reasoning`   | `string`     | Chain-of-thought explanation (logged, shown in UI)       |
| `tool_calls`  | `ToolCall[]` | Ordered list of tools to execute this iteration          |
| `question`    | `string?`    | If set, the agent pauses and asks the user this question |
| `is_complete` | `bool`       | If true, the task is done and the loop terminates        |

### Planning Strategy

The planner is implemented in the Python sidecar (`inference.plan` method) to keep LLM interaction logic in Python where the ML ecosystem is strongest. The Rust side is responsible for:

- Assembling the context (memory retrieval)
- Sending the planning request via IPC
- Parsing the structured plan response
- Executing the plan (tools, permissions, state transitions)

This separation means you can swap LLM providers (local model, OpenAI API, Anthropic API) by changing the Python inference engine without touching the Rust agent loop.

---

## Memory Integration

### How Memory Feeds the Agent

```
                 Agent Iteration N
                        │
           ┌────────────▼────────────┐
           │  1. Build query string  │
           │     task + recent ctx   │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  2. Generate embedding  │◄─── Python sidecar
           │     (384-dim vector)    │     inference.embed
           └────────────┬────────────┘
                        │
        ┌───────────────┼───────────────┐
        │                               │
┌───────▼───────┐              ┌────────▼────────┐
│ Vector Search │              │ SQL Recency      │
│ sqlite-vec    │              │ Last 20 messages │
│ Top-10 by     │              │ from session     │
│ cosine sim    │              │                  │
└───────┬───────┘              └────────┬─────────┘
        │                               │
        └───────────┬───────────────────┘
                    │
           ┌────────▼────────┐
           │  3. Merge +     │
           │  deduplicate +  │
           │  rank           │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  4. Inject into │
           │  LLM planner    │
           │  prompt         │
           └─────────────────┘
```

### What Gets Stored

Every significant event is persisted:

| Event               | Table              | Also Embedded?          |
| ------------------- | ------------------ | ----------------------- |
| User message        | `messages`         | Yes                     |
| Agent response      | `messages`         | Yes                     |
| Tool call + result  | `tool_executions`  | Result text is embedded |
| Task creation       | `tasks`            | Description is embedded |
| Permission decision | `permission_cache` | No                      |

### Memory Lifecycle

1. **Within a session**: Recent messages are retrieved by recency (SQL query on `session_id`, ordered by `created_at`).
2. **Across sessions**: Vector similarity search finds relevant context from any past session. This is how the agent "remembers" things from weeks ago.
3. **Pruning**: Old embeddings can be pruned by age or count. The SQL tables are append-only (no deletion) to maintain a full audit trail.

---

## Agent Commands

Commands flow from the UI or mobile client to the agent loop via a `tokio::sync::mpsc` channel.

| Command                                        | Source      | Effect                                          |
| ---------------------------------------------- | ----------- | ----------------------------------------------- |
| `Start { task, max_iterations }`               | UI / Mobile | Spawns the agent loop                           |
| `Stop`                                         | UI / Mobile | Cancels the token, loop exits at next check     |
| `Pause`                                        | UI / Mobile | Loop blocks waiting for `Resume` or `Stop`      |
| `Resume`                                       | UI / Mobile | Unblocks a paused loop                          |
| `AnswerQuestion { response }`                  | UI / Mobile | Provides the user's answer to an agent question |
| `ApprovePermission { id, approved, remember }` | UI / Mobile | Responds to a permission request                |

### Event Flow

State changes are broadcast via `tokio::sync::watch`:

```
Agent Loop ──► watch::Sender<AgentState>
                    │
                    ├──► Tauri event emitter ──► React frontend (listen)
                    │
                    └──► WebSocket broadcaster ──► Android client
```

Both the desktop and mobile UI receive the same `AgentState` updates in real time.
