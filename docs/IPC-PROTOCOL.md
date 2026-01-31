# IPC Protocol Specification

Detailed specification for all inter-process communication in Pi-Assistant.

---

## Overview

Pi-Assistant has two IPC boundaries:

| Boundary               | Transport                     | Format                          | Direction     |
| ---------------------- | ----------------------------- | ------------------------------- | ------------- |
| **Rust <-> Python**    | stdin/stdout of child process | NDJSON (newline-delimited JSON) | Bidirectional |
| **Desktop <-> Mobile** | WebSocket (TCP :9120)         | JSON messages                   | Bidirectional |

Both protocols use JSON as the wire format. Message schemas are defined in `protocol/schemas/`.

---

## 1. Rust <-> Python (NDJSON over stdio)

### Transport

- The Rust core spawns the Python sidecar as a child process.
- Rust writes JSON lines to the child's **stdin**.
- Python writes JSON lines to its **stdout**.
- Python's **stderr** is inherited by the Rust process and used for logging only (never protocol data).
- Each message is exactly one line (no embedded newlines in the JSON).
- Lines are terminated by `\n` (LF).
- Encoding: UTF-8.

### Message Types

#### Request (Rust -> Python)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "inference.complete",
  "params": {
    "prompt": "Summarize this code...",
    "model_id": "default-llm",
    "max_tokens": 512,
    "temperature": 0.7
  }
}
```

| Field    | Type             | Required | Description                                                   |
| -------- | ---------------- | -------- | ------------------------------------------------------------- |
| `id`     | string (UUID v4) | Yes      | Correlation ID. The response must echo this ID.               |
| `method` | string           | Yes      | Namespaced method name (e.g., `inference.complete`).          |
| `params` | object           | Yes      | Method-specific parameters. Can be `{}` for no-param methods. |

#### Response (Python -> Rust)

**Success:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "text": "This code implements...",
    "tokens_used": 234,
    "model_id": "default-llm"
  }
}
```

**Error:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "Model 'default-llm' is not loaded. Call model.load first."
  }
}
```

| Field           | Type   | Required       | Description                                         |
| --------------- | ------ | -------------- | --------------------------------------------------- |
| `id`            | string | Yes            | Must match the request's `id`.                      |
| `result`        | any    | Conditional    | Present on success. Absent on error.                |
| `error`         | object | Conditional    | Present on error. Absent on success.                |
| `error.code`    | string | Yes (if error) | Machine-readable error code (SCREAMING_SNAKE_CASE). |
| `error.message` | string | Yes (if error) | Human-readable error description.                   |

A response must have exactly one of `result` or `error`, never both, never neither.

#### Progress (Python -> Rust)

For long-running operations (training), Python sends intermediate progress messages. These do **not** complete the pending request.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": {
    "stage": "training",
    "epoch": 3,
    "total_epochs": 10,
    "global_step": 1500,
    "metrics": {
      "train_loss": 0.342,
      "accuracy": 0.89
    }
  }
}
```

| Field      | Type   | Required | Description                             |
| ---------- | ------ | -------- | --------------------------------------- |
| `id`       | string | Yes      | Must match the original request's `id`. |
| `progress` | object | Yes      | Freeform progress data.                 |

**Rule:** A message with `"progress"` is a progress update. A message with `"result"` or `"error"` is a final response. The Rust router distinguishes them by checking which field is present.

### Methods

#### Lifecycle

| Method               | Params | Response                               | Description                                                       |
| -------------------- | ------ | -------------------------------------- | ----------------------------------------------------------------- |
| `health.ping`        | `{}`   | `{"status": "ok", "version": "0.1.0"}` | Health check. Called after sidecar starts.                        |
| `lifecycle.shutdown` | `{}`   | `{"status": "shutting_down"}`          | Graceful shutdown request. Python exits shortly after responding. |

#### Inference

| Method               | Params                                                | Response                                                  | Description                                                        |
| -------------------- | ----------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------ |
| `inference.complete` | `{"prompt", "model_id", "max_tokens", "temperature"}` | `{"text", "tokens_used", "model_id"}`                     | Text generation.                                                   |
| `inference.embed`    | `{"text", "model_id?"}`                               | `{"embedding": [float; 384]}`                             | Generate text embedding vector. Default model: `all-MiniLM-L6-v2`. |
| `inference.plan`     | `{"task", "iteration", "context"}`                    | `{"reasoning", "tool_calls", "question?", "is_complete"}` | Agent planning step. Returns structured plan for the agent loop.   |

**`inference.plan` response schema:**

```json
{
  "reasoning": "I need to read the package.json to understand the project setup.",
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

| Field         | Type                               | Description                                                     |
| ------------- | ---------------------------------- | --------------------------------------------------------------- |
| `reasoning`   | string                             | Chain-of-thought explanation. Shown in UI, stored in memory.    |
| `tool_calls`  | array of `{tool_name, parameters}` | Ordered list of tools to execute. Can be empty (thinking step). |
| `question`    | string or null                     | If non-null, the agent pauses and asks the user this question.  |
| `is_complete` | bool                               | If true, the task is finished. The agent loop terminates.       |

#### Training

| Method           | Params                                   | Response                                           | Description                                                                        |
| ---------------- | ---------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `training.start` | `{"model_id", "dataset_path", "config"}` | `{"status": "completed", "model_path", "metrics"}` | Start fine-tuning. Sends progress updates. Final response when training completes. |
| `training.stop`  | `{"model_id?"}`                          | `{"status": "stopped"}`                            | Cancel a running training job.                                                     |

**`training.start` config:**

```json
{
  "model_id": "my-custom-model",
  "dataset_path": "/path/to/dataset.jsonl",
  "config": {
    "base_model": "gpt2",
    "max_epochs": 10,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "val_split": 0.1
  }
}
```

#### Model Management

| Method       | Params                     | Response                                             | Description                                         |
| ------------ | -------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| `model.list` | `{}`                       | `{"models": [{"id", "path", "version", "size_mb"}]}` | List all available models.                          |
| `model.load` | `{"model_id"}`             | `{"status": "loaded", "model_id"}`                   | Load a model into memory for inference.             |
| `model.save` | `{"model_id", "version?"}` | `{"status": "saved", "path"}`                        | Save current model state with optional version tag. |

### Error Codes

| Code                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `MODEL_NOT_LOADED`     | Requested model is not loaded. Call `model.load` first. |
| `MODEL_NOT_FOUND`      | Model ID doesn't exist in the registry.                 |
| `TRAINING_IN_PROGRESS` | Cannot start training while another is running.         |
| `TRAINING_NOT_RUNNING` | `training.stop` called but no training is active.       |
| `INVALID_PARAMS`       | Request params failed validation.                       |
| `INFERENCE_ERROR`      | Model inference failed (OOM, corrupt weights, etc.).    |
| `UNKNOWN_METHOD`       | The method name is not recognized.                      |
| `INTERNAL_ERROR`       | Unhandled exception in the sidecar.                     |

### Timeout Policy

| Method Category      | Timeout      | Rationale                                                             |
| -------------------- | ------------ | --------------------------------------------------------------------- |
| `health.*`           | 10s          | Should be instant                                                     |
| `inference.embed`    | 30s          | Small model, fast                                                     |
| `inference.complete` | 120s         | LLM generation can be slow                                            |
| `inference.plan`     | 120s         | Same as completion                                                    |
| `training.start`     | 300s (5 min) | Training is long-running; progress messages keep the connection alive |
| `model.*`            | 60s          | Model loading from disk                                               |

The Rust side enforces timeouts. On timeout, the pending request is removed from the routing table, and the caller receives an error.

### Concurrency

Multiple requests can be in-flight simultaneously. Python handles them concurrently using `asyncio.create_task()`. The correlation ID ensures responses are routed to the correct caller regardless of arrival order.

**Example:**

```
Rust -> Python: {"id":"aaa","method":"inference.embed","params":{...}}
Rust -> Python: {"id":"bbb","method":"inference.complete","params":{...}}
Python -> Rust: {"id":"aaa","result":{"embedding":[...]}}         ← embed finishes first
Python -> Rust: {"id":"bbb","result":{"text":"..."}}              ← complete finishes second
```

---

## 2. Desktop <-> Mobile (WebSocket)

### Transport

- Protocol: WebSocket over TCP.
- Default endpoint: `ws://DESKTOP_IP:9120/ws`
- The Rust core runs an Axum-based WebSocket server.
- JSON text frames only (no binary frames).

### Authentication

1. **Pairing (first connection):**
   - Desktop displays a 6-digit code.
   - Android app sends: `{"type": "auth", "payload": {"code": "123456"}}`.
   - Server responds: `{"type": "auth_result", "payload": {"token": "...", "success": true}}`.
   - The token is stored on the Android device for future connections.

2. **Reconnection:**
   - Android sends the `Authorization: Bearer <token>` header on the WebSocket upgrade request.
   - If the token is valid, the connection proceeds without a pairing code.

### Message Envelope

Every WebSocket message follows this structure:

```json
{
  "type": "<message_type>",
  "payload": { ... }
}
```

| Field     | Type          | Required | Description                                                          |
| --------- | ------------- | -------- | -------------------------------------------------------------------- |
| `type`    | string (enum) | Yes      | Message type identifier.                                             |
| `payload` | object        | No       | Type-specific data. Can be null or absent for types with no payload. |

### Message Types

#### Server -> Client

| Type                 | Payload                                                                                         | Description                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `auth_result`        | `{"token": string, "success": bool, "error?": string}`                                          | Response to authentication attempt.                 |
| `agent_state_update` | `AgentState` (see below)                                                                        | Emitted whenever the agent state changes.           |
| `message`            | `{"role": string, "content": string, "timestamp": string}`                                      | Chat message from the agent or system.              |
| `permission_request` | `{"id": string, "tool_name": string, "command": string, "description": string, "tier": string}` | Agent needs user approval for a tool call.          |
| `task_update`        | `{"task_id": string, "iteration": number, "status": string}`                                    | Progress on the current task.                       |
| `ping`               | `null`                                                                                          | Keep-alive ping. Client should respond with `pong`. |

#### Client -> Server

| Type                  | Payload                                                          | Description                                                 |
| --------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `auth`                | `{"code": string}`                                               | Pairing code during initial setup.                          |
| `command`             | `{"action": string, "task?": string, "max_iterations?": number}` | Agent control. Actions: `start`, `stop`, `pause`, `resume`. |
| `message`             | `{"role": "user", "content": string}`                            | User chat message or answer to agent question.              |
| `permission_response` | `{"id": string, "approved": bool, "remember?": bool}`            | Response to a permission request.                           |
| `pong`                | `null`                                                           | Response to server ping.                                    |

#### AgentState Payload

```json
{
  "status": "Running",
  "data": {
    "task_id": "550e8400-...",
    "iteration": 5
  }
}
```

Status values: `"Idle"`, `"Running"`, `"Paused"`, `"Stopped"`.

The `data` field varies by status:

- **Idle**: `null`
- **Running**: `{"task_id", "iteration"}`
- **Paused**: `{"task_id", "question?", "awaiting_permission?"}`
- **Stopped**: `{"task_id", "reason"}` where reason is `"Completed"`, `"ManualStop"`, `"Error"`, or `"IterationLimit"`.

### Connection Lifecycle

```
Client                              Server
  |                                    |
  |-- WebSocket upgrade (+ Bearer) --> |
  |                                    |-- validate token
  |<-- 101 Switching Protocols --------|
  |                                    |
  |<-- agent_state_update (current) ---|   (immediate state sync)
  |                                    |
  |-- command: start ----------------->|
  |<-- agent_state_update: Running ----|
  |<-- message: "I'll start by..." ---|
  |<-- task_update: iteration 0 ------|
  |                                    |
  |<-- permission_request -------------|
  |-- permission_response: approved -->|
  |<-- agent_state_update: Running ----|
  |                                    |
  |<-- agent_state_update: Stopped ----|
  |                                    |
  |-- close --------------------------> |
```

On connect, the server immediately sends the current `AgentState` so the client is in sync without polling.

### Reconnection

The Android `ConnectionManager` implements exponential backoff:

| Attempt | Delay     |
| ------- | --------- |
| 1       | 1s        |
| 2       | 2s        |
| 3       | 4s        |
| 4       | 8s        |
| 5       | 16s       |
| 6+      | 30s (max) |

On successful reconnect, the server re-sends the current agent state.

---

## 3. Schema Files

All protocol types are defined as JSON Schema in `protocol/schemas/`:

| File                       | Defines                                          |
| -------------------------- | ------------------------------------------------ |
| `ipc-message.schema.json`  | Request, Response, Progress (Rust <-> Python)    |
| `ws-message.schema.json`   | WebSocket message envelope and all payload types |
| `agent-state.schema.json`  | AgentState enum with per-variant data            |
| `tool-request.schema.json` | ToolCall and ToolResult structures               |
| `permission.schema.json`   | PermissionRequest and PermissionResponse         |

These schemas are the **source of truth**. The Rust, Python, and Kotlin implementations must conform to them. When updating the protocol:

1. Modify the JSON Schema first.
2. Update the Rust serde types (`protocol/rust/src/lib.rs`).
3. Update the Python Pydantic models (`protocol/python/pi_protocol/messages.py`).
4. Update the Kotlin data classes (`protocol/kotlin/PiProtocol.kt`).
5. Run cross-language tests to verify compatibility.
