# Pi-Assistant Roadmap

This roadmap outlines planned features and enhancements across the Pi-Assistant system. Items are organized by domain and roughly ordered by priority within each section. Dependencies between items are noted where relevant.

---

## 1. Agent Intelligence & Planning

- [x] **1.1 Hierarchical Task Decomposition** [IMPLEMENTED]
  - Introduced a hierarchical planner that breaks complex tasks into subtask trees.
  - Subtask state management (Pending, Running, Blocked, Done) and persistence.
  - TaskTree visualizer in the UI with progress indicators.

### 1.2 Reflection & Self-Correction Loop

After each tool execution, add an explicit reflection step where the agent evaluates whether the result matches expectations. If the result diverges (e.g., a command fails, output is unexpected), the agent should revise its plan before continuing rather than blindly proceeding to the next tool call.

- Compare expected vs. actual tool output using lightweight classifier
- Maintain a per-task error budget: after N consecutive failures, pause and ask the user
- Log reflection reasoning alongside tool results for transparency

### 1.3 Cost-Aware Planning

When multiple LLM providers are available (local models, Anthropic API, OpenAI API), the planner should consider the cost and latency tradeoff for each planning step. Simple decisions (file reads, directory listings) can use a small local model. Complex reasoning (architecture decisions, debugging) can escalate to a larger cloud model.

- Token budget per task, configurable by user
- Automatic provider selection based on task complexity estimate
- Dashboard showing cumulative cost and token usage per session

### 1.4 Multi-Agent Collaboration

Extend the existing agent pool to support concurrent agents working on related subtasks. Agents share memory but maintain separate execution contexts. A coordinator agent assigns work and merges results.

- Shared memory namespace with per-agent read/write isolation
- Conflict resolution when two agents modify the same file
- Visual timeline showing parallel agent activity

---

## 2. Tool Expansion

### 2.1 Task-Specific ML Model Training

Expand the existing PyTorch Lightning training tool into a full workflow for training, evaluating, and deploying task-specific models that the agent can then use as specialized tools.

**Training Pipeline:**

- User provides a dataset (CSV, JSON, or a directory of files) and describes the task
- Agent selects an appropriate base model architecture (classification, sequence-to-sequence, regression, NER, etc.)
- Agent configures hyperparameters based on dataset size and task type
- Training runs in the Python sidecar with real-time metric streaming to the UI
- Checkpoints are saved to the model registry with metadata (task, accuracy, training config)

**Supported Task Types:**

- Text classification (sentiment, intent, topic)
- Named entity recognition
- Text summarization / paraphrasing
- Tabular regression and classification
- Image classification (with torchvision integration)
- Time-series forecasting

**Agent Integration:**

- Trained models register as new tools in the `ToolRegistry` automatically
- The planner can invoke trained models by name (e.g., `classify_support_ticket`, `predict_sales`)
- Model versioning: the agent can retrain and compare metrics across versions
- Active learning: the agent identifies low-confidence predictions and asks the user for labels

### 2.2 Retrieval-Augmented Generation (RAG) Tool

A dedicated tool that lets the agent ingest documents (PDFs, markdown, code repositories, web pages) into a searchable vector store and query them during planning.

- Chunking strategies: fixed-size, sentence-boundary, AST-aware (for code)
- Hybrid retrieval: combine vector similarity with BM25 keyword matching
- Source attribution: every retrieved chunk links back to its origin document and page/line
- Incremental indexing: add new documents without re-indexing the entire corpus

### 2.3 Database Tool

Direct SQL access to user-specified databases (SQLite, PostgreSQL, MySQL) for data exploration, querying, and modification.

- Read-only mode by default; writes require explicit permission approval
- Schema introspection: the agent can discover tables, columns, and relationships
- Query explanation: the agent can run `EXPLAIN` and interpret query plans
- Result visualization: tables rendered in the UI, with optional chart generation

### 2.4 API Integration Tool

A generic HTTP client tool that the agent can use to interact with external APIs.

- OpenAPI/Swagger spec ingestion: given a spec URL, the agent learns all available endpoints
- Authentication management: OAuth2 flows, API key storage (encrypted in permission-gated keyring)
- Rate limiting awareness: the agent respects `Retry-After` headers and backs off appropriately
- Response caching: avoid redundant API calls within the same task

### 2.5 Drawing & Diagram Tool

Generate and render diagrams (flowcharts, sequence diagrams, architecture diagrams, ER diagrams) using Mermaid or D2 syntax, rendered live in the Canvas.

- Agent can produce diagrams as part of its reasoning (e.g., "here's the architecture I'm proposing")
- Diagrams stored as versioned artifacts alongside task history
- User can request modifications ("add a cache layer between the API and database")

### 2.6 Containerized Execution Environment

Run agent tool calls inside isolated Docker containers for safety and reproducibility.

- Per-task container with configurable base image
- Filesystem snapshot before/after tool execution for diff visualization
- Network isolation modes: none, host-only, full
- Resource limits (CPU, memory, disk) configurable per task

---

## 3. Memory & Knowledge

### 3.1 Structured Knowledge Graph

Augment the current flat embedding store with a knowledge graph that captures relationships between entities the agent has encountered.

- Entities: files, functions, people, projects, concepts, tools
- Relations: `depends_on`, `authored_by`, `related_to`, `modified_in`, `calls`
- Graph stored in SQLite using adjacency list tables
- Graph queries inform the planner: "what files are related to the authentication module?"
- Visualization: interactive graph explorer in the UI (force-directed layout)

### 3.2 Episodic Memory with Summarization

Long-running sessions generate large volumes of messages and tool results. Introduce an episodic memory layer that summarizes completed task episodes into concise narrative summaries.

- After a task completes, generate a summary: what was asked, what was done, what was the outcome
- Summaries are embedded and searchable alongside raw messages
- Configurable retention policy: keep raw data for N days, keep summaries indefinitely
- Summary quality improves over time via user feedback (thumbs up/down on summary accuracy)

### 3.3 Markdown-Based Knowledge Files

Allow the agent to maintain a set of markdown files as a persistent, human-readable knowledge base.

- `~/.pi-assistant/knowledge/` directory organized by topic
- Agent creates and updates files as it learns (e.g., `projects/my-app.md`, `preferences/coding-style.md`)
- Files are indexed for vector search alongside database entries
- User can edit files directly; changes are picked up on next retrieval
- Agent references specific sections when explaining its reasoning: "Based on your coding preferences (see preferences/coding-style.md)..."

### 3.4 Conversation Branching & Replay

Allow users to branch a conversation at any point, explore an alternative approach, and optionally merge results back.

- Branch creates a snapshot of memory state at that point
- Each branch maintains independent tool execution history
- Merge operation combines tool results and memory from two branches
- UI shows branch tree with diff between branch outcomes

### 3.5 Cross-Session Learning

Use patterns from past sessions to improve future performance.

- Track which tool sequences successfully completed specific task types
- Build a task-type classifier from historical data
- Suggest proven tool sequences to the planner for similar new tasks
- User can mark sessions as "good examples" to weight them higher in retrieval

---

## 4. Human-Agent Interaction

### 4.1 Rich Permission Workflows

Extend the current 3-tier permission system with more granular and context-aware controls.

- **Scoped approvals**: "Allow all `git` commands for this session" instead of per-command approval
- **Time-limited permissions**: auto-revoke after N minutes or when the task completes
- **Permission templates**: save and reuse permission sets for common task types (e.g., "web development" auto-approves npm, git, and localhost browser access)
- **Audit log viewer**: searchable history of all permission decisions with the associated tool calls and outcomes

### 4.2 Interactive Plan Review

Before executing a complex plan, present it to the user as an editable checklist.

- User can reorder steps, remove steps, or add constraints
- User can mark steps as "skip" or "ask me first"
- Plan diffs: when the agent revises a plan mid-task, show what changed and why
- Approval modes: auto-execute (trust the agent), step-by-step (approve each action), review-plan-only (approve the plan, then auto-execute)

### 4.3 Voice Interaction

Extend the existing Android speech recognition to a full voice interaction loop.

- Speech-to-text on both desktop (browser Web Speech API or Whisper) and mobile
- Text-to-speech for agent responses (configurable voice, speed, pitch)
- Wake word detection for hands-free operation ("Hey Pi, ...")
- Voice commands for common actions: "stop", "pause", "approve", "deny", "go back"
- Transcription displayed in the chat interface alongside audio playback controls

### 4.4 Contextual Suggestions

Proactively suggest actions based on the current context without waiting for explicit user requests.

- File change detection: "I noticed you modified `auth.rs` — want me to run the tests?"
- Error pattern detection: "This error usually means the database migration is missing. Should I check?"
- Workflow completion: "You've committed but haven't pushed. Want me to create a PR?"
- Suggestions appear as dismissible cards in the UI, not as interruptions

### 4.5 Collaborative Editing

Real-time collaborative editing where the agent and user can work on the same file simultaneously.

- Agent proposes changes as highlighted diffs in the editor
- User can accept, reject, or modify each change inline
- Conflict resolution when both edit the same region
- Change attribution: color-coded by author (user vs. agent)

### 4.6 Guided Tutorials & Explanations

The agent can generate step-by-step walkthroughs of its actions for educational purposes.

- "Explain mode": the agent narrates what it's doing and why at each step
- Generated tutorials saved as markdown in the knowledge base
- Difficulty-adaptive: adjusts explanation depth based on user's demonstrated expertise
- Code annotations: the agent can add temporary inline comments explaining generated code

---

## 5. Visualization & Observability

### 5.1 Execution Timeline

A horizontal timeline view showing every action the agent has taken during a task.

- Each tool call is a node on the timeline with duration, status (success/failure), and a preview of the result
- Nodes are color-coded by tool type (shell=blue, browser=green, code=yellow, ML=purple)
- Click a node to expand full input/output details
- Filter by tool type, status, or time range
- Parallel tool calls shown as vertically stacked nodes at the same time position

### 5.2 Live Resource Monitor

Real-time dashboard showing system resource usage during agent execution.

- CPU, memory, disk I/O, network usage
- Per-process breakdown (Rust core, Python sidecar, headless Chrome, child processes)
- GPU utilization during ML training (CUDA/ROCm metrics from PyTorch)
- Historical chart with markers for each agent action
- Alerts when resources approach configured thresholds

### 5.3 Memory Visualization

Interactive visualization of the agent's memory state.

- **Embedding space**: 2D projection (UMAP/t-SNE) of stored embeddings, colored by source type (message, tool result, task)
- **Retrieval heatmap**: which memories were retrieved for each planning step
- **Memory timeline**: chronological view of all stored items with search and filter
- **Cluster detection**: automatically identify topic clusters in stored knowledge

### 5.4 Plan Visualization

Render the agent's current plan as an interactive directed graph.

- Nodes represent planned actions; edges represent dependencies
- Completed nodes are checked; in-progress nodes pulse; failed nodes are flagged
- Subtask expansion: click a node to see its child plan
- Critical path highlighting: show which steps are blocking completion

### 5.5 Diff & Change Visualization

When the agent modifies files, show rich diffs in the UI.

- Side-by-side and inline diff views
- Syntax-highlighted diffs for all supported languages
- Cumulative diff: show all changes made during a task as a single unified diff
- Before/after file tree comparison when the agent creates or deletes files

### 5.6 Training Dashboard

Dedicated view for monitoring ML model training in real time.

- Loss curves (training and validation) updating live
- Metric charts (accuracy, F1, BLEU, etc.) per epoch
- Learning rate schedule visualization
- GPU memory and utilization graphs
- Checkpoint timeline with performance annotations
- Early stopping indicator with configurable patience
- Model comparison table across training runs

---

## 6. Platform & Integration

### 6.1 Plugin / Skill System

A standardized plugin architecture allowing third-party tool and skill development.

- Plugin manifest (JSON/TOML) declaring tool name, schema, permission tier, and entry point
- Plugins can be written in Rust (native), Python (sidecar-hosted), or JavaScript (sandboxed)
- Plugin marketplace: discover, install, and update plugins from a central registry
- MCP (Model Context Protocol) compatibility for interoperability with other agent frameworks

### 6.2 Project Profiles

Configurable per-project settings that the agent loads when working in a specific directory.

- `.pi-assistant.toml` in the project root
- Defines: allowed tools, permission overrides, preferred LLM provider, custom tool configurations
- Auto-detected when the agent opens a project directory
- Shareable across teams via version control

### 6.3 iOS Client

Extend mobile support beyond Android with a native iOS client.

- SwiftUI interface matching the Android client's functionality
- Same WebSocket protocol, same state synchronization
- Siri Shortcuts integration for common agent commands
- Widget for home screen status monitoring

### 6.4 Web Client

Browser-based client for accessing Pi-Assistant remotely without installing the desktop app.

- Connects via WebSocket to the Rust core (same protocol as mobile)
- Progressive Web App (PWA) for installability
- Responsive design for desktop and mobile browsers
- End-to-end encryption for remote access over the internet

### 6.5 CI/CD Integration

Allow the agent to participate in CI/CD pipelines as an automated reviewer or fixer.

- GitHub Actions / GitLab CI integration: trigger the agent on PR events
- Automated code review: the agent comments on PRs with suggestions
- Auto-fix mode: the agent can push fix commits for failing CI checks (with approval)
- Pipeline status monitoring: the agent tracks build/test status and notifies the user

### 6.6 Calendar & Task Manager Integration

Connect with external productivity tools so the agent can schedule and track work.

- Google Calendar / CalDAV: schedule tasks, set reminders, block focus time
- Todoist / Linear / Jira: create issues, update status, link to agent tasks
- Bidirectional sync: changes in external tools reflect in the agent's task state

---

## 7. Safety & Reliability

### 7.1 Sandboxed Execution Profiles

Define execution profiles with strict resource and access boundaries.

- Network policies: allowlist/denylist of domains, ports, and protocols
- Filesystem policies: read-only mounts, writable scratch directories, size quotas
- Process policies: max child processes, CPU time limits, memory caps
- Profiles are composable: "web-dev" = "shell-safe" + "browser-localhost" + "npm-install"

### 7.2 Rollback & Undo

Every file modification and shell command is recorded in a transaction log that supports rollback.

- Automatic filesystem snapshots before destructive operations
- "Undo last action" command that reverses the most recent tool execution
- "Undo task" command that reverses all changes from a completed task
- Git-aware rollback: if changes were committed, create a revert commit

### 7.3 Anomaly Detection

Monitor agent behavior for patterns that indicate something has gone wrong.

- Token budget runaway: alert if the agent is consuming tokens much faster than expected
- Loop detection: flag if the agent is repeating the same tool call with the same parameters
- Output size monitoring: alert if a tool produces unexpectedly large output
- Behavioral baseline: learn normal patterns per task type and flag deviations

### 7.4 Audit & Compliance

Comprehensive logging for environments that require auditability.

- Immutable audit log of all agent actions, decisions, and user approvals
- Exportable reports (PDF, JSON) for compliance review
- Data retention policies with configurable auto-purge
- PII detection and redaction in stored memories

---

## 8. Performance & Scalability

### 8.1 Streaming Responses

Stream LLM planner output token-by-token to the UI instead of waiting for the complete response.

- Partial plan display: show reasoning as it's generated
- Early tool dispatch: begin executing the first tool call as soon as it's fully parsed, before the rest of the plan is complete
- Cancellation: user can stop generation mid-stream if the plan is heading in the wrong direction

### 8.2 Embedding Cache & Batch Processing

Optimize the memory system for larger knowledge bases.

- LRU cache for frequently accessed embeddings
- Batch embedding generation: process multiple texts in a single sidecar call
- Background re-indexing: update embeddings when the underlying model changes without blocking the agent
- Quantized embeddings (int8) to reduce storage and speed up similarity search

### 8.3 Model Quantization & Optimization

Reduce resource requirements for local model inference.

- GGUF/GGML quantized model support via llama.cpp integration
- ONNX Runtime for optimized inference on CPU
- Dynamic model loading: load models on demand, unload when idle
- Model distillation: train smaller task-specific models from larger teacher models

### 8.4 Persistent Sidecar Pool

Keep multiple Python sidecar instances warm for parallel tool execution.

- Pool size configurable based on available system resources
- Affinity routing: send training tasks to GPU-equipped sidecars, inference to CPU sidecars
- Health checking and automatic restart of unhealthy sidecars
- Graceful scaling: spin up additional sidecars under load, wind down when idle

### 8.5 Client-Side WebAssembly Inference [IMPLEMENTED]

Enable the agent to execute light-weight models directly in the web browser using the Candle framework compiled to WebAssembly.

- **pi-ai crate**: Native Rust AI logic compiled to Wasm.
- **In-browser execution**: Zero-latency inference for small models (embeddings, lightweight classifiers).
- **Zustand integration**: Model state managed via `agentStore.ts`.
- **useClientAI Hook**: Unified interface for initializing and calling the Wasm model.

---

## 9. Compute Mobility & Device Management (Phase 1 partially complete)

The agent should be able to treat compute hardware as a fluid resource — migrating itself, its models, and its workloads between CPU, GPU, and remote machines as conditions demand.

### 9.1 GPU ↔ CPU Live Migration [IMPLEMENTED]

Allow the agent's inference engine and loaded models to transfer between GPU and CPU at runtime without restarting the sidecar or interrupting the agent loop.

- **Device-aware model registry**: every loaded model tracks its current device (`cpu`, `cuda:0`, `cuda:1`, `mps`, etc.) — _Implemented in `LoadedModel.device` field and `ModelRegistry.list_models()` device info_
- **On-demand migration**: the agent (or the user) issues a `model.to_device` command that calls `model.to(device)` in the Python sidecar, moving tensors and optimizer state between CPU and GPU memory — _Implemented via `ModelRegistry.migrate_model()`, `model.migrate` IPC handler, and `migrate_model` Tauri command_
- **Automatic fallback**: if GPU memory is exhausted during inference or training, the sidecar catches the `OutOfMemoryError`, migrates the model to CPU, retries the operation, and logs the fallback event — _Implemented in `InferenceEngine._complete_local_stream()` OOM catch-and-retry wrapper_
- **Mixed-device operation**: keep the embedding model on CPU (small, fast enough) while a larger generative model occupies the GPU — the agent's device allocator decides placement based on model size, VRAM headroom, and current utilization — _Implemented via `DeviceManager.best_device_for()` placement logic_
- **Hot-swap during idle**: when the agent is idle and a training job finishes, automatically reclaim GPU memory by moving the inference model back from CPU to GPU
- **UI indicator**: the resource monitor (5.2) shows a per-model device badge (CPU/GPU) and a one-click "Move to GPU" / "Move to CPU" button — _Frontend store implemented in `deviceStore.ts` with `migrateModel()` action_

**Implementation sketch (Python sidecar):**

```
┌──────────────────────────────────────────────────────┐
│  DeviceAllocator                                      │
│                                                      │
│  Models:                                             │
│    embedding    → cpu    (384-dim, ~90 MB)           │
│    planner_llm  → cuda:0 (7B params, ~14 GB)        │
│    classifier_v2→ cpu    (fine-tuned, ~440 MB)       │
│                                                      │
│  VRAM budget:   24 GB total, 14.2 GB used            │
│  RAM budget:    32 GB total, 4.1 GB used (models)    │
│                                                      │
│  migrate(model_name, target_device):                 │
│    1. Acquire target device lock                     │
│    2. Check available memory on target               │
│    3. model.to(target_device)                        │
│    4. Update registry                                │
│    5. Emit device_changed event → UI + agent loop    │
│    6. Release source device memory (gc + empty_cache)│
└──────────────────────────────────────────────────────┘
```

### 9.2 Remote Compute Offloading

The agent can dispatch heavy workloads (training, large-model inference, batch embedding) to a remote machine and continue light planning and tool execution locally. After the remote job completes, the agent pulls results back and resumes.

- **Remote sidecar protocol**: extend the existing NDJSON IPC to work over SSH tunnels or a lightweight RPC layer (e.g., ZeroMQ, gRPC) to a remote Python sidecar running on a GPU server
- **Job lifecycle**: `submit → running(progress%) → completed → pull_results` — the agent does not block; it continues other tool calls while the remote job runs
- **Checkpoint streaming**: during remote training, stream checkpoints back to the local machine periodically so progress is not lost if the connection drops
- **Automatic reconnect**: if the SSH tunnel or network connection drops, buffer pending IPC messages locally and replay them when the connection is restored
- **Credential management**: SSH keys and remote host configs stored in the permission-gated keyring, never logged or embedded in memory
- **Cost tracking**: log compute-hours consumed on remote machines, display in the cost dashboard (1.3)

```
Local Machine (CPU)                    Remote Machine (GPU)
┌─────────────────────┐  SSH/gRPC     ┌──────────────────────┐
│ Rust Agent Loop      │──────────────►│ Python Sidecar       │
│  - planning (local)  │  submit job   │  - training          │
│  - shell/code tools  │◄──────────────│  - large inference   │
│  - memory queries    │  stream       │  - batch embedding   │
│                      │  progress     │                      │
│ Python Sidecar (CPU) │◄──────────────│  Return:             │
│  - embeddings        │  pull model/  │  - trained model     │
│  - light inference   │  results      │  - inference results │
└─────────────────────┘               └──────────────────────┘
```

### 9.3 Multi-GPU Orchestration

On machines with multiple GPUs, the agent should be able to distribute work across them intelligently.

- **Per-GPU memory and utilization tracking** via `torch.cuda` and `nvidia-smi` polling
- **Model sharding**: large models that don't fit on a single GPU are split across multiple GPUs using PyTorch's `device_map="auto"` or FSDP (Fully Sharded Data Parallel)
- **Pipeline parallelism for training**: split model layers across GPUs so that forward and backward passes overlap, increasing throughput
- **Data parallelism**: replicate the model on each GPU and split training batches across them using `DistributedDataParallel`
- **GPU affinity rules**: user can pin specific models to specific GPUs (e.g., "always keep the planner on GPU 0, use GPU 1 for training")
- **Dynamic rebalancing**: if one GPU becomes a bottleneck, migrate workloads to balance utilization

### 9.4 Suspend, Serialize, and Resume Agent State

The agent can snapshot its entire execution state — loaded models, memory context, plan progress, tool history — serialize it to disk, and resume later on the same or a different machine.

- **State snapshot** includes: current plan, iteration counter, in-flight tool calls, loaded model weights and device placement, memory retrieval cache, permission state
- **Portable format**: snapshot stored as a directory containing a metadata JSON file, model checkpoints (safetensors), and a SQLite memory export
- **Cross-machine resume**: copy the snapshot to another machine (with compatible hardware), start Pi-Assistant, and issue a `resume_from_snapshot` command — the agent picks up exactly where it left off
- **Use cases**: start a task on a laptop, suspend, transfer to a desktop with a GPU for training, transfer back to the laptop when training is done
- **Integrity verification**: SHA-256 checksums on all snapshot artifacts to detect corruption during transfer

### 9.5 Heterogeneous Device Awareness [IMPLEMENTED]

The agent should discover and adapt to whatever compute is available at runtime.

- **Hardware probe at startup**: detect CPU architecture (x86*64, aarch64), GPU vendor and VRAM (NVIDIA/CUDA, AMD/ROCm, Apple/MPS), available RAM, disk speed — \_Implemented in `DeviceManager.probe()` with `GpuInfo`, `CpuInfo`, `SystemInfo` dataclasses*
- **Capability matrix**: map each detected device to the operations it can accelerate (CUDA → training + inference, MPS → inference only, CPU → everything but slower) — _Implemented in `DeviceManager.get_capabilities()` returning `DeviceCapability` per device_
- **Automatic model selection**: if only 4 GB of VRAM is available, select a quantized 4-bit model instead of a full-precision 7B model; if no GPU is present, default to CPU-optimized ONNX models — _Partially implemented: `best_device_for()` chooses placement; model format selection not yet automated_
- **Runtime adaptation**: if a USB eGPU is connected or disconnected during a session, detect the change and migrate models accordingly
- **User override**: the user can force a specific device via settings even if the agent would prefer a different one — _Implemented via `migrate_model` Tauri command and frontend `deviceStore.migrateModel()` action_

---

## 10. Advanced ML & Deep Learning (Phase 1 partially complete)

### 10.1 Train-Deploy-Use Cycle [IMPLEMENTED]

A complete lifecycle where the agent trains a model, evaluates it, deploys it as a callable tool, and uses it in future tasks — all within the same session or across sessions.

```
User Request                Agent Actions
────────────                ─────────────
"Classify these             1. Inspect dataset (code tool)
 support tickets            2. Select architecture (planner)
 by priority"               3. Train classifier (training tool, GPU)
        │                   4. Evaluate on held-out set
        │                   5. Register as tool: classify_ticket(text) → priority
        │                   6. Migrate model to CPU (free GPU for next task)
        ▼                   7. Apply to remaining tickets using new tool
"Here are 500               8. Return results to user
 more tickets"              9. Use classify_ticket tool directly (no retraining)
```

- Trained models persist across sessions via the model registry — _Implemented: `TrainingService.deploy()` loads checkpoint, registers in `ModelRegistry`; `RunInfo` tracks deployment metadata persisted to `runs.json`_
- The agent can decide to retrain when it detects distribution shift (accuracy drop on new data)
- Old model versions are kept for comparison; the agent can A/B test predictions

**Implementation details:**

- `TrainingService.deploy(run_id, tool_name, device)` rebuilds model from config, loads weights, registers as `LoadedModel`
- `TrainingService.predict(tool_name, input_data)` runs inference with task-type-aware post-processing (classification → softmax, regression → raw output)
- `DeployedModelTool` (Rust) wraps each deployed model as a callable `Tool` in the `ToolRegistry`
- `TrainingTool` extended with `deploy`, `predict`, `list_deployed` actions; auto-registers `DeployedModelTool` on deploy
- Deadlock prevention: tool `Arc` cloned before execution in agent loop; sidecar lock dropped before registry write-lock acquired during deploy

### 10.2 Reinforcement Learning for Agent Self-Optimization

Use RL (via TorchRL or Stable Baselines3) to train policies that improve the agent's own decision-making over time.

- **Action space**: which tool to call, what parameters to use, when to ask the user vs. proceed autonomously
- **Reward signal**: task completion (binary), number of iterations to complete (lower is better), user satisfaction (thumbs up/down), permission denials (penalty)
- **Offline RL**: train on logged trajectories from past sessions without live exploration — safe, no risk of destructive actions during training
- **Policy deployment**: the trained policy acts as an advisor to the LLM planner, biasing tool selection toward historically successful sequences
- **Continuous improvement**: retrain the policy periodically as more session data accumulates
- **Guardrails**: the RL policy can only suggest actions; the LLM planner and permission engine remain the final decision-makers

### 10.3 Neural Architecture Search (NAS) for Task-Specific Models

When the agent trains a task-specific model (10.1), it can use NAS to automatically discover the best architecture rather than relying on a fixed template.

- Search space: layer count, hidden dimensions, activation functions, attention heads, dropout rates
- Search strategy: Bayesian optimization (Optuna) or evolutionary search within a time/compute budget
- Constraint-aware: the user specifies max model size, max inference latency, or target device — NAS respects these constraints
- Results stored in the model registry with architecture metadata for reproducibility

### 10.4 Continual & Incremental Learning

Models trained by the agent should be updatable with new data without catastrophic forgetting.

- **Elastic Weight Consolidation (EWC)**: penalize changes to weights that were important for previous tasks
- **Replay buffer**: store a small subset of old training data and mix it into new training batches
- **Adapter layers (LoRA/QLoRA)**: instead of full fine-tuning, train lightweight adapters that can be stacked — one per data batch or domain
- **Forgetting detection**: after incremental training, evaluate on a held-out set from each previous data batch and alert if performance degrades

### 10.5 Transfer Learning & Domain Adaptation

The agent should be able to take a model trained on one task and adapt it for a related task with minimal data.

- **Feature extraction**: freeze base model weights, train only a new classification head
- **Gradual unfreezing**: unfreeze layers one at a time from the top, training briefly at each step
- **Domain-adversarial training**: for cases where source and target domains differ significantly (e.g., formal text → colloquial text)
- **Few-shot adaptation**: given only 5-20 labeled examples, fine-tune with aggressive regularization and data augmentation
- **Zero-shot via prompting**: for generative models, the agent can attempt the task with prompt engineering before deciding whether fine-tuning is needed

### 10.6 Distributed Training Across Machines

Extend training beyond a single machine by coordinating multiple remote sidecars.

- **Data-parallel training**: each machine holds a full model replica and processes a portion of the data; gradients are synchronized via `torch.distributed` (NCCL backend for GPU, Gloo for CPU)
- **Launcher**: the agent provisions remote sidecars (9.2), distributes the dataset, and starts the distributed training job
- **Fault tolerance**: if a node drops, the remaining nodes continue from the last checkpoint; the agent re-provisions the lost node and adds it back
- **Monitoring**: the training dashboard (5.6) shows per-node metrics, communication overhead, and gradient synchronization times

### 10.7 Model Interpretability & Explainability

After training a model, the agent should be able to explain what the model learned and why it makes specific predictions.

- **Feature importance**: SHAP values, integrated gradients, or attention weight visualization
- **Confusion matrix and error analysis**: identify systematic misclassifications and suggest data improvements
- **Counterfactual explanations**: "If the input had said X instead of Y, the prediction would change from A to B"
- **Model cards**: the agent generates a markdown model card (stored in the knowledge base) documenting the model's purpose, training data, performance metrics, known limitations, and fairness considerations

### 10.8 Synthetic Data Generation

When real training data is scarce, the agent can generate synthetic data to augment it.

- **Text augmentation**: paraphrasing, back-translation, entity substitution, synonym replacement
- **LLM-generated examples**: use the planner LLM (or a dedicated generative model) to produce labeled examples matching the target distribution
- **Tabular data synthesis**: SMOTE for class imbalance, Gaussian copulas for preserving statistical properties
- **Quality filtering**: the agent evaluates generated samples against a discriminator or heuristic and discards low-quality ones before training
- **Human-in-the-loop**: present borderline synthetic examples to the user for validation before including them in the training set

---

## 11. Autonomous Self-Improvement & Meta-Learning

### 11.1 Tool Proficiency Tracking

The agent maintains a profile of how effectively it uses each tool and improves weak areas.

- **Success rate per tool**: percentage of tool calls that produced the expected result
- **Common failure modes**: log recurring errors per tool (e.g., shell timeouts, permission denials, malformed browser selectors)
- **Self-coaching**: before calling a tool the agent has historically struggled with, it retrieves past failures and their resolutions from memory to avoid repeating mistakes
- **Skill decay detection**: if a tool's success rate drops (e.g., a website changed its DOM structure, breaking browser selectors), the agent flags it and attempts to re-learn the correct approach

### 11.2 Meta-Learning Across Tasks

Learn generalizable strategies from past tasks that transfer to new, unseen tasks.

- **Task embeddings**: embed each completed task description and its successful tool trajectory into a shared space
- **k-nearest task retrieval**: when a new task arrives, find the k most similar past tasks and use their trajectories as demonstrations for the planner
- **Strategy distillation**: periodically distill successful patterns into a set of heuristic rules (stored as markdown in the knowledge base) that the planner consults
- **Curriculum learning**: order tasks by difficulty based on historical iteration counts; use easier tasks to bootstrap strategies for harder ones

### 11.3 Autonomous Skill Acquisition

The agent can identify gaps in its capabilities and proactively acquire new skills.

- **Gap detection**: if the agent repeatedly fails at a task type or frequently asks the user for help on the same topic, it logs a "skill gap"
- **Self-study**: the agent uses the browser and RAG tools to research solutions — reading documentation, tutorials, and Stack Overflow — and stores findings in its knowledge base
- **Practice runs**: the agent can create sandboxed practice tasks (in a container) to test new approaches without affecting the user's system
- **Skill certification**: after practicing, the agent evaluates its performance on a held-out set of similar tasks; if it passes a threshold, it marks the skill as acquired

### 11.4 Feedback-Driven Calibration

Use explicit and implicit user feedback to calibrate the agent's confidence and behavior.

- **Explicit feedback**: thumbs up/down on agent responses, tool results, and plan quality
- **Implicit feedback**: user edits to agent-generated code (the diff is the feedback), user manually re-doing a step the agent already did (indicates the agent's version was wrong)
- **Confidence calibration**: adjust the agent's willingness to act autonomously vs. ask the user based on historical accuracy — high accuracy on a task type → less asking; low accuracy → more asking
- **Preference learning**: track which code styles, tool patterns, and communication styles the user prefers; encode these as soft constraints in the planner prompt

---

## 12. Multi-Modal Perception & Generation

### 12.1 Vision Understanding

The agent can process and reason about images, screenshots, and visual content.

- **Screenshot analysis**: the agent takes a screenshot (via the browser tool or desktop capture) and uses a vision model to describe what it sees, identify UI elements, or detect errors
- **Image-to-code**: given a mockup or wireframe image, the agent generates corresponding HTML/CSS/React code
- **OCR**: extract text from images, PDFs, and scanned documents
- **Visual diff**: compare two screenshots and describe what changed (useful for UI testing)
- **Model support**: CLIP for image-text similarity, LLaVA or similar for visual question answering, running in the Python sidecar

### 12.2 Audio Processing

The agent can process and generate audio beyond simple speech recognition.

- **Audio transcription**: Whisper-based transcription of audio files (meetings, lectures, podcasts) stored as searchable text in memory
- **Speaker diarization**: identify who said what in multi-speaker audio
- **Sound event detection**: classify audio events (alarm, doorbell, crash) for ambient monitoring use cases
- **Text-to-speech generation**: generate audio files from text using a TTS model (Bark, VITS) for creating voice notes or accessibility features
- **Music and audio analysis**: extract tempo, key, and structure from audio files for creative projects

### 12.3 Document Understanding

Deep processing of structured and semi-structured documents.

- **PDF parsing**: extract text, tables, images, and form fields from PDFs using a combination of rule-based extraction and vision models
- **Table extraction**: identify and extract tabular data from PDFs, images, and web pages into structured formats (CSV, JSON, dataframes)
- **Chart reading**: given a chart image, extract the underlying data points
- **Multi-page reasoning**: answer questions that require synthesizing information across multiple pages of a document
- **Citation tracking**: when the agent references a document, link to the specific page and paragraph

### 12.4 Code-Aware Vision

Specialized visual understanding for software development contexts.

- **UI component recognition**: identify buttons, forms, navigation elements, and layout patterns in screenshots
- **Accessibility audit**: analyze screenshots for contrast ratios, missing labels, and touch target sizes
- **Design system compliance**: compare a rendered UI against a design system specification and flag deviations
- **Error screenshot diagnosis**: given a screenshot of an error state, correlate it with code and logs to diagnose the issue

---

## 13. Environment Awareness & Adaptation

### 13.1 System Health Monitoring

The agent continuously monitors the host system and adapts its behavior accordingly.

- **Thermal awareness**: on laptops and SBCs (Raspberry Pi, Jetson), read thermal sensors and throttle agent activity (reduce batch sizes, pause training, migrate to CPU) before thermal throttling kicks in
- **Battery awareness**: on battery-powered devices, reduce GPU usage and defer non-urgent tasks until plugged in
- **Disk space monitoring**: before writing large files (model checkpoints, datasets), check available disk space and warn the user if space is low
- **Memory pressure response**: if system RAM is under pressure, proactively unload idle models and reduce memory retrieval batch sizes

### 13.2 Network-Aware Behavior

Adapt to changing network conditions.

- **Bandwidth detection**: measure available bandwidth before starting large transfers (model downloads, dataset uploads, remote sidecar communication)
- **Offline mode**: if the network is unavailable, fall back to local-only models and tools; queue API calls and remote operations for when connectivity returns
- **Metered connection awareness**: on mobile hotspots or metered connections, avoid large downloads and prefer local models
- **Latency-adaptive provider selection**: if a cloud LLM provider is responding slowly, switch to a local model or a different provider

### 13.3 Peripheral & Sensor Integration

Interact with hardware peripherals and sensors connected to the host machine.

- **Camera access**: capture photos or video via connected cameras for vision tasks (12.1)
- **Microphone access**: continuous listening for wake word detection (4.3) and ambient sound monitoring (12.2)
- **GPIO (on SBCs)**: read sensors and control actuators on Raspberry Pi or Jetson — temperature, humidity, relays, LEDs — enabling IoT and robotics use cases
- **USB device detection**: when a new device is connected (e.g., a USB drive, Arduino, eGPU), the agent detects it and offers relevant actions ("I see you connected an Arduino. Want me to set up the development environment?")
- **Bluetooth**: discover and interact with BLE devices for IoT monitoring and control

### 13.4 Context-Aware Scheduling

The agent decides when to perform tasks based on system and environmental context.

- **Low-priority background tasks**: training jobs, re-indexing, and batch processing scheduled for when the user is idle (no keyboard/mouse input for N minutes)
- **Power-aware scheduling**: defer GPU-heavy tasks until the device is plugged in
- **Time-of-day preferences**: learn the user's work schedule and avoid notifications or heavy processing during off-hours
- **Resource-aware queuing**: if a task requires more resources than currently available (e.g., GPU is occupied by training), queue it and start automatically when resources free up

### 13.5 Edge Deployment & Embedded Targets

Deploy trained models to resource-constrained edge devices.

- **Model export**: convert trained PyTorch models to ONNX, TensorFlow Lite, or Core ML format for deployment on edge devices, mobile phones, or microcontrollers
- **Quantization for edge**: apply post-training quantization (INT8, INT4) or quantization-aware training to minimize model size and inference latency on target hardware
- **Benchmark on target**: if the target device is reachable (e.g., a Raspberry Pi on the local network), the agent can deploy the model, run inference benchmarks, and report latency and accuracy
- **Over-the-air updates**: push updated models to deployed edge devices when retraining produces a better version
- **Model pruning**: remove unnecessary parameters (structured or unstructured pruning) to meet memory and compute constraints of the target device
