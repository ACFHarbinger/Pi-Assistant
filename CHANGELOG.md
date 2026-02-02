# Changelog

All notable changes to Pi-Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Rollback & Undo System** — added safety layer for reversing agent actions
  - `TransactionManager` tracks all file modifications and shell commands in a transaction log
  - `undo` command reverses the most recent tool execution (restores files, cleans up)
  - Atomic file writes with automatic backups (`.bak` files)
  - `CodeTool` integrated with `TransactionManager` for safe writes
- **API Tool Enhancements** — Added OpenAPI Ingestion support to `ApiTool`.
  - Parses JSON/YAML OpenAPI specs and extracts endpoints/schemas.
  - Allows the agent to learn API structures dynamically.
- **Tool System Refactor** — Refactored the `Tool` trait and implementation.
  - Standardized `execute` signature to include `ToolContext`.
  - Enables tools to access shared resources like `TransactionManager`.
- **Rich Permission Workflows (Backend)** — Implemented backend support for granular and time-limited permissions.
  - Updated `PermissionEngine` to support wildcard patterns (e.g., `git *`) and expiration.
  - Added `expires_at` column to `permission_cache` table for persistent, time-limited overrides.
  - Updated `AgentLoop` to persist user overrides to the database.
- **Structured Knowledge Graph** — Implemented backend storage and tools for entity-relationship data.
  - Added `entities` and `relations` tables to SQLite schema with `MemoryManager` methods.
  - created `KnowledgeGraphTool` for the agent to `upsert_entity`, `add_relation`, and `find_related`.
- **Database Tool** — Implemented direct SQL access tool with safety features.
  - Actions: `connect`, `query`, `schema`, `explain`, `list_tables`, `disconnect`.
  - Read-only by default (`SQLITE_OPEN_READ_ONLY`) unless explicitly initiated with `readonly: false`.
  - Introspection support via `PRAGMA table_info` and `sqlite_master`.
- **Heterogeneous Device Awareness** — `DeviceManager` module probes CPU, GPU (CUDA/MPS), and RAM at startup; exposes hardware info to the LLM planner so it can make device-aware decisions
  - `device.info` and `device.refresh` IPC handlers
  - `get_device_info`, `refresh_device_memory` Tauri commands
  - `deviceStore.ts` Zustand store for frontend device info
  - Device capabilities injected into agent planner context (`loop.rs`)
- **System Health Monitoring** — `SystemTool` for real-time host metrics.
  - Tracking CPU usage, RAM, Swap, Uptime, and Load Average.
  - Process listing with resource usage sorting.
  - Implements backend for Roadmap 13.1 and 5.2.
- **Live Resource Monitor (Frontend)** — real-time dashboard for CPU, RAM, and GPU utilization (Roadmap 5.2).
  - Periodic 2-second polling via `spawn_resource_monitor` in Rust core.
  - GPU metrics bridged from ML sidecar `device.refresh`.
  - `resourceStore.ts` with 30-snapshot rolling history.
  - `ResourceMonitor.tsx` with sparkline charts.
- **Domain Adaptation Support** — implemented utilities for distribution shift and transfer learning
  - `MMDLoss` for Maximum Mean Discrepancy based alignment
  - `GradientReversalLayer` and `DomainDiscriminator` for Adversarial Training (DANN)
  - Integration into `PiLightningModule` and `TrainingOrchestrator`
- **Continual Learning Utilities** — added strategies to prevent catastrophic forgetting
  - `EWCCallback` for Elastic Weight Consolidation
  - `ReplayBuffer` and `ReplayDataset` for Experience Replay
- **GPU ↔ CPU Live Migration** — models can be moved between CPU and GPU at runtime without restarting the sidecar
  - `ModelRegistry.migrate_model()` handles both transformers (`.to(device)`) and llama.cpp (reload with different `n_gpu_layers`)
  - `model.migrate` IPC handler and `migrate_model` Tauri command
  - OOM auto-fallback in `InferenceEngine`: catches CUDA out-of-memory, auto-migrates model to CPU, retries inference
  - `LoadedModel` extended with `device` and `model_size_mb` tracking fields
  - Device-aware model placement via `DeviceManager.best_device_for()`
- **Client-Side AI** — Integrated Candle ML framework compiled to WebAssembly for in-browser inference.
- **RAG Tool** — Added support for document ingestion and vector similarity search via `rag` tool (Roadmap 2.2).
- **Episodic Memory** — Automatic summary generation on task completion for long-session context management (Roadmap 3.2).
- **Markdown-Based Knowledge Files** — Added a human-readable knowledge base in markdown format with automatic RAG indexing (Roadmap 3.3).
- **Cross-Session Retrieval** — Extended `RagTool` and `MemoryManager` to support querying multiple sessions (local + global) simultaneously.
- **Hierarchical Task Decomposition** — Added a hierarchical planner and subtask management system with UI visualization.
- **Loop Detection (Self-Correction)** — prevents infinite loops by identifying repetitive identical tool calls (Roadmap 1.2).
  - Tracks hash of `(tool_name, parameters)` across iterations.
  - Pauses execution and requests user intervention after 3 repeats.
- **Containerized Execution Sandbox** — support for running shell commands in isolated Docker containers (Roadmap 2.6).
  - Added `sandbox` and `image` parameters to `ShellTool`.
  - Automatic host directory mapping and user UID/GID preservation (on Linux).
- **Adaptive Context Pruning** — token-aware context management for long conversations (Roadmap 1.3/1.5).
  - Character-based token estimation and sliding window pruning in `MemoryManager`.
  - Ensures planning stability by keeping context within model limits.
- **Reflection & Self-Correction Loop** — Implemented error budgeting, reflection in planning, and UI indicators for agent self-evaluation.
- **Train-Deploy-Use Cycle** — trained models can be deployed as callable tools the agent uses in future iterations
  - `TrainingService.deploy()` loads checkpoint, rebuilds model, registers as `LoadedModel` in the registry
  - `TrainingService.predict()` runs inference with task-type-aware post-processing (classification/regression)
  - `TrainingService.list_deployed()` lists all deployed model tools
  - `RunInfo` extended with deployment metadata (`tool_name`, `deployed`, `deploy_device`, `task_type`)
  - `training.deploy`, `training.predict`, `training.list_deployed` IPC handlers
  - `DeployedModelTool` (Rust) — dynamic `Tool` wrapper for each deployed model, auto-registered in `ToolRegistry`
  - `TrainingTool` extended with `deploy`, `predict`, `list_deployed` actions
  - GPU memory cleanup after training completes (`gc.collect()` + `torch.cuda.empty_cache()`)
- **Cost-Aware Planning** — Track and limit token usage per session.
  - `CostManager` tracks prompt/completion tokens and estimates costs.
  - User-configurable limits for Max Tokens and Max Cost in Settings.
  - `CostDashboard` component for real-time monitoring.
  - Backend integration in `AgentCommand::Start` to enforce budgets.
- **Multi-Agent Support** — Refactored backend and frontend to support multiple concurrent agents.
  - Backend `AgentCoordinator` manages multiple `AgentLoop` instances.
  - `agent_id` tracking in `AgentState` and `AgentCommand`.
  - Frontend `AgentStore` tracks collection of agents.
  - `AgentSelector` UI component for switching agent contexts.
- **Training Dashboard** — Initial UI for managing ML training runs (Roadmap 5.6).
  - List historical and active training runs.
  - Start new training runs with JSON config.
  - Stop running jobs.
  - Deploy trained models to `DeviceManager`.
- Project architecture design and documentation
- File structure for Rust (Tauri v2), React/TypeScript, Python sidecar, and Android client
- Agent loop design with state machine (Idle/Running/Paused/Stopped)
- Tool system design: Shell, Browser, Code, Python Inference
- NDJSON-over-stdio IPC protocol between Rust core and Python sidecar
- Memory system design: SQLite for structured data, sqlite-vec for vector search
- Three-tier permission engine (Auto-Approve / Ask User / Block)
- WebSocket server design for Android mobile client
- Android client architecture with OkHttp, Kotlin Flow, Jetpack Compose
- Human-in-the-loop support (pause/ask/resume cycle)
- PyTorch Lightning module skeleton for model fine-tuning
- Comprehensive documentation suite

### Fixed

- **Agent loop deadlock prevention** — tool `Arc` is now cloned and the `RwLock` read guard dropped before async tool execution, preventing deadlock when `TrainingTool.deploy` needs a write lock on the `ToolRegistry` to register a new `DeployedModelTool`
- **ToolRegistry self-reference** — `TrainingTool` is now registered separately in `state.rs` after the `ToolRegistry` is wrapped in `Arc<RwLock>`, giving it write access for auto-registering deployed models

### Architecture Decisions

- NDJSON over stdin/stdout chosen for Rust<->Python IPC (cross-platform, simple, auto-cleanup)
- sqlite-vec chosen over Qdrant for vector search (zero deployment overhead)
- Axum chosen for WebSocket server (Tokio-native, lightweight)
- chromiumoxide chosen for headless browser (async, CDP, Tokio-native)
- Zustand chosen for frontend state management (lightweight, TypeScript-first)
- llama.cpp migration via reload (not `.to()`) because GGUF models don't support PyTorch device transfer

---

## [0.1.0] - Unreleased

Initial release. Target feature set:

- [ ] Tauri v2 desktop application shell
- [ ] React frontend with agent status, chat, and permission dialogs
- [ ] Rust agent loop with continuous mode
- [ ] Shell, Code, and Browser tools
- [ ] Python sidecar with NDJSON IPC
- [ ] Embedding generation (all-MiniLM-L6-v2)
- [ ] SQLite + sqlite-vec memory system
- [ ] Three-tier permission engine
- [ ] WebSocket server for mobile
- [ ] Android client with chat and voice input
- [ ] PyTorch Lightning training pipeline

---

<!-- Links -->

[Unreleased]: https://github.com/ACFHarbinger/Pi-Assistant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ACFHarbinger/Pi-Assistant/releases/tag/v0.1.0
