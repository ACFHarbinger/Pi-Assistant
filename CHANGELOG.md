# Changelog

All notable changes to Pi-Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Heterogeneous Device Awareness** — `DeviceManager` module probes CPU, GPU (CUDA/MPS), and RAM at startup; exposes hardware info to the LLM planner so it can make device-aware decisions
  - `device.info` and `device.refresh` IPC handlers
  - `get_device_info`, `refresh_device_memory` Tauri commands
  - `deviceStore.ts` Zustand store for frontend device info
  - Device capabilities injected into agent planner context (`loop.rs`)
- **GPU ↔ CPU Live Migration** — models can be moved between CPU and GPU at runtime without restarting the sidecar
  - `ModelRegistry.migrate_model()` handles both transformers (`.to(device)`) and llama.cpp (reload with different `n_gpu_layers`)
  - `model.migrate` IPC handler and `migrate_model` Tauri command
  - OOM auto-fallback in `InferenceEngine`: catches CUDA out-of-memory, auto-migrates model to CPU, retries inference
  - `LoadedModel` extended with `device` and `model_size_mb` tracking fields
  - Device-aware model placement via `DeviceManager.best_device_for()`
- **Client-Side AI** — Integrated Candle ML framework compiled to WebAssembly for in-browser inference.
- **Hierarchical Task Decomposition** — Added a hierarchical planner and subtask management system with UI visualization.
- **Train-Deploy-Use Cycle** — trained models can be deployed as callable tools the agent uses in future iterations
  - `TrainingService.deploy()` loads checkpoint, rebuilds model, registers as `LoadedModel` in the registry
  - `TrainingService.predict()` runs inference with task-type-aware post-processing (classification/regression)
  - `TrainingService.list_deployed()` lists all deployed model tools
  - `RunInfo` extended with deployment metadata (`tool_name`, `deployed`, `deploy_device`, `task_type`)
  - `training.deploy`, `training.predict`, `training.list_deployed` IPC handlers
  - `DeployedModelTool` (Rust) — dynamic `Tool` wrapper for each deployed model, auto-registered in `ToolRegistry`
  - `TrainingTool` extended with `deploy`, `predict`, `list_deployed` actions
  - GPU memory cleanup after training completes (`gc.collect()` + `torch.cuda.empty_cache()`)
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
