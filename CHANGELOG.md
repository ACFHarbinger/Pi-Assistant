# Changelog

All notable changes to Pi-Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
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

### Architecture Decisions
- NDJSON over stdin/stdout chosen for Rust<->Python IPC (cross-platform, simple, auto-cleanup)
- sqlite-vec chosen over Qdrant for vector search (zero deployment overhead)
- Axum chosen for WebSocket server (Tokio-native, lightweight)
- chromiumoxide chosen for headless browser (async, CDP, Tokio-native)
- Zustand chosen for frontend state management (lightweight, TypeScript-first)

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
