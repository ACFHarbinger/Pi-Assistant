# Pi-Assistant

**Languages**

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Kotlin](https://img.shields.io/badge/kotlin-%237F52FF.svg?style=for-the-badge&logo=kotlin&logoColor=white)](https://kotlinlang.org/)

**Frameworks**

[![Tauri](https://img.shields.io/badge/tauri-%2324C8DB.svg?style=for-the-badge&logo=tauri&logoColor=%23262626)](https://tauri.app/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://react.dev/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch_lightning-792ee5.svg?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![Axum](https://img.shields.io/badge/Axum-%23000000.svg?style=for-the-badge&logo=Axum&logoColor=white)](https://github.com/tokio-rs/axum)
[![Tokio](https://img.shields.io/badge/Tokio-%23000000.svg?style=for-the-badge&logo=Tokio&logoColor=white)](https://tokio.rs/)

**AI & ML**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![sentence-transformers](https://img.shields.io/badge/sentence_transformers-UKPLab-orange?style=for-the-badge)](https://www.sbert.net/)
[![sqlite-vec](https://img.shields.io/badge/sqlite_vec-000000?style=for-the-badge&logo=sqlite&logoColor=white)](https://github.com/asg017/sqlite-vec)

**Frontend**

[![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Zustand](https://img.shields.io/badge/zustand-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://github.com/pmndrs/zustand)

**Database**

[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

**Tools**

[![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg?style=for-the-badge)](https://github.com/astral-sh/uv)
[![Gradle](https://img.shields.io/badge/Gradle-02303A.svg?style=for-the-badge&logo=Gradle&logoColor=white)](https://gradle.org/)
[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com/)

**Hardware**

[![CUDA RTX 4080](https://img.shields.io/badge/CUDA-RTX_4080-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CUDA RTX 3090ti](https://img.shields.io/badge/CUDA-RTX_3090ti-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Intel](https://img.shields.io/badge/Intel-0071C5?style=for-the-badge&logo=intel&logoColor=white)](https://www.intel.com/)

**Operating System**

[![Kubuntu](https://img.shields.io/badge/Kubuntu-0079C1?style=for-the-badge&logo=kubuntu&logoColor=white)](https://kubuntu.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com/)

A Universal Agent Harness — a desktop application that hosts an autonomous AI agent capable of controlling the computer, maintaining persistent memory, and training sub-models.

Built with **Rust** (Tauri v2), **TypeScript** (React), **Python** (PyTorch Lightning), and **Kotlin** (Android).

---

## Overview

Pi-Assistant is a multi-runtime system where a Rust core orchestrates an AI agent that can:

- **Execute shell commands** with a three-tier permission safety layer
- **Browse the web** via a headless Chrome instance
- **Write and modify code** with path-based access controls
- **Remember context** across sessions using SQLite + vector similarity search
- **Train and fine-tune models** through a Python sidecar running PyTorch Lightning
- **Accept commands from mobile** via a Kotlin/Android app over WebSockets

```
┌──────────────┐    Tauri IPC    ┌─────────────────────────────┐
│ React / TS   │◄───────────────►│        Rust Core            │
│ Desktop UI   │                 │                             │
└──────────────┘                 │  Agent Loop → Tools         │
                                 │      ↓           ↓          │
┌──────────────┐   WebSocket     │  Permission   Memory        │
│ Kotlin       │◄───────────────►│  Engine       (SQLite +     │
│ Android App  │   :9120/ws      │               sqlite-vec)   │
└──────────────┘                 │      ↓                      │
                                 │  IPC Bridge (NDJSON/stdio)  │
                                 └──────────┬──────────────────┘
                                            │
                                 ┌──────────▼──────────────────┐
                                 │  Python Sidecar              │
                                 │  PyTorch Lightning            │
                                 │  sentence-transformers        │
                                 └──────────────────────────────┘
```

## Features

### Agent Loop (Continuous Mode)
The agent iterates autonomously on a task until completion, a manual stop, or an iteration limit. Each iteration: retrieve context from memory, plan the next step via LLM, check permissions, execute tools, store results.

### Tool System
| Tool | Description |
|------|-------------|
| **Shell** | Execute commands with safety-tier gating |
| **Browser** | Headless Chrome via Chrome DevTools Protocol |
| **Code** | Read/write/patch files within allowed directories |
| **ML Inference** | Delegate to Python sidecar for embeddings and completions |

### Human-in-the-Loop
The agent can pause and ask the user a question via the desktop UI or the mobile app. Permission requests for medium-risk operations are surfaced as approval dialogs on both interfaces.

### Persistent Memory
Dual-layer memory system:
- **SQLite** for structured data (sessions, messages, task logs, tool executions)
- **sqlite-vec** for semantic vector search over embeddings (384-dim, all-MiniLM-L6-v2)

### Mobile Remote
An Android app connects over the local network via WebSockets, providing:
- Real-time agent status monitoring
- Chat interface for sending commands
- Permission approval/denial
- Voice input via Android SpeechRecognizer

### Safety Layer
Three-tier permission engine:
- **Auto-Approve**: Read-only operations (`ls`, `cat`, `git status`, etc.)
- **Ask User**: Mutations (`git commit`, `npm install`, file writes)
- **Block**: Destructive operations (`sudo`, `rm -rf /`, `dd`, credential exposure)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical details.

## Project Structure

```
Pi-Assistant/
├── src-tauri/          # Rust core (Tauri v2 backend)
├── src/                # React/TypeScript frontend
├── sidecar/            # Python ML sidecar (PyTorch Lightning)
├── android/            # Kotlin/Android mobile client
├── protocol/           # Shared protocol schemas (JSON Schema)
├── crates/             # Additional Rust workspace crates
└── docs/               # Extended documentation
```

## Quick Start

### Prerequisites

- **Rust** 1.75+ (with `cargo`)
- **Node.js** 20+ (with `npm`)
- **Python** 3.11+ (with `uv` recommended, or `pip`)
- **Android Studio** (for mobile development only)
- **System libraries** for Tauri: see [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/)

### Build & Run

```bash
# 1. Clone the repository
git clone https://github.com/ACFHarbinger/Pi-Assistant.git
cd Pi-Assistant

# 2. Install frontend dependencies
npm install

# 3. Set up Python sidecar
cd sidecar
uv sync        # or: pip install -e ".[dev]"
cd ..

# 4. Run in development mode
npm run tauri dev
```

The desktop app will launch with the React frontend. The Python sidecar starts automatically. The WebSocket server listens on port `9120` for mobile connections.

### Android Client

```bash
cd android
./gradlew installDebug
```

Configure the server IP in the Android app's settings screen to connect.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flows, code snippets |
| [AGENTS.md](AGENTS.md) | Agent loop internals, tool system, planning |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Developer setup and build instructions |
| [DEPENDENCIES.md](DEPENDENCIES.md) | All dependencies across all runtimes |
| [TESTING.md](TESTING.md) | Testing strategy and how to run tests |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [TUTORIAL.md](TUTORIAL.md) | End-to-end getting started tutorial |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [docs/IPC-PROTOCOL.md](docs/IPC-PROTOCOL.md) | IPC wire protocol specification |
| [docs/SAFETY-MODEL.md](docs/SAFETY-MODEL.md) | Permission tiers and sandboxing model |

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
