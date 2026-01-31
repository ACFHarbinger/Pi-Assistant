# Pi-Assistant Python Sidecar 🧠

The Python sidecar is the "cognitive engine" of Pi-Assistant. It runs as a child process of the main Tauri application and handles all ML/AI operations that are better suited for Python's ecosystem, including:

- **LLM Inference**: Unified interface for OpenAI, Anthropic, Gemini, and local models.
- **Agent Planning**: Logic for breaking down user requests into tool calls.
- **Personality Engine**: Loading and applying the "soul" of the assistant (from `soul.md`).
- **Embeddings**: Generating semantic vector embeddings for the Rust memory system.

## 🏗 Architecture

The sidecar operates as a standalone Python process spawned by the Rust backend.

- **Communication**: Standard Input/Output (stdio) using newline-delimited JSON (NDJSON).
- **Protocol**: JSON-RPC style request/response with correlation IDs.
- **Lifecycle**: Managed by the Rust `SidecarHandle`. Auto-restarts on crash.

### Directory Structure

```
sidecar/
├── src/pi_sidecar/
│   ├── inference/       # LLM provider implementations (Anthropic, Gemini, Local)
│   ├── ipc/             # NDJSON transport and request handlers
│   ├── personality.py   # Personality loading and system prompt generation
│   └── __main__.py      # Entry point
├── tests/               # Pytest suite
├── pyproject.toml       # Build configuration (hatchling + uv)
└── uv.lock              # Locked dependencies
```

## 🛠 Setup & Development

We use **[uv](https://github.com/astral-sh/uv)** for fast, reliable dependency management.

### Prerequisites

- Python 3.11+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
cd sidecar
uv sync
```

### Running Locally

You can run the sidecar manually for testing (though typically the Rust app spawns it):

```bash
# Activate virtual environment
source .venv/bin/activate

# Run module
python3 -m pi_sidecar
```

> **Note**: When running manually, the sidecar expects NDJSON input on stdin.

### Running Tests

```bash
uv run pytest
```

## 🔌 API Methods

The sidecar exposes several JSON-RPC methods to the Rust backend:

| Method | Description |
|--------|-------------|
| `health.ping` | Simple connectivity check. |
| `inference.plan` | Generates an execution plan based on user capability + tools. |
| `inference.complete` | Raw text completion/chat. |
| `inference.embed` | Generate vector embeddings for text chunks. |
| `personality.get_hatching` | Retrieve the initial "hatching" welcome message. |
| `personality.get_prompt` | Get the full personality-infused system prompt. |
| `lifecycle.shutdown` | Gracefully terminate the process. |

## 🧩 Key Components

### 1. Personality Engine (`personality.py`)
Loads the `soul.md` file from the repository root to configure the agent's identity, tone, and constraints. This ensures Pi acts consistently across sessions.

### 2. Inference Engine (`inference/engine.py`)
Abstracts away differences between model providers.
- Supports **Anthropic** (Claude), **Google** (Gemini), and **Local** (via OpenAI compatible endpoints).
- Handles context window management and prompt formatting.

### 3. IPC Transport (`ipc/ndjson_transport.py`)
Reads line-based JSON from stdin and writes responses to stdout. This "share-nothing" architecture simplifies concurrency—Rust manages the process, Python just handles the brain.
