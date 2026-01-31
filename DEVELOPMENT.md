# Development Guide

How to set up your development environment, build each component, and run Pi-Assistant locally.

---

## Prerequisites

### Required

| Tool            | Version | Install                                                         |
| --------------- | ------- | --------------------------------------------------------------- |
| **Rust**        | 1.75+   | [rustup.rs](https://rustup.rs/)                                 |
| **Node.js**     | 20+     | [nodejs.org](https://nodejs.org/) or `nvm`                      |
| **Python**      | 3.11+   | [python.org](https://www.python.org/) or system package manager |
| **System libs** | —       | See platform-specific section below                             |

### Recommended

| Tool           | Purpose                      | Install                                                       |
| -------------- | ---------------------------- | ------------------------------------------------------------- |
| `uv`           | Fast Python package manager  | `curl -LsSf https://astral.sh/uv/install.sh \| sh`            |
| `just`         | Command runner (like `make`) | `cargo install just`                                          |
| Android Studio | Mobile client development    | [developer.android.com](https://developer.android.com/studio) |

### Platform-Specific System Libraries

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y \
    libwebkit2gtk-4.1-dev \
    libappindicator3-dev \
    librsvg2-dev \
    patchelf \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file
```

#### Fedora

```bash
sudo dnf install -y \
    webkit2gtk4.1-devel \
    openssl-devel \
    gtk3-devel \
    libappindicator-gtk3-devel \
    librsvg2-devel \
    patchelf
```

#### macOS

```bash
# Xcode command line tools (required)
xcode-select --install

# No additional system libraries needed — Tauri uses WebKit (built into macOS)
```

#### Arch Linux

```bash
sudo pacman -S --needed \
    webkit2gtk-4.1 \
    base-devel \
    curl \
    wget \
    file \
    openssl \
    appmenu-gtk-module \
    gtk3 \
    libappindicator-gtk3 \
    librsvg
```

---

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ACFHarbinger/Pi-Assistant.git
cd Pi-Assistant
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Set Up the Python Sidecar

```bash
cd sidecar

# Using uv (recommended):
uv venv
uv sync

# Using pip:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cd ..
```

### 4. Verify Rust Toolchain

```bash
rustc --version    # should be 1.75+
cargo --version
```

### 5. (Optional) Set Up Android

```bash
# Ensure ANDROID_HOME is set
export ANDROID_HOME=$HOME/Android/Sdk

cd android
./gradlew assembleDebug    # verify the build works
cd ..
```

---

## Running in Development

### Desktop App (Primary)

```bash
npm run tauri dev
```

This command:

1. Starts the Vite dev server for the React frontend (hot reload enabled).
2. Compiles and runs the Rust Tauri backend.
3. The Rust backend spawns the Python sidecar automatically.
4. Opens the desktop window.

The frontend is served at `http://localhost:1420` (Vite default). The WebSocket server for mobile starts on port `9120`.

### Frontend Only (No Rust)

Useful for UI development when you don't need the backend:

```bash
npm run dev
```

The frontend runs standalone. Tauri IPC calls will fail, but you can mock them for UI development.

### Python Sidecar Only

Useful for testing the ML pipeline independently:

```bash
cd sidecar
source .venv/bin/activate   # or: uv run
python -m pi_sidecar
```

The sidecar reads NDJSON from stdin and writes to stdout. You can test it manually:

```bash
echo '{"id":"test-1","method":"health.ping","params":{}}' | python -m pi_sidecar
```

### Android Client

```bash
cd android
./gradlew installDebug
```

Then configure the server IP in the app's settings to point to your desktop machine.

---

## Building for Production

### Desktop App

```bash
npm run tauri build
```

This produces platform-specific installers:

- **Linux**: `.deb`, `.AppImage` in `src-tauri/target/release/bundle/`
- **macOS**: `.dmg`, `.app` in `src-tauri/target/release/bundle/`
- **Windows**: `.msi`, `.exe` in `src-tauri/target/release/bundle/`

### Python Sidecar (Bundled Binary)

The sidecar must be bundled as a standalone executable for distribution:

```bash
cd sidecar
python build.py
```

This uses PyInstaller to create a single binary. The output is placed in `src-tauri/binaries/` with the appropriate target triple suffix:

```
src-tauri/binaries/
  pi-sidecar-x86_64-unknown-linux-gnu
  pi-sidecar-aarch64-apple-darwin
  pi-sidecar-x86_64-pc-windows-msvc.exe
```

Tauri's `externalBin` configuration in `tauri.conf.json` references these paths.

### Android Release

```bash
cd android
./gradlew assembleRelease
```

Sign with your keystore before distribution.

---

## Project Layout for Developers

```
Pi-Assistant/
├── src-tauri/src/
│   ├── main.rs              ← Desktop entry point
│   ├── lib.rs               ← Tauri builder, plugin setup
│   ├── state.rs             ← AppState, AgentState
│   ├── commands/            ← Tauri #[command] handlers (frontend IPC)
│   ├── agent/               ← Agent loop, planner, executor
│   ├── tools/               ← Tool trait + implementations
│   ├── ipc/                 ← Rust <-> Python NDJSON bridge
│   ├── ws/                  ← WebSocket server for mobile
│   ├── memory/              ← SQLite + sqlite-vec memory
│   ├── safety/              ← Permission engine
│   └── config/              ← Default settings
│
├── src/                     ← React/TypeScript frontend
│   ├── components/          ← UI components
│   ├── hooks/               ← Custom React hooks
│   ├── stores/              ← Zustand stores
│   └── services/            ← Tauri IPC wrappers
│
├── sidecar/src/pi_sidecar/  ← Python ML sidecar
│   ├── ipc/                 ← NDJSON transport + handler
│   ├── inference/           ← Embedding + completion engines
│   ├── training/            ← Lightning module + trainer
│   └── models/              ← Model registry
│
├── android/app/src/main/    ← Kotlin Android client
│   └── java/dev/piassistant/android/
│       ├── network/         ← WebSocket client
│       ├── viewmodel/       ← ViewModels
│       ├── ui/              ← Compose screens + components
│       └── voice/           ← Speech recognizer
│
├── protocol/                ← Shared protocol definitions
│   ├── schemas/             ← JSON Schema files
│   ├── rust/                ← Rust serde types
│   ├── python/              ← Python Pydantic models
│   └── kotlin/              ← Kotlin data classes
│
└── crates/pi-core/          ← Shared Rust types (no Tauri dep)
```

---

## Environment Variables

| Variable                  | Default                  | Description                                                        |
| ------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `PI_ASSISTANT_DATA_DIR`   | `~/.pi-assistant/data`   | SQLite database and data storage                                   |
| `PI_ASSISTANT_MODELS_DIR` | `~/.pi-assistant/models` | ML model storage                                                   |
| `PI_ASSISTANT_WS_PORT`    | `9120`                   | WebSocket server port                                              |
| `PI_ASSISTANT_LOG_LEVEL`  | `info`                   | Rust tracing log level (`trace`, `debug`, `info`, `warn`, `error`) |
| `PI_ASSISTANT_PYTHON`     | `python3`                | Path to Python interpreter for sidecar                             |
| `PI_ASSISTANT_CHROME`     | (auto-detected)          | Path to Chrome/Chromium binary                                     |

---

## Common Development Tasks

### Adding a Tauri Command

1. Create the handler in `src-tauri/src/commands/`:
   ```rust
   #[tauri::command]
   async fn my_command(state: tauri::State<'_, Arc<AppState>>) -> Result<String, String> {
       Ok("result".to_string())
   }
   ```
2. Register in `lib.rs`:
   ```rust
   .invoke_handler(tauri::generate_handler![..., commands::my_command])
   ```
3. Call from frontend:
   ```typescript
   const result = await invoke<string>("my_command");
   ```

### Adding a Python IPC Method

1. Implement the handler in `sidecar/src/pi_sidecar/ipc/handler.py`.
2. Register in `RequestHandler._handlers`.
3. Call from Rust:
   ```rust
   let result = sidecar.request("your.method", json!({"key": "value"})).await?;
   ```

### Inspecting the Database

```bash
sqlite3 ~/.pi-assistant/data/pi-assistant.db
.tables
.schema messages
SELECT * FROM messages ORDER BY created_at DESC LIMIT 10;
```

### Inspecting IPC Traffic

Set the log level to `debug` to see all NDJSON messages:

```bash
PI_ASSISTANT_LOG_LEVEL=debug npm run tauri dev
```

Or test the sidecar directly:

```bash
echo '{"id":"1","method":"inference.embed","params":{"text":"hello world"}}' | python -m pi_sidecar
```

---

## IDE Setup

### VS Code (Recommended)

Recommended extensions:

- `rust-analyzer` — Rust language server
- `tauri-vscode` — Tauri integration
- `ms-python.python` — Python support
- `esbenp.prettier-vscode` — TypeScript formatting
- `bradlc.vscode-tailwindcss` — Tailwind IntelliSense

### IntelliJ IDEA / Android Studio

Use for Android development. The `android/` directory is a standalone Gradle project that can be opened directly.

### PyCharm

Open the `sidecar/` directory as a standalone Python project. Set the interpreter to the virtual environment at `sidecar/.venv/`.
