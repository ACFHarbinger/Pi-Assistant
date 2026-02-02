# Dependencies

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Kotlin](https://img.shields.io/badge/kotlin-%237F52FF.svg?style=for-the-badge&logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![Tauri](https://img.shields.io/badge/tauri-%2324C8DB.svg?style=for-the-badge&logo=tauri&logoColor=%23262626)](https://tauri.app/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Jetpack Compose](https://img.shields.io/badge/Jetpack_Compose-4285F4?style=for-the-badge&logo=jetpackcompose&logoColor=white)](https://developer.android.com/jetpack/compose)

Complete dependency reference for all four runtimes in Pi-Assistant.

---

## Rust (Tauri Core)

Defined in `src-tauri/Cargo.toml` and the workspace `Cargo.toml`.

### Workspace Dependencies

These versions are pinned at the workspace level and shared across all crates.

| Crate                | Version                       | Purpose                      |
| -------------------- | ----------------------------- | ---------------------------- |
| `serde`              | 1.x                           | Serialization framework      |
| `serde_json`         | 1.x                           | JSON serialization           |
| `tokio`              | 1.x (features: `full`)        | Async runtime                |
| `tracing`            | 0.1                           | Structured logging facade    |
| `tracing-subscriber` | 0.3                           | Log output formatting        |
| `uuid`               | 1.x (features: `v4`, `serde`) | Unique IDs for correlation   |
| `chrono`             | 0.4 (features: `serde`)       | Date/time handling           |
| `thiserror`          | 2.x                           | Derive macro for error types |
| `anyhow`             | 1.x                           | Flexible error handling      |

### Tauri App (`src-tauri`)

| Crate                | Version                    | Purpose                               |
| -------------------- | -------------------------- | ------------------------------------- |
| `tauri`              | 2.x                        | Desktop app framework                 |
| `tauri-plugin-shell` | 2.x                        | Shell access, sidecar management      |
| `tauri-build`        | 2.x                        | Build script for Tauri                |
| `tokio-util`         | 0.7                        | `CancellationToken` for agent loop    |
| `axum`               | 0.8                        | WebSocket server for mobile clients   |
| `rusqlite`           | 0.32 (features: `bundled`) | SQLite database (statically linked)   |
| `sqlite-vec`         | 0.1.x                      | Vector similarity search extension    |
| `chromiumoxide`      | 0.7                        | Headless Chrome (CDP) browser control |
| `regex`              | 1.x                        | Permission pattern matching           |
| `async-trait`        | 0.1                        | Async methods in traits               |

### Protocol Crate (`protocol/rust`)

| Crate        | Version | Purpose              |
| ------------ | ------- | -------------------- |
| `serde`      | 1.x     | Serialization        |
| `serde_json` | 1.x     | JSON serialization   |
| `uuid`       | 1.x     | Request/response IDs |

### Core Crate (`crates/pi-core`)

| Crate   | Version | Purpose       |
| ------- | ------- | ------------- |
| `serde` | 1.x     | Serialization |
| `uuid`  | 1.x     | ID types      |

---

## Python (ML Sidecar)

Defined in `sidecar/pyproject.toml`. Requires Python >= 3.11.

### Core Dependencies

| Package                 | Version | Purpose                                      |
| ----------------------- | ------- | -------------------------------------------- |
| `torch`                 | >= 2.2  | Tensor computation, GPU acceleration         |
| `pytorch-lightning`     | >= 2.2  | Training framework with callbacks            |
| `transformers`          | >= 4.40 | HuggingFace model hub, tokenizers, pipelines |
| `sentence-transformers` | >= 3.0  | Text embedding models (all-MiniLM-L6-v2)     |
| `pydantic`              | >= 2.5  | Data validation and settings management      |

### Dev Dependencies

| Package          | Version | Purpose              |
| ---------------- | ------- | -------------------- |
| `pytest`         | >= 8.0  | Test framework       |
| `pytest-asyncio` | >= 0.23 | Async test support   |
| `ruff`           | >= 0.4  | Linter and formatter |

### Build Dependencies

| Package       | Version | Purpose                                 |
| ------------- | ------- | --------------------------------------- |
| `pyinstaller` | >= 6.5  | Bundle sidecar as standalone executable |

### Transitive / Notable

| Package           | Pulled By      | Notes                         |
| ----------------- | -------------- | ----------------------------- |
| `tokenizers`      | `transformers` | Rust-backed fast tokenization |
| `safetensors`     | `transformers` | Safe model weight format      |
| `huggingface-hub` | `transformers` | Model download and caching    |
| `numpy`           | `torch`        | Array computation             |
| `tqdm`            | `transformers` | Progress bars (stderr only)   |

### Models Downloaded at Runtime

| Model                                    | Size    | Dimensions | Purpose                            |
| ---------------------------------------- | ------- | ---------- | ---------------------------------- |
| `sentence-transformers/all-MiniLM-L6-v2` | ~80 MB  | 384        | Text embeddings for memory         |
| `gpt2` (or user-configured)              | ~500 MB | —          | Default base model for fine-tuning |

Models are cached in `~/.pi-assistant/models/` and managed by the `ModelRegistry`.

---

## TypeScript (React Frontend)

Defined in `package.json`. Uses Vite as the build tool.

### Runtime Dependencies

| Package                    | Version | Purpose                               |
| -------------------------- | ------- | ------------------------------------- |
| `react`                    | ^18.x   | UI framework                          |
| `react-dom`                | ^18.x   | DOM rendering                         |
| `@tauri-apps/api`          | ^2.x    | Tauri v2 IPC (invoke, events, window) |
| `@tauri-apps/plugin-shell` | ^2.x    | Shell access from frontend (optional) |
| `zustand`                  | ^4.x    | Lightweight state management          |
| `react-markdown`           | ^9.x    | Render agent responses as markdown    |
| `lucide-react`             | ^0.400  | Icon library                          |

### Dev Dependencies

| Package                | Version | Purpose                     |
| ---------------------- | ------- | --------------------------- |
| `typescript`           | ^5.x    | Type checking               |
| `vite`                 | ^5.x    | Dev server and bundler      |
| `@vitejs/plugin-react` | ^4.x    | React Fast Refresh for Vite |
| `tailwindcss`          | ^3.x    | Utility-first CSS           |
| `postcss`              | ^8.x    | CSS processing              |
| `autoprefixer`         | ^10.x   | Vendor prefix automation    |
| `@tauri-apps/cli`      | ^2.x    | Tauri CLI for dev/build     |
| `eslint`               | ^8.x    | Linting                     |
| `prettier`             | ^3.x    | Code formatting             |

---

## Kotlin (Android Client)

Defined in `android/gradle/libs.versions.toml` using Gradle version catalogs.

### Core Dependencies

| Library                      | Version | Purpose                         |
| ---------------------------- | ------- | ------------------------------- |
| `okhttp`                     | 5.0.x   | HTTP client + WebSocket         |
| `okhttp-coroutines`          | 5.0.x   | Coroutine extensions for OkHttp |
| `kotlinx-serialization-json` | 1.7.x   | JSON serialization              |
| `kotlinx-coroutines-core`    | 1.10.x  | Coroutine primitives            |
| `kotlinx-coroutines-android` | 1.10.x  | Android main dispatcher         |

### Jetpack / AndroidX

| Library                       | Version | Purpose                            |
| ----------------------------- | ------- | ---------------------------------- |
| `compose-bom`                 | 2025.x  | Jetpack Compose version alignment  |
| `material3`                   | 1.3.x   | Material Design 3 components       |
| `lifecycle-viewmodel-compose` | 2.8.x   | ViewModel integration with Compose |
| `lifecycle-runtime-compose`   | 2.8.x   | Lifecycle-aware Compose utilities  |
| `activity-compose`            | 1.9.x   | Activity integration for Compose   |
| `navigation-compose`          | 2.8.x   | Navigation for Compose screens     |

### Build Toolchain

| Tool                    | Version | Purpose                 |
| ----------------------- | ------- | ----------------------- |
| Android Gradle Plugin   | 8.5.x   | Android build system    |
| Kotlin                  | 2.0.x   | Language compiler       |
| Kotlin Compose Compiler | 2.0.x   | Compose compiler plugin |
| Gradle                  | 8.9+    | Build orchestration     |

### Test Dependencies

| Library           | Version | Purpose            |
| ----------------- | ------- | ------------------ |
| `junit`           | 4.13.x  | Unit testing       |
| `espresso-core`   | 3.6.x   | UI testing         |
| `compose-ui-test` | (BOM)   | Compose UI testing |
| `turbine`         | 1.1.x   | Flow testing       |
| `mockk`           | 1.13.x  | Mocking framework  |

---

## System Requirements

### Development Machine

| Requirement | Minimum                            | Recommended                         |
| ----------- | ---------------------------------- | ----------------------------------- |
| OS          | Linux (x86_64), macOS (ARM/x86_64) | Ubuntu 22.04+, macOS 14+            |
| RAM         | 8 GB                               | 16 GB+ (for ML model loading)       |
| Disk        | 5 GB free                          | 20 GB+ (models, build artifacts)    |
| GPU         | Not required                       | NVIDIA with CUDA 12+ (for training) |

### Runtime

| Component       | Required                         |
| --------------- | -------------------------------- |
| Chrome/Chromium | For headless browser tool        |
| Network         | LAN for mobile client connection |
| Port 9120       | WebSocket server (configurable)  |

---

## Dependency Update Policy

- **Rust**: Use `cargo update` for patch versions. Major version bumps require review for breaking changes, especially for `tauri` and `axum`.
- **Python**: Pin exact versions in `uv.lock`. Use `uv lock --upgrade` to refresh.
- **TypeScript**: Use `npm update` for patch/minor. Audit with `npm audit`.
- **Kotlin**: Version catalog in `libs.versions.toml` centralizes all versions. Update BOM versions together.

### Security Auditing

```bash
# Rust
cargo audit

# Python
pip-audit

# TypeScript
npm audit

# Kotlin
./gradlew dependencyCheckAnalyze  # OWASP dependency-check plugin
```
