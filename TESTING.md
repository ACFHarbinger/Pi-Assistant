# Testing

Testing strategy, test organization, and instructions for running tests across all four runtimes.

---

## Testing Philosophy

Pi-Assistant spans four languages and multiple process boundaries. The testing strategy is organized into layers:

| Layer | Scope | Speed | Tools |
|-------|-------|-------|-------|
| **Unit** | Single function/struct, mocked dependencies | Fast (ms) | `cargo test`, `pytest`, `vitest`, `JUnit` |
| **Integration** | Module-to-module within one runtime | Medium (seconds) | Same tools, but with real dependencies |
| **IPC** | Rust <-> Python round-trip communication | Medium | Custom test harness |
| **End-to-End** | Full app: UI -> Rust -> Python -> Storage | Slow (minutes) | Tauri test driver, Playwright |

The focus is on thorough **unit and integration tests** for each runtime, plus targeted **IPC tests** for the cross-language boundary.

---

## Rust Tests

### Running

```bash
# All workspace tests
cargo test --workspace

# Specific crate
cargo test -p pi-assistant   # src-tauri
cargo test -p pi-core        # crates/pi-core
cargo test -p pi-protocol    # protocol/rust

# Specific test
cargo test -p pi-assistant agent::loop_::tests::test_cancellation

# With output (for debugging)
cargo test --workspace -- --nocapture
```

### Organization

Tests live alongside the code they test, using Rust's `#[cfg(test)]` convention:

```
src-tauri/src/
├── agent/
│   ├── loop.rs          # contains #[cfg(test)] mod tests { ... }
│   └── planner.rs       # contains #[cfg(test)] mod tests { ... }
├── tools/
│   ├── shell.rs         # unit tests for command parsing, permission checks
│   └── code.rs          # unit tests for path validation
├── ipc/
│   ├── ndjson.rs        # codec tests (serialize/deserialize)
│   └── router.rs        # correlation ID routing tests
├── memory/
│   ├── sqlite.rs        # schema creation, CRUD operations
│   └── vector.rs        # vector insert/search with test embeddings
└── safety/
    ├── permission.rs    # permission engine rule matching
    └── rules.rs         # regex pattern tests
```

### What to Test

| Module | Key Test Cases |
|--------|---------------|
| `agent/loop.rs` | Cancellation token stops loop; iteration limit enforced; pause/resume cycle works; stop during pause exits cleanly |
| `tools/shell.rs` | Command output capture; timeout enforcement; exit code handling |
| `tools/code.rs` | Path validation blocks `..`; blocked directories rejected; canonicalization works |
| `ipc/ndjson.rs` | Valid JSON round-trips; malformed lines are skipped; newline delimiters |
| `ipc/router.rs` | Responses routed to correct caller by ID; timeout removes pending entry; progress messages don't complete requests |
| `memory/sqlite.rs` | Schema creation idempotent; insert + query; foreign key constraints |
| `memory/vector.rs` | Insert vector + search by similarity; empty table returns empty; dimension mismatch errors |
| `safety/permission.rs` | Block rules override auto-approve; user overrides take precedence; default is ask-user; path traversal blocked |

### Mocking the Python Sidecar

For unit tests that need IPC without starting Python, use a mock sidecar:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Create a mock sidecar that returns predefined responses
    fn mock_sidecar() -> SidecarHandle {
        // Use tokio::process::Command to spawn a simple echo script
        // or use a channel-based mock that simulates request/response
        todo!("implement mock sidecar for tests")
    }
}
```

Alternatively, test the NDJSON codec and router separately from the process spawning.

---

## Python Tests

### Running

```bash
cd sidecar

# All tests
pytest

# Specific file
pytest tests/test_ipc.py

# Specific test
pytest tests/test_ipc.py::test_ndjson_roundtrip

# With coverage
pytest --cov=pi_sidecar --cov-report=html

# Verbose
pytest -v
```

### Organization

```
sidecar/tests/
├── test_ipc.py           # NDJSON transport, request handler routing
├── test_inference.py     # Embedding generation, completion
└── test_training.py      # Lightning module forward pass, callback behavior
```

### What to Test

| Module | Key Test Cases |
|--------|---------------|
| `ipc/ndjson_transport.py` | JSON round-trip; malformed input skipped; concurrent writes don't interleave; progress messages have correct format |
| `ipc/handler.py` | Known methods dispatch correctly; unknown method raises error; health ping returns version |
| `inference/engine.py` | Embedding returns correct dimensions (384); completion returns non-empty text; model loading is lazy |
| `inference/embeddings.py` | Same text produces same embedding; different text produces different embeddings; empty string handled |
| `training/lightning_module.py` | Forward pass produces loss; training step logs metrics; optimizer configured correctly |
| `training/trainer.py` | Progress callback called per epoch; stop training cancels cleanly |
| `models/registry.py` | List models on empty dir returns empty; load non-existent model raises error |

### Async Tests

Use `pytest-asyncio` for testing async code:

```python
import pytest

@pytest.mark.asyncio
async def test_ndjson_roundtrip():
    transport = NdjsonTransport()
    # ... test async read/write
```

### Fixtures for ML Tests

ML tests that load models are slow. Use pytest fixtures with `scope="session"` to load once:

```python
@pytest.fixture(scope="session")
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
```

Mark these tests so they can be skipped in CI without GPU:

```python
@pytest.mark.slow
@pytest.mark.requires_model
def test_embedding_dimensions(embedding_model):
    vec = embedding_model.encode("hello")
    assert len(vec) == 384
```

```bash
# Skip slow tests
pytest -m "not slow"

# Run everything
pytest
```

---

## TypeScript Tests

### Running

```bash
# Lint
npm run lint

# Type check
npm run typecheck

# Unit tests (if configured with vitest)
npm run test
```

### Organization

```
src/
├── components/
│   ├── AgentStatus.tsx
│   └── __tests__/
│       └── AgentStatus.test.tsx
├── hooks/
│   └── __tests__/
│       └── useAgentState.test.ts
├── stores/
│   └── __tests__/
│       └── agentStore.test.ts
└── services/
    └── __tests__/
        └── tauriIpc.test.ts
```

### What to Test

| Module | Key Test Cases |
|--------|---------------|
| `stores/agentStore.ts` | State transitions correct; addMessage appends; setTask updates |
| `hooks/useAgentState.ts` | Subscribes to Tauri events; unsubscribes on unmount |
| `components/PermissionDialog.tsx` | Renders command text; approve calls handler; deny calls handler; "remember" checkbox toggles |
| `services/tauriIpc.ts` | Functions call `invoke` with correct command names and arguments |

### Mocking Tauri

In test environments, `@tauri-apps/api` isn't available. Mock it:

```typescript
// __mocks__/@tauri-apps/api/core.ts
export const invoke = vi.fn();

// __mocks__/@tauri-apps/api/event.ts
export const listen = vi.fn(() => Promise.resolve(() => {}));
```

---

## Kotlin / Android Tests

### Running

```bash
cd android

# Unit tests
./gradlew test

# Instrumented tests (requires emulator or device)
./gradlew connectedAndroidTest
```

### Organization

```
android/app/src/
├── test/                           # Unit tests (JVM)
│   └── java/dev/piassistant/android/
│       ├── network/
│       │   └── WebSocketClientTest.kt
│       └── viewmodel/
│           └── AgentViewModelTest.kt
└── androidTest/                    # Instrumented tests (device)
    └── java/dev/piassistant/android/
        └── ui/
            └── HomeScreenTest.kt
```

### What to Test

| Module | Key Test Cases |
|--------|---------------|
| `WebSocketClient` | Message parsing; connection state transitions; send returns false when disconnected |
| `ConnectionManager` | Exponential backoff timing; reconnect on error; disconnect cancels reconnect |
| `AgentViewModel` | State flow emits correct agent state; chat messages accumulate; permission requests forwarded |
| `SpeechRecognizerHelper` | Callbacks invoked on result; error callback on failure |

### Testing Flows with Turbine

Use [Turbine](https://github.com/cashapp/turbine) for testing Kotlin Flows:

```kotlin
@Test
fun `agent state updates are emitted`() = runTest {
    val viewModel = AgentViewModel(mockWsClient)

    viewModel.agentState.test {
        assertEquals("Idle", awaitItem())

        // Simulate incoming WebSocket message
        mockWsClient.emitMessage(WsMessage(type = "agent_state_update", ...))

        assertEquals("Running", awaitItem())
    }
}
```

---

## IPC Integration Tests

These tests verify the Rust <-> Python communication boundary.

### Approach

1. **Rust-side test** spawns the actual Python sidecar as a child process.
2. Sends a request via NDJSON.
3. Asserts the response is correct.
4. Verifies progress messages for training requests.

```rust
#[tokio::test]
async fn test_sidecar_health_ping() {
    let mut sidecar = SidecarHandle::new();
    sidecar.start().await.expect("sidecar should start");

    let response = sidecar.request("health.ping", json!({})).await.unwrap();
    assert_eq!(response["status"], "ok");

    sidecar.stop().await.unwrap();
}

#[tokio::test]
async fn test_sidecar_embedding() {
    let mut sidecar = SidecarHandle::new();
    sidecar.start().await.unwrap();

    let response = sidecar.request("inference.embed", json!({
        "text": "hello world"
    })).await.unwrap();

    let embedding = response["embedding"].as_array().unwrap();
    assert_eq!(embedding.len(), 384);

    sidecar.stop().await.unwrap();
}
```

These tests require the Python sidecar to be installed. Mark them appropriately:

```rust
#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_sidecar_health_ping() { ... }
```

```bash
# Run integration tests
cargo test --features integration
```

---

## CI Pipeline (GitHub Actions)

Recommended workflow:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: sudo apt install -y libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
      - run: cargo fmt --check --all
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo test --workspace

  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e "./sidecar[dev]"
      - run: cd sidecar && ruff check .
      - run: cd sidecar && ruff format --check .
      - run: cd sidecar && pytest -m "not slow"

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck

  android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: "17"
          distribution: "temurin"
      - run: cd android && ./gradlew test
```

---

## Code Coverage

### Rust

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --workspace --out html
# Opens htmlcov/ report
```

### Python

```bash
cd sidecar
pytest --cov=pi_sidecar --cov-report=html
# Opens htmlcov/index.html
```

### TypeScript

```bash
npm run test -- --coverage
```

### Coverage Targets

| Runtime | Target | Critical Modules |
|---------|--------|-----------------|
| Rust | 70%+ | `safety/permission.rs` (90%+), `ipc/ndjson.rs` (90%+), `memory/` (80%+) |
| Python | 70%+ | `ipc/` (90%+), `inference/` (80%+) |
| TypeScript | 60%+ | `stores/`, `services/` |
| Kotlin | 60%+ | `network/`, `viewmodel/` |

The safety and IPC modules are the most critical — bugs there can cause security issues or silent data corruption.
