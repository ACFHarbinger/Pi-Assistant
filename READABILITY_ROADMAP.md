# Readability & Human Understanding Roadmap

This roadmap targets improvements that make the Pi-Assistant codebase easier for new and existing developers to read, navigate, debug, and extend. It is organized by priority and domain.

---

## Current State Summary

| Component | Module Docs | Inline Docs | Examples | Runbooks | Grade |
|-----------|-------------|-------------|----------|----------|-------|
| Rust backend | Good | Poor | None | None | C+ |
| React frontend | Good | Minimal | None | None | C |
| Python sidecar | Excellent | Good | None | Limited | B+ |
| Kotlin Android | None | None | None | None | D |
| Architecture docs | Excellent | N/A | Limited | Limited | A- |
| IPC protocol | Good | Good | Some | None | B |
| Build / deploy | Good | Poor | None | None | C+ |

The project has **strong high-level architecture documentation** (ARCHITECTURE.md, CLAUDE.md, docs/) but **inconsistent inline code documentation** and **no runbooks or worked examples**.

---

## Phase 1 — Critical (Core execution paths)

### 1.1 Document the Rust Agent Loop

The plan-execute-observe cycle in `src-tauri/src/agent/loop.rs` is the heart of the system and currently has no function-level doc comments.

- [ ] Add `///` doc comment to `agent_loop()` explaining the iteration cycle, state transitions, and termination conditions
- [ ] Add `///` doc comment to `spawn_agent_loop()` explaining cancellation token semantics and the `AgentLoopHandle` lifetime
- [ ] Document the error budget mechanism: what `consecutive_errors` and `MAX_CONSECUTIVE_ERRORS` control and why the threshold is 3
- [ ] Add inline comments for the subtask management block (lines 355-411) explaining the `manage_subtasks` virtual tool
- [ ] Annotate the tool execution block explaining why the `Arc` is cloned and the read lock dropped before `execute()` (deadlock prevention)

**Files:** `src-tauri/src/agent/loop.rs`, `src-tauri/src/agent/planner.rs`, `src-tauri/src/agent/executor.rs`

### 1.2 Document React Store Actions

`src/stores/agentStore.ts` has 20+ methods with no JSDoc. Developers cannot tell `sendAnswer()` from `sendMessage()` or understand when `startAgent()` will throw.

- [ ] Add JSDoc to every public method in `agentStore.ts` with parameters, return values, and one-line usage example
- [ ] Add JSDoc to `deviceStore.ts` explaining `DeviceCapability` fields (`can_train`, `memory_total_mb` vs `vram_total_mb`)
- [ ] Add JSDoc to `trainingStore.ts` explaining the run lifecycle
- [ ] Document the Tauri event listener setup in `setupListeners()`: which events exist, what payloads they carry, and when they fire

**Files:** `src/stores/agentStore.ts`, `src/stores/deviceStore.ts`, `src/stores/trainingStore.ts`

### 1.3 Document Permission Precedence

`src-tauri/src/safety/permission.rs` checks overrides, wildcards, and tool-specific rules in a specific order. That order matters and is not explained.

- [ ] Add a module-level doc comment to `permission.rs` with a numbered precedence list (exact override > wildcard override > tool-specific rule > default NeedsApproval)
- [ ] Document `find_override()` explaining the wildcard matching algorithm and its O(n) cost
- [ ] Document `add_scoped_override()` explaining expiry semantics and the relationship to `chrono::Utc::now()`
- [ ] Add inline comment explaining why expired overrides fall through to rule-based checks instead of being removed eagerly

**Files:** `src-tauri/src/safety/permission.rs`, `src-tauri/src/safety/rules.rs`

### 1.4 Document IPC Sidecar Lifecycle

`src-tauri/src/ipc/sidecar.rs` manages correlation IDs, pending requests, and progress callbacks with no documentation on failure modes.

- [ ] Document `SidecarHandle` struct: what each field is for, when the process is spawned, and how cleanup works
- [ ] Document the correlation ID scheme: how `pending` and `pending_progress` maps work and what happens on ID collision
- [ ] Document timeout behavior: what happens if a sidecar response never arrives
- [ ] Document reconnection behavior: what happens if the child process crashes mid-request

**Files:** `src-tauri/src/ipc/sidecar.rs`, `src-tauri/src/ipc/mod.rs`

---

## Phase 2 — High Priority (Developer onboarding)

### 2.1 Create a Debugging Guide

There is no document explaining how to diagnose common problems.

- [ ] Create `docs/DEBUGGING.md` covering:
  - Sidecar crash: how to check logs, restart behavior, common causes
  - Agent stuck in infinite loop: how to identify (iteration counter), how to stop, loop detection
  - Tool permission denied: where rules are defined, how to add overrides, testing permissions
  - IPC message tracing: how to enable debug logging for NDJSON traffic
  - Memory issues: how to inspect SQLite database, check embedding quality
  - Frontend state desync: how to diagnose stale Zustand state, re-sync with backend

### 2.2 Annotate Magic Numbers and Constants

Hard-coded values throughout the codebase have no rationale.

- [ ] `shell.rs`: Document why the default timeout is 60 seconds
- [ ] `loop.rs`: Document why `MAX_CONSECUTIVE_ERRORS = 3` and the command channel size of 32
- [ ] `state.rs`: Document why the agent command channel capacity is 32
- [ ] `sqlite.rs`: Document why `retrieve_context` defaults to 10 recent messages
- [ ] `api.rs`: Document why the default request timeout is 30 seconds and cache key uses SHA-256
- [ ] `browser.rs`: Document why only `localhost` and `127.0.0.1` are in the default domain allowlist

### 2.3 Document Tool Implementation Pattern

A developer wanting to add a new tool must reverse-engineer the pattern from existing tools.

- [ ] Create `docs/WRITING-A-TOOL.md` covering:
  - The `Tool` trait and what each method must return
  - The sub-action dispatch pattern (single tool, `action` parameter, match block)
  - How to format `ToolResult` output for both human and structured consumption
  - How to register a tool in `ToolRegistry::new()` vs `lib.rs` (when AppHandle is needed)
  - Permission tier selection criteria
  - How to add tool-specific permission rules in `permission.rs`
  - Complete worked example: a minimal "echo" tool from struct to registration

### 2.4 Document the Python Handler Registry

`sidecars/ml/src/pi_sidecar/ml_sidecar_main.py` builds a handler map but the full list of supported IPC methods is not centrally documented.

- [ ] Add a module-level docstring to `ml_sidecar_main.py` listing all supported methods grouped by domain (health, inference, training, device, model)
- [ ] Document each handler's expected parameters and return schema
- [ ] Document the method naming convention (`domain.action`)
- [ ] Create a reference table in `docs/IPC-PROTOCOL.md` mapping each method to its handler function and expected parameters

### 2.5 Document React Component Hierarchy

New frontend developers have no map of the component tree.

- [ ] Add a component-level JSDoc to each component in `src/components/`:
  - Purpose (one sentence)
  - Which store(s) it reads from
  - What Tauri commands/events it uses
  - Key internal state
- [ ] Create a data flow comment block at the top of `App.tsx` showing the component hierarchy and which overlays are conditional

---

## Phase 3 — Medium Priority (Long-term maintainability)

### 3.1 Configuration Reference

Users and developers cannot find a single reference for all configurable values.

- [ ] Create `docs/CONFIGURATION.md` covering:
  - `~/.pi-assistant/` directory layout and what each file does
  - Environment variables that affect behavior
  - `mcp_config.json` schema and fields
  - Project-level `.pi-assistant.toml` (if implemented)
  - Sidecar configuration (Python model paths, device selection)
- [ ] Add a `config.example.toml` with commented defaults

### 3.2 Android Client Documentation

The Kotlin client has zero documentation.

- [ ] Add KDoc to `WebSocketClient.kt` explaining:
  - Why `ws://10.0.2.2:9120/ws` is the default URL (Android emulator loopback)
  - How to configure for physical devices on the same LAN
  - Reconnection behavior and message queueing
- [ ] Add KDoc to `ChatViewModel.kt` explaining the `StateFlow` architecture
- [ ] Create `android/README.md` with build instructions, API level requirements, and emulator setup

### 3.3 Explain State Machine Transitions

`AgentState` in `crates/pi-core/src/agent_types.rs` defines the state enum but does not document valid transitions.

- [ ] Add doc comments to each `AgentState` variant listing which transitions are valid from that state
- [ ] Add doc comments to each `AgentCommand` variant explaining which state it can be sent in and what the result is
- [ ] Add doc comments to `StopReason` variants explaining what causes each one

### 3.4 Error Code Reference

Python sidecar errors use string codes ("MODEL_NOT_LOADED", "DEVICE_NOT_AVAILABLE") that are not centrally listed.

- [ ] Create an enum or constant module in the Python sidecar listing all error codes
- [ ] Document each code's meaning, common causes, and resolution steps
- [ ] Reference the error codes from `docs/DEBUGGING.md`

### 3.5 Sequence Diagrams for Key Flows

Architecture docs have box diagrams but no sequence diagrams showing message flow over time.

- [ ] Add sequence diagrams (Mermaid) to `ARCHITECTURE.md` or a new `docs/FLOWS.md` for:
  - User submits task -> agent starts -> plans -> executes tool -> stores result -> completes
  - User denies permission -> agent skips tool -> re-plans
  - Agent asks question -> pauses -> user answers -> resumes
  - Training tool deploys model -> registers as new tool -> agent uses it
  - Mobile client connects via WebSocket -> receives state updates -> sends command

---

## Phase 4 — Nice to Have (Developer experience)

### 4.1 Worked Examples

- [ ] Create `examples/custom-tool/` with a complete minimal tool (Rust file + registration + permission rule)
- [ ] Create `examples/training-workflow/` showing how to train a classifier and deploy it
- [ ] Create `examples/android-debug/` showing how to connect a physical device

### 4.2 Inline Type Narrowing Hints

TypeScript components frequently cast with `as any`. These should be narrowed or annotated.

- [ ] Audit all `as any` casts in `src/` and replace with proper type guards or document why the cast is necessary
- [ ] Add discriminated union helpers for `AgentState` variants so components don't need manual `"iteration" in state.data` checks

### 4.3 Test Documentation

- [ ] Add doc comments to all `#[cfg(test)]` modules explaining what each test validates and why
- [ ] Add a test naming convention comment to the test modules (e.g., `test_<function>_<scenario>_<expected>`)

### 4.4 Dependency Rationale

- [ ] Add comments in `Cargo.toml` for non-obvious dependencies explaining why they were chosen over alternatives
- [ ] Add comments in `package.json` for non-obvious frontend dependencies

---

## Measurement

Progress can be tracked with these metrics:

| Metric | Current | Target |
|--------|---------|--------|
| Rust functions with `///` docs (public) | ~30% | 90% |
| React store methods with JSDoc | 0% | 100% |
| Components with purpose docstring | ~20% | 100% |
| Runbook scenarios documented | 0 | 6+ |
| Worked examples | 0 | 3+ |
| Android KDoc coverage | 0% | 80% |
| Magic numbers with rationale comments | ~10% | 100% |
