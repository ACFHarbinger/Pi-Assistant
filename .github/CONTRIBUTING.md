# Contributing to Pi-Assistant

Thank you for your interest in contributing to Pi-Assistant. This document covers the guidelines, workflow, and standards for contributing to the project.

---

## Code of Conduct

Be respectful, constructive, and patient. This is a complex multi-language project — contributors come from different backgrounds (systems programming, ML, mobile, web). Assume good intent.

---

## Getting Started

1. **Read the docs first**: [ARCHITECTURE.md](ARCHITECTURE.md), [DEVELOPMENT.md](DEVELOPMENT.md), and [AGENTS.md](AGENTS.md) provide essential context.
2. **Set up your environment**: Follow [DEVELOPMENT.md](DEVELOPMENT.md) for build instructions.
3. **Find an issue**: Check the [issue tracker](https://github.com/ACFHarbinger/Pi-Assistant/issues) for items labeled `good first issue` or `help wanted`.

---

## Contribution Workflow

### 1. Fork and Branch

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Pi-Assistant.git
cd Pi-Assistant
git remote add upstream https://github.com/ACFHarbinger/Pi-Assistant.git
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the style guidelines below.
- Keep commits focused — one logical change per commit.
- Write or update tests for your changes.
- Update documentation if you change behavior or add features.

### 3. Test

```bash
# Rust
cargo test --workspace

# Python
cd sidecar && pytest

# TypeScript
npm run lint
npm run typecheck

# Android
cd android && ./gradlew test
```

See [TESTING.md](TESTING.md) for full testing details.

### 4. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `build` | Build system or dependency changes |
| `ci` | CI/CD configuration |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks |

**Scopes:**
| Scope | Component |
|-------|-----------|
| `core` | Rust Tauri core (`src-tauri/`) |
| `agent` | Agent loop and tools |
| `ipc` | Rust <-> Python IPC |
| `memory` | Memory subsystem |
| `safety` | Permission engine |
| `sidecar` | Python sidecar |
| `frontend` | React/TypeScript UI |
| `android` | Kotlin mobile client |
| `protocol` | Shared protocol schemas |

**Examples:**

```
feat(agent): add iteration timeout per tool call
fix(ipc): handle sidecar crash during inference request
docs(safety): document path canonicalization behavior
refactor(memory): extract vector search into dedicated module
test(sidecar): add integration tests for NDJSON transport
```

### 5. Pull Request

```bash
git push origin feature/your-feature-name
```

Open a pull request against `main`. In the PR description:

- **Summary**: What does this change do and why?
- **Testing**: How did you test it? What tests were added?
- **Breaking changes**: Does this change any public API, IPC protocol, or stored data format?
- **Screenshots**: If the change affects the UI, include before/after screenshots.

---

## Style Guidelines

### Rust

- **Edition**: 2021
- **Formatting**: `cargo fmt` (default rustfmt settings)
- **Linting**: `cargo clippy -- -D warnings` (all warnings are errors in CI)
- **Error handling**: Use `thiserror` for library errors, `anyhow` for application errors. Never use `.unwrap()` in production paths — use `.expect("reason")` only for invariants that are genuinely impossible to violate.
- **Async**: Use `tokio` primitives. Prefer `tokio::select!` over manual polling. Never block the async runtime with synchronous I/O.
- **Naming**: Follow Rust conventions. Types are `PascalCase`, functions are `snake_case`, constants are `SCREAMING_SNAKE_CASE`.
- **Modules**: One file per module. Use `mod.rs` sparingly — prefer named files (e.g., `tools/shell.rs` over `tools/mod.rs` containing shell logic).

### Python

- **Version**: 3.11+ (use modern syntax: `X | Y` unions, `match` statements)
- **Formatting**: `ruff format` (Black-compatible)
- **Linting**: `ruff check` with default rules
- **Type hints**: Required on all public functions. Use `from __future__ import annotations`.
- **Async**: Use `asyncio` for all I/O. The sidecar's main loop is async.
- **Imports**: Standard library, then third-party, then local. One blank line between groups.
- **Docstrings**: Google style for public functions and classes.

### TypeScript

- **Formatting**: Prettier (default settings)
- **Linting**: ESLint with recommended rules
- **Types**: Strict TypeScript — no `any` except in genuinely dynamic contexts. Prefer `unknown` and narrow.
- **Components**: Functional components with hooks. No class components.
- **State**: Use Zustand for global state. Avoid prop drilling beyond 2 levels.
- **Naming**: Components are `PascalCase`, hooks are `useCamelCase`, utilities are `camelCase`.

### Kotlin

- **Formatting**: `ktlint` with default Android Kotlin style
- **Coroutines**: Use structured concurrency. Never use `GlobalScope`. Scope coroutines to `viewModelScope` or `lifecycleScope`.
- **State**: Use `StateFlow` and `SharedFlow`. Expose read-only flows from ViewModels.
- **Compose**: Follow Compose guidelines — stateless composables where possible, state hoisting.
- **Naming**: Follow Kotlin conventions. Composables are `PascalCase`.

---

## Architecture Guidelines

### Cross-Language Changes

If your change affects the IPC protocol or WebSocket message format:

1. Update the JSON schema in `protocol/schemas/`.
2. Update the Rust types in `protocol/rust/src/lib.rs`.
3. Update the Python models in `protocol/python/pi_protocol/messages.py`.
4. Update the Kotlin data classes in `protocol/kotlin/PiProtocol.kt`.
5. Update [docs/IPC-PROTOCOL.md](docs/IPC-PROTOCOL.md).

All four must be kept in sync. A PR that changes one without the others will be rejected.

### Adding a New Tool

1. Create `src-tauri/src/tools/your_tool.rs` implementing the `Tool` trait.
2. Register it in `ToolRegistry::new()`.
3. Define its default permission tier.
4. Add permission rules for it in `safety/rules.rs` if needed.
5. Update [AGENTS.md](AGENTS.md) with the tool's documentation.
6. Write integration tests.

### Adding a New IPC Method

1. Add the method handler in `sidecar/src/pi_sidecar/ipc/handler.py`.
2. Register it in `RequestHandler._handlers`.
3. Add the corresponding Rust caller in `src-tauri/src/ipc/`.
4. Update the JSON schema in `protocol/schemas/ipc-message.schema.json`.
5. Write round-trip tests (Rust sends, Python responds).

---

## What We Will Not Accept

- Changes that bypass the permission engine or weaken the safety layer without explicit discussion.
- Dependencies with known critical vulnerabilities.
- Code without tests for non-trivial logic.
- Breaking changes to the IPC protocol without a migration path.
- Large refactors without prior discussion in an issue.

---

## License

By contributing to Pi-Assistant, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE).
