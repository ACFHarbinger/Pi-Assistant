# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Pi-Assistant, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, report via one of these channels:

1. **GitHub Security Advisories**: Use the [private vulnerability reporting](https://github.com/ACFHarbinger/Pi-Assistant/security/advisories/new) feature.
2. **Email**: Send details to the repository maintainer (see GitHub profile for contact info).

### What to Include

- Description of the vulnerability.
- Steps to reproduce.
- Affected components (Rust core, Python sidecar, Android client, etc.).
- Potential impact.
- Suggested fix, if you have one.

### Response Timeline

| Action             | Target                             |
| ------------------ | ---------------------------------- |
| Acknowledgment     | Within 48 hours                    |
| Initial assessment | Within 1 week                      |
| Fix or mitigation  | Within 30 days for critical issues |

---

## Security Architecture

Pi-Assistant is an application that grants an AI agent controlled access to computer operations. Security is enforced at multiple layers:

### 1. Permission Engine (Primary Defense)

All agent tool calls pass through a three-tier permission system:

- **Auto-approve**: Read-only operations (no side effects).
- **Ask user**: Mutations require explicit user approval via UI dialog.
- **Block**: Dangerous operations are always rejected.

See [docs/SAFETY-MODEL.md](docs/SAFETY-MODEL.md) for full details.

### 2. Path Restrictions

File access is restricted:

- System directories (`/etc`, `/sys`, `/proc`, etc.) are blocked.
- Sensitive user directories (`~/.ssh`, `~/.gnupg`, `~/.aws`) are blocked.
- Path traversal (`..`) is rejected.
- All paths are canonicalized to prevent symlink escapes.

### 3. Environment Sanitization

Environment variables matching credential patterns (`*_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`) are stripped from the Python sidecar's environment.

### 4. Network Isolation

The headless browser is restricted to a domain allowlist (default: `localhost` only). External network access requires explicit user configuration.

### 5. Process Isolation

- Shell commands have a 60-second timeout.
- The Python sidecar is spawned with `kill_on_drop(true)`.
- All child processes are terminated when the application exits.

### 6. Local-Only Mobile Access

The WebSocket server for mobile clients binds to the local network. It uses token-based authentication with a pairing code mechanism. No internet-exposed endpoints.

---

## Known Limitations

| Limitation                             | Description                                                                                                        | Mitigation                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Regex-based command filtering**      | Sophisticated command obfuscation (e.g., base64-encoded payloads) could bypass pattern matching.                   | Default-deny policy means unrecognized commands require user approval.                               |
| **No filesystem sandboxing (yet)**     | Shell commands run with the user's full permissions.                                                               | Permission engine blocks known dangerous commands. User approval required for mutations.             |
| **Python sidecar shares user context** | The sidecar runs as the same user. A compromised model could theoretically be prompted to exploit the IPC channel. | The Rust core validates all IPC responses. Tool execution always goes through the permission engine. |
| **Local network WebSocket**            | Anyone on the same LAN who obtains the pairing code can connect.                                                   | Token-based auth with pairing flow. Consider adding mTLS for high-security environments.             |

---

## Supported Versions

| Version          | Supported |
| ---------------- | --------- |
| 0.1.x (upcoming) | Yes       |

Security updates will be applied to the latest release and the `main` branch.

---

## Dependencies

We monitor dependencies for known vulnerabilities:

```bash
# Rust
cargo audit

# Python
pip-audit

# TypeScript
npm audit

# Kotlin
# Via OWASP dependency-check Gradle plugin
```

Dependency updates for security patches are prioritized and released promptly.
