# Troubleshooting

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/)
[![Tauri](https://img.shields.io/badge/tauri-%2324C8DB.svg?style=for-the-badge&logo=tauri&logoColor=%23262626)](https://tauri.app/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com/)

Common issues and their solutions when developing or running Pi-Assistant.

---

## Build Issues

### Tauri build fails with missing system libraries

**Symptom:** Errors mentioning `webkit2gtk`, `libsoup`, `javascriptcoregtk`, or similar.

**Solution:** Install the platform-specific system libraries listed in [DEVELOPMENT.md](DEVELOPMENT.md#platform-specific-system-libraries).

```bash
# Ubuntu/Debian
sudo apt install -y libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev

# Fedora
sudo dnf install -y webkit2gtk4.1-devel
```

---

### `cargo build` fails with `rusqlite` or `sqlite-vec` errors

**Symptom:** Compilation errors related to SQLite bindings.

**Solution:** Ensure `rusqlite` is using the `bundled` feature (compiles SQLite from source). Check `src-tauri/Cargo.toml`:

```toml
rusqlite = { version = "0.32", features = ["bundled"] }
```

If you see linker errors about `sqlite3`, you may have a conflicting system SQLite. The `bundled` feature avoids this.

---

### `npm install` fails or is slow

**Symptom:** Dependency resolution errors or extremely slow installs.

**Solution:**

```bash
# Clear the cache and retry
rm -rf node_modules package-lock.json
npm install

# Or use a fresh lock file
npm cache clean --force
npm install
```

---

### Rust compile times are very long

**Symptom:** First build takes 10+ minutes.

**Solution:** This is normal for the first build — Tauri and its dependencies are large. Subsequent incremental builds are much faster.

To speed up development builds:

1. Use `cargo install sccache` and set `RUSTC_WRAPPER=sccache`.
2. In `src-tauri/Cargo.toml`, add a dev profile:

   ```toml
   [profile.dev]
   opt-level = 0
   debug = true

   [profile.dev.package."*"]
   opt-level = 2  # optimize dependencies but not your code
   ```

3. Use `mold` (Linux) or `lld` as the linker for faster linking:
   ```toml
   # .cargo/config.toml
   [target.x86_64-unknown-linux-gnu]
   rustflags = ["-C", "link-arg=-fuse-ld=mold"]
   ```

---

## Runtime Issues

### Python sidecar fails to start

**Symptom:** Error message: `Sidecar health check failed` or `Failed to capture sidecar stdout`.

**Causes and solutions:**

1. **Python not found:** Ensure `python3` is on your PATH, or set `PI_ASSISTANT_PYTHON`:

   ```bash
   export PI_ASSISTANT_PYTHON=/path/to/python3
   ```

2. **Missing Python dependencies:** Ensure the sidecar's virtual environment is set up:

   ```bash
   cd sidecar
   uv sync   # or: pip install -e .
   ```

3. **Virtual environment not activated (dev mode):** During development, the Rust core looks for `python3` on PATH and runs `python -m pi_sidecar`. If you installed deps into a venv, either:
   - Activate the venv before running `npm run tauri dev`
   - Set `PI_ASSISTANT_PYTHON` to `sidecar/.venv/bin/python`

4. **Port conflict or stderr noise:** Check stderr output from the sidecar. The Rust core inherits the sidecar's stderr, so its logs appear in the terminal where you ran `npm run tauri dev`.

---

### Python sidecar crashes mid-request

**Symptom:** Error: `Sidecar stdout stream ended — process likely exited`.

**Common causes:**

1. **Out of memory (OOM):** Loading large ML models can exceed available RAM. Check `dmesg` or system logs. Reduce model size or add swap space.

2. **CUDA error:** If using GPU, ensure CUDA and PyTorch CUDA versions match:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Unhandled exception:** The sidecar logs exceptions to stderr. Check the terminal output for Python tracebacks.

The Rust core will attempt to detect the crash and can restart the sidecar automatically.

---

### WebSocket connection fails from Android

**Symptom:** Android app shows "Connection Error" or fails to connect.

**Checklist:**

1. **Same network:** Both devices must be on the same local network (WiFi).

2. **Correct IP:** The server IP shown in the Android settings must match the desktop machine's LAN IP:

   ```bash
   # Linux
   ip addr show | grep "inet " | grep -v 127.0.0.1

   # macOS
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

3. **Firewall:** Ensure port 9120 is open:

   ```bash
   # Linux (ufw)
   sudo ufw allow 9120/tcp

   # Linux (firewalld)
   sudo firewall-cmd --add-port=9120/tcp --permanent
   sudo firewall-cmd --reload
   ```

4. **Server running:** The WebSocket server only starts when the desktop app is running. Verify with:

   ```bash
   curl -s http://localhost:9120/ws
   # Should return an HTTP 426 (Upgrade Required) — this means the server is running
   ```

5. **Auth token:** Ensure you completed the pairing process and the token is valid.

---

### Permission dialog not appearing

**Symptom:** Agent appears stuck in "Paused" state but no dialog shows.

**Causes:**

1. **Frontend event listener not registered:** Ensure `usePermission` hook is mounted in the component tree.
2. **Mobile client disconnected:** If the user is on mobile and the WebSocket drops, the permission request is lost. Reconnect and the latest pending request will be re-sent.
3. **Browser window minimized:** Tauri events are still delivered, but the user may not see the dialog.

---

### Agent loop doesn't start

**Symptom:** Clicking "Start" does nothing. Agent stays in "Idle" state.

**Checklist:**

1. **Check the Rust logs** for errors. Run with `PI_ASSISTANT_LOG_LEVEL=debug`.
2. **Python sidecar must be running.** The agent loop calls the planner via IPC on the first iteration. If the sidecar isn't ready, the loop will error.
3. **Check for panics** in the terminal output. A panic in the spawned tokio task will silently fail.

---

### Memory/vector search returns no results

**Symptom:** The agent doesn't seem to remember previous context.

**Causes:**

1. **No embeddings stored:** Check if the `memory_vectors` table has entries:

   ```bash
   sqlite3 ~/.pi-assistant/data/pi-assistant.db "SELECT COUNT(*) FROM memory_vectors;"
   ```

2. **Embedding model not loaded:** The Python sidecar lazy-loads the embedding model on first use. The first `inference.embed` call downloads `all-MiniLM-L6-v2` (~80 MB). This requires internet access.

3. **sqlite-vec not loaded:** The extension must be loaded into the SQLite connection at startup. Check Rust logs for `sqlite-vec` initialization messages.

---

## Data and Storage

### Where is data stored?

| Data                 | Default Location                       |
| -------------------- | -------------------------------------- |
| SQLite database      | `~/.pi-assistant/data/pi-assistant.db` |
| ML models            | `~/.pi-assistant/models/`              |
| Rust build artifacts | `src-tauri/target/`                    |
| Node modules         | `node_modules/`                        |
| Python venv          | `sidecar/.venv/`                       |
| Android build        | `android/app/build/`                   |

### How to reset all data

```bash
rm -rf ~/.pi-assistant/
```

This deletes the database, all stored memories, and cached models. The app will recreate the directory structure on next launch.

### How to reset just the database

```bash
rm ~/.pi-assistant/data/pi-assistant.db
```

The schema is recreated automatically on next launch.

---

## Performance Issues

### High CPU usage during idle

**Cause:** The agent loop should not be running when idle. If CPU is high, check if the Python sidecar is stuck in a computation. Kill and restart:

```bash
pkill -f pi_sidecar
```

The Rust core will restart it automatically.

### Large memory usage

**Causes:**

- ML models loaded into RAM. `all-MiniLM-L6-v2` uses ~300 MB. A full GPT-2 model uses ~500 MB. Larger models require proportionally more.
- Headless Chrome instance (if browser tool is active) uses ~200-500 MB.
- SQLite with many vector embeddings can grow. Check database size:
  ```bash
  du -h ~/.pi-assistant/data/pi-assistant.db
  ```

---

## Getting Help

If your issue isn't listed here:

1. Search the [GitHub Issues](https://github.com/ACFHarbinger/Pi-Assistant/issues).
2. Run with `PI_ASSISTANT_LOG_LEVEL=debug` and include the log output.
3. Open a new issue with:
   - OS and version
   - Rust, Python, Node versions
   - Steps to reproduce
   - Full error output
