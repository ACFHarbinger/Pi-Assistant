# Pi-Assistant: Moltbot Feature Parity Roadmap

This document outlines the implementation plan to bring Pi-Assistant up to feature parity with OpenClaw/Moltbot, a mature personal AI assistant with multi-channel messaging, proactive automation, voice interaction, and a distinctive personality.

---

## Executive Summary

**Goal**: Upgrade Pi-Assistant with Moltbot's best capabilities while preserving our local-first, Rust+Python+React architecture.

**Status**: All 8 implementation phases are **complete**. Core feature parity with Moltbot has been achieved. All high-priority items are now implemented: Telegram media handling with auto-transcription, push-to-talk voice input, and webhook receiver (already existed at `POST /webhook` on port 8080). Remaining work is limited to polish items: Discord slash commands, bundled skill packs, and conversation mode.

---

## Feature Gap Analysis

| Feature | OpenClaw/Moltbot | Pi-Assistant | Status |
|---------|------------------|--------------|--------|
| **Personality/Hatching** | Molty the space lobster | `soul.md` + `personality.py` | 🟢 Complete |
| **Messaging Channels** | WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Matrix, Teams | Android WebSocket, Telegram (`teloxide`), Discord (`serenity`) | 🟢 3 channels live |
| **Proactive Automation** | Cron jobs, webhooks, Gmail Pub/Sub | Cron scheduler + webhook endpoint (port 8080) | 🟢 Complete |
| **Voice Interaction** | Voice Wake, Talk Mode, ElevenLabs TTS | Vosk wake word, ElevenLabs TTS, Whisper STT | 🟢 Complete |
| **Live Canvas (A2UI)** | Agent-driven visual workspace | `Canvas.tsx` + `canvas.rs` tool | 🟢 Complete |
| **Multi-Agent Routing** | Route channels to isolated agents | `AgentPool` + `SessionTool` | 🟢 Complete |
| **Skills Platform** | 50+ bundled skills | SKILL.md loader, workspace + global paths | 🟡 Infra done, no bundled skills yet |
| **Browser Control** | CDP-based | chromiumoxide | 🟢 Comparable |
| **Memory System** | Sessions + semantic search | SQLite + sqlite-vec | 🟢 Comparable |

---

## Implementation Phases

### Phase 13: Personality & Hatching — ✅ Complete
**Priority: High** | **Effort: 1 week**

Added humanlike personality and first-run "hatching" experience.

**Implemented Files:**
- `soul.md` — Personality definition (traits, communication style, hatching message, boundaries)
- `sidecar/src/pi_sidecar/personality.py` — Loads `soul.md`, injects into system prompt, extracts hatching message

**Implemented Features:**
- [x] Configurable personality via `soul.md`
- [x] Hatching message on first encounter
- [x] Personality-aware response generation via sidecar injection
- [x] Agent name extraction and update capability

**Not Implemented:**
- [ ] `docs/PERSONALITY.md` — Configuration guide (not created)
- [ ] First-run hatching animation in `src/App.tsx`
- [ ] "Hatched" state tracking in memory module

---

### Phase 14: Telegram Bot Integration — ✅ Complete
**Priority: Critical** | **Effort: 2 weeks**

**Implemented Files:**
- `src-tauri/src/channels/mod.rs` — Channel abstraction trait with `MediaAttachment` and `MediaType` types
- `src-tauri/src/channels/telegram.rs` — Full Telegram implementation with media handling

**Dependencies:**
- `teloxide = "0.14"` (upgraded from planned `0.12`)

**Implemented Features:**
- [x] Text message handling with user ID and display name
- [x] Allowlist-based security with authorized user checks
- [x] Markdown V2 response formatting
- [x] Reply-to message support
- [x] Config-driven auto-start at application launch
- [x] Graceful shutdown via signal handling
- [x] Photo message handling (downloads largest resolution, attaches path)
- [x] Voice message handling (downloads OGG, auto-transcribes via Whisper STT)
- [x] Audio file handling (downloads MP3, auto-transcribes via Whisper STT)
- [x] Document handling (downloads with original extension, attaches path)
- [x] Caption support for media messages
- [x] Media files saved to `~/.pi-assistant/media/telegram/`

**Not Implemented:**
- [ ] Pairing codes for unknown senders
- [ ] Video message handling

---

### Phase 15: Discord Bot Integration — ✅ Complete
**Priority: High** | **Effort: 2 weeks**

**Implemented Files:**
- `src-tauri/src/channels/discord.rs` — Full Discord implementation

**Dependencies:**
- `serenity = "0.12"`

**Implemented Features:**
- [x] DM and guild message support
- [x] Mention-based activation (strips mentions from message text)
- [x] Bot identification — only responds to non-bot messages
- [x] Gateway intents: `GUILD_MESSAGES`, `DIRECT_MESSAGES`, `MESSAGE_CONTENT`
- [x] Config-driven auto-start at application launch
- [x] Shard manager for connection lifecycle

**Not Implemented:**
- [ ] Slash commands

---

### Phase 16: Proactive Automation — ✅ Complete
**Priority: High** | **Effort: 2 weeks**

**Implemented Files:**
- `src-tauri/src/cron/mod.rs` — CronManager with job CRUD and persistence
- `src-tauri/src/tools/cron.rs` — Agent-callable CronTool (add, remove, list)
- `src-tauri/src/ws/server.rs` — Webhook endpoint at `POST /webhook` on port 8080

**Dependencies:**
- `tokio-cron-scheduler = "0.10"`
- `axum` (workspace) — HTTP server

**Implemented Features:**
- [x] Standard cron expression scheduling
- [x] Job creation, removal, and listing via agent tool
- [x] Job persistence to `cron.json` in config directory
- [x] Automatic job reload on startup
- [x] Agent receives scheduled tasks as `ChatMessage` commands
- [x] HTTP webhook receiver (`POST /webhook` on port 8080)
- [x] Webhook supports `is_chat` mode (chat message) and task mode (agent start)
- [x] Webhook accepts `provider` and `model_id` parameters

**Not Implemented:**
- [ ] Timezone-aware scheduling
- [ ] Webhook signature verification for security

---

### Phase 17: Voice Interaction — ✅ Complete
**Priority: High** | **Effort: 3 weeks**

**Implemented Files:**
- `src-tauri/src/voice/mod.rs` — VoiceManager orchestrating recording + wake detection
- `src-tauri/src/voice/wake.rs` — WakeWordDetector using Vosk
- `src-tauri/src/voice/audio.rs` — AudioRecorder using cpal (PulseAudio/PipeWire)
- `sidecar/src/pi_sidecar/tts/elevenlabs.py` — ElevenLabs TTS integration
- `sidecar/src/pi_sidecar/stt/whisper.py` — Whisper STT via `faster_whisper`
- `src/components/VoicePanel.tsx` — Frontend voice UI

**Dependencies:**
- `vosk = "0.3"` — Wake word detection
- `cpal = "0.15"` — Cross-platform audio capture
- `hound = "3.5"` — Audio file I/O
- `elevenlabs` (Python) — TTS API
- `faster_whisper` (Python) — STT

**Implemented Features:**
- [x] Wake phrase detection ("Hey Pi" / "Ok Pi") via Vosk
- [x] Background audio listener loop (500ms polling) with cancellation support
- [x] ElevenLabs TTS with configurable voice (default: "Rachel")
- [x] Whisper STT with model size selection (base → large-v3)
- [x] Voice UI with start/stop, status display, animated indicator
- [x] F32 and I16 sample format support
- [x] Tauri command integration (`start_voice_listener` / `stop_voice_listener`)
- [x] Proper `stop_voice_listener` with `CancellationToken` — cleanly stops background task and releases audio stream
- [x] Push-to-talk mode — hold-to-record with automatic STT transcription
- [x] Push-to-talk Tauri commands (`push_to_talk_start` / `push_to_talk_stop`)
- [x] Audio saved as WAV to `~/.pi-assistant/media/voice/` for STT processing
- [x] VoicePanel UI with dual controls: wake word toggle + hold-to-talk button
- [x] Visual states: idle, listening (wake word), recording (PTT), processing (STT)

**Prerequisites:**
- Vosk model must be downloaded to `~/.pi-assistant/voice/vosk-model-small-en-us-0.15/`
- ElevenLabs API key required for TTS (set via environment variable)

**Not Implemented:**
- [ ] Continuous conversation mode (talk mode)

---

### Phase 18: Live Canvas (A2UI) — ✅ Complete
**Priority: Medium** | **Effort: 3 weeks**

**Implemented Files:**
- `src/components/Canvas.tsx` — Floating visual workspace with iframe sandbox
- `src-tauri/src/tools/canvas.rs` — Agent-callable canvas tool (push, clear)

**Implemented Features:**
- [x] Agent pushes HTML/React content to canvas via `canvas-push` event
- [x] Clear/reset canvas via `canvas-clear` event
- [x] Snapshot/download as HTML file
- [x] Fullscreen toggle
- [x] Sandboxed iframe rendering for security
- [x] Status indicator with pulse animation

**Not Implemented:**
- [ ] JavaScript eval action from agent
- [ ] Canvas state persistence across restarts

---

### Phase 19: Multi-Agent Routing — ✅ Complete
**Priority: Medium** | **Effort: 2 weeks**

**Implemented Files:**
- `src-tauri/src/agent/pool.rs` — AgentPool managing multiple independent agent instances
- `src-tauri/src/tools/sessions.rs` — SessionTool (list, create, remove, route)

**Implemented Features:**
- [x] Multiple agents with isolated loops, state, and command channels
- [x] Channel-to-agent routing via RoutingConfig
- [x] Default agent fallback
- [x] Shared resources (tool registry, memory, sidecar, permissions)
- [x] Dynamic agent creation and removal via tool
- [x] Agent listing

**Not Implemented:**
- [ ] Direct agent-to-agent communication

---

### Phase 20: Skills Platform Enhancement — ✅ Complete (Infrastructure)
**Priority: Medium** | **Effort: 2 weeks**

**Implemented Files:**
- `src-tauri/src/skills/mod.rs` — SkillManager with YAML frontmatter parsing

**Implemented Features:**
- [x] SKILL.md file format with YAML frontmatter (name, description, version)
- [x] Recursive directory traversal for skill discovery
- [x] Dual skill paths: `.agent/skills/` (workspace) and `~/.pi-assistant/skills/` (global)
- [x] Auto-load at startup with error resilience
- [x] Version tracking
- [x] Fallback to directory name if no frontmatter present

**Not Implemented:**
- [ ] Bundled skills shipped with the application
- [ ] Skill marketplace or install mechanism

---

## Dependencies Summary

### Rust Crates (Actual)
```toml
[dependencies]
teloxide = "0.14"              # Telegram
serenity = "0.12"              # Discord
tokio-cron-scheduler = "0.10"  # Cron
cpal = "0.15"                  # Audio capture
vosk = "0.3"                   # Wake word detection
hound = "3.5"                  # Audio file I/O
```

### Python Packages (Actual)
```toml
[project.optional-dependencies]
voice = ["elevenlabs", "faster_whisper"]
```

---

## Verification Plan

### Automated Tests
```bash
# Channel tests
cargo test channels::

# TTS integration tests
uv run pytest sidecar/tests/test_tts.py

# Frontend component tests
npm run test
```

### Integration Tests
1. Send Telegram message → agent responds with personality
2. Schedule cron job → executes on time
3. Say "Hey Pi" → agent activates voice mode
4. Agent pushes content to Canvas

---

## Success Metrics

- [x] Agent greets user with personality on first run (hatching)
- [x] 3+ messaging channels supported (Android, Telegram, Discord)
- [x] Cron jobs executing scheduled tasks
- [x] Voice wake detection on Linux
- [x] Skills installable from workspace

---

## Resolved Questions

1. **Wake phrase**: "Hey Pi" and "Ok Pi" — hardcoded in Vosk detector, not yet user-configurable.
2. **Personality**: Shipped default via `soul.md` in the repository root. User can edit it.
3. **TTS voice**: ElevenLabs "Rachel" voice (ID: `pNInz6ovAn45no7UM98t`) with configurable stability/similarity settings.
4. **Canvas persistence**: Not implemented — canvas resets on restart.

---

## Remaining Work

The following items were scoped in the original roadmap but are not yet implemented:

### Medium Priority
| Item | Phase | Description |
|------|-------|-------------|
| Discord slash commands | 15 | Native Discord command integration |
| Hatching animation | 13 | First-run visual sequence in `src/App.tsx` |
| Canvas eval action | 18 | Execute JS in canvas from agent |
| Agent-to-agent communication | 19 | Cross-agent messaging |
| Webhook signature verification | 16 | HMAC/secret-based webhook auth |

### Low Priority
| Item | Phase | Description |
|------|-------|-------------|
| Bundled skills | 20 | Ship default skill packs with the application |
| Personality docs | 13 | `docs/PERSONALITY.md` configuration guide |
| Canvas persistence | 18 | Save/restore canvas state across restarts |
| Timezone-aware cron | 16 | TZ support for cron expressions |
| Continuous conversation mode | 17 | Multi-turn voice without re-triggering wake word |
| Telegram video handling | 14 | Download and process video messages |
| Telegram pairing codes | 14 | Authorization codes for unknown senders |
