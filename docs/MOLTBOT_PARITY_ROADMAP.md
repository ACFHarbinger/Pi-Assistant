# Pi-Assistant: Moltbot Feature Parity Roadmap

This document outlines the implementation plan to bring Pi-Assistant up to feature parity with OpenClaw/Moltbot, a mature personal AI assistant with multi-channel messaging, proactive automation, voice interaction, and a distinctive personality.

---

## Executive Summary

**Goal**: Upgrade Pi-Assistant with Moltbot's best capabilities while preserving our local-first, Rust+Python+React architecture.

**Timeline**: ~4-5 months for full parity, or ~6-8 weeks for focused MVP (Personality + Telegram + Cron + Voice).

---

## Feature Gap Analysis

| Feature | OpenClaw/Moltbot | Pi-Assistant | Status |
|---------|------------------|--------------|--------|
| **Personality/Hatching** | Molty the space lobster 🦞 | None | 🔴 Missing |
| **Messaging Channels** | WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Matrix, Teams | Android WebSocket only | 🔴 Critical |
| **Proactive Automation** | Cron jobs, webhooks, Gmail Pub/Sub | None | 🔴 Missing |
| **Voice Interaction** | Voice Wake, Talk Mode, ElevenLabs TTS | None | 🔴 Missing |
| **Live Canvas (A2UI)** | Agent-driven visual workspace | None | 🔴 Missing |
| **Multi-Agent Routing** | Route channels to isolated agents | Single agent | 🟡 Partial |
| **Skills Platform** | 50+ bundled skills | MCP tools only | 🟡 Partial |
| **Browser Control** | CDP-based | chromiumoxide | 🟢 Comparable |
| **Memory System** | Sessions + semantic search | SQLite + sqlite-vec | 🟢 Comparable |

---

## Implementation Phases

### Phase 13: Personality & Hatching
**Priority: High** | **Effort: 1 week**

Add humanlike personality and first-run "hatching" experience.

**New Files:**
- `soul.md` — Agent personality definition (voice, quirks, communication style)
- `docs/PERSONALITY.md` — Personality configuration guide

**Modifications:**
- `sidecar/src/pi_sidecar/inference/engine.py` — Inject soul.md into system prompt
- `src/App.tsx` — First-run hatching animation sequence
- `src-tauri/src/memory/mod.rs` — Track "hatched" state

**Features:**
- Configurable personality via `soul.md`
- First-run animation where agent introduces itself
- Personality-aware response generation
- Memorable agent identity

---

### Phase 14: Telegram Bot Integration
**Priority: Critical** | **Effort: 2 weeks**

**New Files:**
- `src-tauri/src/channels/mod.rs` — Channel abstraction trait
- `src-tauri/src/channels/telegram.rs` — Telegram implementation

**Dependencies:**
- `teloxide` — Rust Telegram Bot API

**Features:**
- Text, image, audio message handling
- Transcription of voice messages
- Pairing codes for unknown senders
- Allowlist-based security

---

### Phase 15: Discord Bot Integration
**Priority: High** | **Effort: 2 weeks**

**New Files:**
- `src-tauri/src/channels/discord.rs` — Discord implementation

**Dependencies:**
- `serenity` — Rust Discord API

**Features:**
- DM and guild message support
- Slash commands
- Mention-based activation in groups

---

### Phase 16: Proactive Automation
**Priority: High** | **Effort: 2 weeks**

**New Files:**
- `src-tauri/src/cron/mod.rs` — Cron scheduler
- `src-tauri/src/webhooks/mod.rs` — Webhook receiver

**Dependencies:**
- `tokio-cron-scheduler` — Cron scheduling

**Features:**
- Agent can create/modify cron jobs
- HTTP webhooks trigger agent tasks
- Timezone-aware scheduling
- Signature verification for security

---

### Phase 17: Voice Interaction
**Priority: High** | **Effort: 3 weeks**

**New Files:**
- `src-tauri/src/voice/wake.rs` — Wake word detection
- `sidecar/src/pi_sidecar/tts/elevenlabs.py` — TTS API
- `src/components/VoicePanel.tsx` — Voice UI

**Dependencies:**
- `vosk` or `sherpa-onnx-rs` — Local wake word detection
- `cpal` — Audio capture
- `elevenlabs` (Python) — TTS API

**Features:**
- Configurable wake phrase ("Hey Pi")
- Continuous conversation mode
- ElevenLabs voice synthesis
- Push-to-talk option

---

### Phase 18: Live Canvas (A2UI)
**Priority: Medium** | **Effort: 3 weeks**

**New Files:**
- `src/components/Canvas.tsx` — Visual workspace
- `src-tauri/src/tools/canvas.rs` — Canvas tools

**Features:**
- Agent pushes HTML/React to canvas
- Snapshot/reset/eval commands
- Rich output beyond text

---

### Phase 19: Multi-Agent Routing
**Priority: Medium** | **Effort: 2 weeks**

**Modifications:**
- `src-tauri/src/agent.rs` — Multiple agent instances

**New Files:**
- `src-tauri/src/tools/sessions.rs` — Session tools

**Features:**
- Isolated workspaces per agent
- Route channels to specific agents
- Agent-to-agent communication

---

### Phase 20: Skills Platform Enhancement
**Priority: Medium** | **Effort: 2 weeks**

**New Format:**
- `skills/*/SKILL.md` — Skill definition with YAML frontmatter

**Features:**
- Bundled and workspace skills
- Auto-load at startup
- Skill versioning

---

## Dependencies Summary

### Rust Crates
```toml
[dependencies]
teloxide = "0.12"          # Telegram
serenity = "0.12"          # Discord
tokio-cron-scheduler = "0.10"
cpal = "0.15"              # Audio
# vosk or sherpa-onnx-rs for wake word
```

### Python Packages
```toml
[project.optional-dependencies]
voice = ["elevenlabs"]
```

---

## Verification Plan

### Automated Tests
```bash
# New channel tests
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

- [ ] Agent greets user with personality on first run (hatching)
- [ ] 3+ messaging channels supported (Android, Telegram, Discord)
- [ ] Cron jobs executing scheduled tasks
- [ ] Voice wake detection on macOS/Linux
- [ ] Skills installable from workspace

---

## Open Questions

1. **Wake phrase**: "Hey Pi" or user-configurable?
2. **Personality**: Shipped default or user-defined from start?
3. **TTS voice**: ElevenLabs model selection?
4. **Canvas persistence**: Save state across restarts?

---

## Recommended Execution Order

1. **Phase 13** — Personality gives agent an identity
2. **Phase 14** — Telegram enables mobile reach
3. **Phase 16** — Cron enables proactive behavior
4. **Phase 17** — Voice makes it feel alive
5. **Phases 15, 18-20** — Polish and expansion
