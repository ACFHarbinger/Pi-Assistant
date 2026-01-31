# Tutorial: Getting Started with Pi-Assistant

A step-by-step walkthrough from first launch to running your first autonomous agent task.

---

## Part 1: First Launch

### 1.1 Build and Start

After completing the setup in [DEVELOPMENT.md](DEVELOPMENT.md):

```bash
cd Pi-Assistant
npm run tauri dev
```

The desktop window opens. You'll see:

- **Agent Status**: A badge showing "Idle" (grey).
- **Chat Interface**: A message area with an input field at the bottom.
- **Task Panel**: An empty task queue on the side.

The terminal where you ran the command shows Rust logs. Look for:

```
[INFO] Pi-Assistant starting
[INFO] Starting Python sidecar: python3 -m pi_sidecar
[sidecar] Pi-Assistant sidecar starting
[INFO] Python sidecar is healthy
[INFO] WebSocket server listening on 0.0.0.0:9120
```

If you see all four lines, everything is running.

### 1.2 Understanding the UI

```
┌─────────────────────────────────────────────────────┐
│  Pi-Assistant                           [─] [□] [x] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Agent Status │  │                             │  │
│  │             │  │       Chat Interface         │  │
│  │  ● Idle     │  │                             │  │
│  │             │  │  Messages appear here...    │  │
│  │ ─────────── │  │                             │  │
│  │             │  │                             │  │
│  │ Task Queue  │  │                             │  │
│  │             │  │                             │  │
│  │ (empty)     │  │                             │  │
│  │             │  │  ┌─────────────────────┐    │  │
│  │             │  │  │ Type a message...   │    │  │
│  └─────────────┘  └──┴─────────────────────┴────┘  │
│                                                     │
│  [▶ Start]  [⏸ Pause]  [⏹ Stop]                    │
└─────────────────────────────────────────────────────┘
```

---

## Part 2: Your First Agent Task

### 2.1 Starting a Task

Type a task description in the chat input:

```
Create a Python script that fetches the top 10 Hacker News stories and saves them to a JSON file.
```

Click **Start** (or press Enter). The agent status changes to **Running** (green).

### 2.2 Watching the Agent Work

The agent loop begins iterating. In the chat interface, you'll see the agent's reasoning and actions:

```
Agent: I'll create a Python script to fetch HN stories.
       First, let me check what's in the current directory.

[Tool: shell] ls -la
→ total 12
→ drwxr-xr-x 2 user user 4096 ...
→ -rw-r--r-- 1 user user  661 ... LICENSE
→ ...

Agent: The directory is mostly empty. I'll create the script.

[Tool: code] Write: hn_stories.py
```

### 2.3 Permission Prompts

When the agent tries to write a file, a **Permission Dialog** appears:

```
┌──────────────────────────────────────────┐
│  Permission Request                      │
│                                          │
│  The agent wants to:                     │
│  Write file: ./hn_stories.py             │
│                                          │
│  ☐ Always allow this pattern             │
│                                          │
│  [Allow]              [Deny]             │
└──────────────────────────────────────────┘
```

Click **Allow** to let the agent proceed. If you check "Always allow this pattern," future writes to this directory won't prompt again.

### 2.4 Agent Questions

Sometimes the agent needs clarification. The status changes to **Paused** (yellow):

```
Agent: Should I use the official HN API (https://hacker-news.firebaseio.com)
       or scrape the website directly? The API is more reliable but requires
       multiple requests.
```

Type your response:

```
Use the official API.
```

The agent resumes.

### 2.5 Task Completion

When the agent finishes, the status changes to **Stopped (Completed)** (grey):

```
Agent: Done. I've created hn_stories.py that:
       1. Fetches top story IDs from the HN API
       2. Fetches details for the top 10
       3. Saves them to hn_top_stories.json

       The script is ready to run: python hn_stories.py
```

---

## Part 3: Understanding Permission Tiers

The safety system protects your computer. Here's what to expect:

### Auto-Approved (No Prompt)

These run silently:

```
ls, cat, head, tail, grep, find (no -exec)
git status, git log, git diff, git show
npm list, pip list, cargo tree
uname, whoami, date, pwd
```

### Requires Approval (Dialog Appears)

You'll be asked to approve:

```
cp, mv, mkdir, touch, file writes
git commit, git push, git checkout
npm install, pip install
curl, wget
python, node, cargo run
```

### Blocked (Silently Denied)

These never execute:

```
sudo, su, rm -rf /, dd, mkfs
chmod 777, chown root
Editing /etc/, /sys/, /proc/
```

If the agent tries a blocked command, you'll see it logged in the chat:

```
[Denied] sudo apt install something — Privilege escalation via sudo
```

---

## Part 4: Connecting the Android App

### 4.1 Install the App

```bash
cd android
./gradlew installDebug
```

### 4.2 Find Your Desktop IP

On your desktop:

```bash
# Linux
hostname -I | awk '{print $1}'

# macOS
ipconfig getifaddr en0
```

### 4.3 Configure the Connection

1. Open the Pi-Assistant app on your Android device.
2. Go to **Settings**.
3. Enter the server URL: `ws://YOUR_DESKTOP_IP:9120/ws`
4. The desktop app shows a **6-digit pairing code**. Enter it on the Android app.
5. The connection state changes to **Connected** (green dot).

### 4.4 Using the Mobile App

The mobile app mirrors the desktop chat interface:

- **Send text commands**: Type in the chat input.
- **Send voice commands**: Tap the microphone button, speak, and the transcribed text is sent to the agent.
- **Approve permissions**: When the agent needs approval, a notification/dialog appears on the mobile app too.
- **Monitor status**: The agent's state (Idle/Running/Paused/Stopped) is shown in real time.

Both desktop and mobile can send commands simultaneously. The agent processes them in order.

---

## Part 5: Using Memory

### 5.1 How Memory Works

Everything the agent does is stored:

- Your messages
- Agent responses
- Tool calls and their results
- Task descriptions and outcomes

This data is searchable in two ways:

- **Recency**: The last N messages from the current session.
- **Semantic similarity**: The agent can find relevant context from any past session by meaning.

### 5.2 Browsing Memory

Click the **Memory** tab in the desktop UI to search stored memories:

```
Search: "Hacker News script"

Results:
─ [2026-01-31] Task: Create a Python script that fetches the top 10 HN stories...
─ [2026-01-31] Tool: shell → ls -la (exit code 0)
─ [2026-01-31] Tool: code → Write hn_stories.py (82 lines)
```

### 5.3 Cross-Session Context

Start a new task in a new session:

```
Update the Hacker News script to also include comment counts.
```

The agent retrieves the previous session's context via vector search:

```
Agent: I found the hn_stories.py script from a previous session.
       Let me read it and add comment count support.

[Tool: code] Read: ./hn_stories.py
```

The agent remembers because the previous task's description and tool results were embedded and stored in the vector database.

---

## Part 6: Training a Model

### 6.1 Starting a Training Run

Training is initiated through the agent or directly via the UI. The agent can decide to fine-tune a model when the task requires it, or you can request it explicitly:

```
Fine-tune a small model on the conversation data from today's sessions.
```

### 6.2 Monitoring Progress

During training, the Python sidecar streams progress updates:

```
Training Progress:
  Model: my-finetuned-llm
  Epoch: 3/10
  Loss: 0.342
  Accuracy: 0.89
  ████████░░ 30%
```

Progress appears in both the desktop UI and the mobile app via the same event system.

### 6.3 Model Management

Trained models are saved with version numbers in `~/.pi-assistant/models/`:

```
~/.pi-assistant/models/
  my-finetuned-llm/
    v1/
    v2/
    current -> v2/     (symlink to active version)
```

---

## Part 7: Advanced Usage

### 7.1 Custom Permission Rules

You can configure custom permission rules through the Settings panel:

- **Add allow pattern**: `^cargo test` — auto-approve running Rust tests.
- **Add block pattern**: `^curl.*external-api.com` — block requests to a specific domain.

Patterns are PCRE-style regular expressions matched against the full command string.

### 7.2 Iteration Limits

When starting a task, you can set a maximum iteration count:

```
Max iterations: 50
```

The agent stops after 50 iterations even if the task isn't complete. This prevents runaway loops on ambiguous tasks.

### 7.3 Browser Tool

The agent can browse the web via a headless Chrome instance:

```
Research the latest Rust async patterns and summarize them.
```

The browser tool navigates to pages, extracts text, and returns content to the agent. By default, only `localhost` is allowed. Configure additional domains in Settings to allow external access.

### 7.4 Continuous Mode vs Single Step

- **Continuous Mode** (default): The agent iterates until the task is complete or stopped.
- **Single Step**: Set max iterations to 1. The agent performs one planning + execution cycle and stops. Useful for debugging or reviewing each step individually.

---

## What's Next

- Read [AGENTS.md](AGENTS.md) for deep technical details on the agent loop and tool system.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design with code snippets.
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you run into issues.
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute to the project.
