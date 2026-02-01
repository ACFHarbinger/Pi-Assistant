# Pi-Assistant Roadmap

This roadmap outlines planned features and enhancements across the Pi-Assistant system. Items are organized by domain and roughly ordered by priority within each section. Dependencies between items are noted where relevant.

---

## 1. Agent Intelligence & Planning

### 1.1 Hierarchical Task Decomposition

The current single-pass planner produces a flat list of tool calls per iteration. Introduce a hierarchical planner that breaks complex tasks into subtask trees, where each subtask can itself be planned and executed independently. This enables the agent to tackle multi-step projects (e.g., "build a REST API with tests and documentation") without losing coherence across iterations.

- Subtask graph stored in memory with parent/child relationships
- Each subtask tracks its own state (pending, running, blocked, done)
- Planner can delegate subtasks to specialized sub-agents from the agent pool
- UI renders the task tree with collapsible nodes and progress indicators

### 1.2 Reflection & Self-Correction Loop

After each tool execution, add an explicit reflection step where the agent evaluates whether the result matches expectations. If the result diverges (e.g., a command fails, output is unexpected), the agent should revise its plan before continuing rather than blindly proceeding to the next tool call.

- Compare expected vs. actual tool output using lightweight classifier
- Maintain a per-task error budget: after N consecutive failures, pause and ask the user
- Log reflection reasoning alongside tool results for transparency

### 1.3 Cost-Aware Planning

When multiple LLM providers are available (local models, Anthropic API, OpenAI API), the planner should consider the cost and latency tradeoff for each planning step. Simple decisions (file reads, directory listings) can use a small local model. Complex reasoning (architecture decisions, debugging) can escalate to a larger cloud model.

- Token budget per task, configurable by user
- Automatic provider selection based on task complexity estimate
- Dashboard showing cumulative cost and token usage per session

### 1.4 Multi-Agent Collaboration

Extend the existing agent pool to support concurrent agents working on related subtasks. Agents share memory but maintain separate execution contexts. A coordinator agent assigns work and merges results.

- Shared memory namespace with per-agent read/write isolation
- Conflict resolution when two agents modify the same file
- Visual timeline showing parallel agent activity

---

## 2. Tool Expansion

### 2.1 Task-Specific ML Model Training

Expand the existing PyTorch Lightning training tool into a full workflow for training, evaluating, and deploying task-specific models that the agent can then use as specialized tools.

**Training Pipeline:**
- User provides a dataset (CSV, JSON, or a directory of files) and describes the task
- Agent selects an appropriate base model architecture (classification, sequence-to-sequence, regression, NER, etc.)
- Agent configures hyperparameters based on dataset size and task type
- Training runs in the Python sidecar with real-time metric streaming to the UI
- Checkpoints are saved to the model registry with metadata (task, accuracy, training config)

**Supported Task Types:**
- Text classification (sentiment, intent, topic)
- Named entity recognition
- Text summarization / paraphrasing
- Tabular regression and classification
- Image classification (with torchvision integration)
- Time-series forecasting

**Agent Integration:**
- Trained models register as new tools in the `ToolRegistry` automatically
- The planner can invoke trained models by name (e.g., `classify_support_ticket`, `predict_sales`)
- Model versioning: the agent can retrain and compare metrics across versions
- Active learning: the agent identifies low-confidence predictions and asks the user for labels

### 2.2 Retrieval-Augmented Generation (RAG) Tool

A dedicated tool that lets the agent ingest documents (PDFs, markdown, code repositories, web pages) into a searchable vector store and query them during planning.

- Chunking strategies: fixed-size, sentence-boundary, AST-aware (for code)
- Hybrid retrieval: combine vector similarity with BM25 keyword matching
- Source attribution: every retrieved chunk links back to its origin document and page/line
- Incremental indexing: add new documents without re-indexing the entire corpus

### 2.3 Database Tool

Direct SQL access to user-specified databases (SQLite, PostgreSQL, MySQL) for data exploration, querying, and modification.

- Read-only mode by default; writes require explicit permission approval
- Schema introspection: the agent can discover tables, columns, and relationships
- Query explanation: the agent can run `EXPLAIN` and interpret query plans
- Result visualization: tables rendered in the UI, with optional chart generation

### 2.4 API Integration Tool

A generic HTTP client tool that the agent can use to interact with external APIs.

- OpenAPI/Swagger spec ingestion: given a spec URL, the agent learns all available endpoints
- Authentication management: OAuth2 flows, API key storage (encrypted in permission-gated keyring)
- Rate limiting awareness: the agent respects `Retry-After` headers and backs off appropriately
- Response caching: avoid redundant API calls within the same task

### 2.5 Drawing & Diagram Tool

Generate and render diagrams (flowcharts, sequence diagrams, architecture diagrams, ER diagrams) using Mermaid or D2 syntax, rendered live in the Canvas.

- Agent can produce diagrams as part of its reasoning (e.g., "here's the architecture I'm proposing")
- Diagrams stored as versioned artifacts alongside task history
- User can request modifications ("add a cache layer between the API and database")

### 2.6 Containerized Execution Environment

Run agent tool calls inside isolated Docker containers for safety and reproducibility.

- Per-task container with configurable base image
- Filesystem snapshot before/after tool execution for diff visualization
- Network isolation modes: none, host-only, full
- Resource limits (CPU, memory, disk) configurable per task

---

## 3. Memory & Knowledge

### 3.1 Structured Knowledge Graph

Augment the current flat embedding store with a knowledge graph that captures relationships between entities the agent has encountered.

- Entities: files, functions, people, projects, concepts, tools
- Relations: `depends_on`, `authored_by`, `related_to`, `modified_in`, `calls`
- Graph stored in SQLite using adjacency list tables
- Graph queries inform the planner: "what files are related to the authentication module?"
- Visualization: interactive graph explorer in the UI (force-directed layout)

### 3.2 Episodic Memory with Summarization

Long-running sessions generate large volumes of messages and tool results. Introduce an episodic memory layer that summarizes completed task episodes into concise narrative summaries.

- After a task completes, generate a summary: what was asked, what was done, what was the outcome
- Summaries are embedded and searchable alongside raw messages
- Configurable retention policy: keep raw data for N days, keep summaries indefinitely
- Summary quality improves over time via user feedback (thumbs up/down on summary accuracy)

### 3.3 Markdown-Based Knowledge Files

Allow the agent to maintain a set of markdown files as a persistent, human-readable knowledge base.

- `~/.pi-assistant/knowledge/` directory organized by topic
- Agent creates and updates files as it learns (e.g., `projects/my-app.md`, `preferences/coding-style.md`)
- Files are indexed for vector search alongside database entries
- User can edit files directly; changes are picked up on next retrieval
- Agent references specific sections when explaining its reasoning: "Based on your coding preferences (see preferences/coding-style.md)..."

### 3.4 Conversation Branching & Replay

Allow users to branch a conversation at any point, explore an alternative approach, and optionally merge results back.

- Branch creates a snapshot of memory state at that point
- Each branch maintains independent tool execution history
- Merge operation combines tool results and memory from two branches
- UI shows branch tree with diff between branch outcomes

### 3.5 Cross-Session Learning

Use patterns from past sessions to improve future performance.

- Track which tool sequences successfully completed specific task types
- Build a task-type classifier from historical data
- Suggest proven tool sequences to the planner for similar new tasks
- User can mark sessions as "good examples" to weight them higher in retrieval

---

## 4. Human-Agent Interaction

### 4.1 Rich Permission Workflows

Extend the current 3-tier permission system with more granular and context-aware controls.

- **Scoped approvals**: "Allow all `git` commands for this session" instead of per-command approval
- **Time-limited permissions**: auto-revoke after N minutes or when the task completes
- **Permission templates**: save and reuse permission sets for common task types (e.g., "web development" auto-approves npm, git, and localhost browser access)
- **Audit log viewer**: searchable history of all permission decisions with the associated tool calls and outcomes

### 4.2 Interactive Plan Review

Before executing a complex plan, present it to the user as an editable checklist.

- User can reorder steps, remove steps, or add constraints
- User can mark steps as "skip" or "ask me first"
- Plan diffs: when the agent revises a plan mid-task, show what changed and why
- Approval modes: auto-execute (trust the agent), step-by-step (approve each action), review-plan-only (approve the plan, then auto-execute)

### 4.3 Voice Interaction

Extend the existing Android speech recognition to a full voice interaction loop.

- Speech-to-text on both desktop (browser Web Speech API or Whisper) and mobile
- Text-to-speech for agent responses (configurable voice, speed, pitch)
- Wake word detection for hands-free operation ("Hey Pi, ...")
- Voice commands for common actions: "stop", "pause", "approve", "deny", "go back"
- Transcription displayed in the chat interface alongside audio playback controls

### 4.4 Contextual Suggestions

Proactively suggest actions based on the current context without waiting for explicit user requests.

- File change detection: "I noticed you modified `auth.rs` — want me to run the tests?"
- Error pattern detection: "This error usually means the database migration is missing. Should I check?"
- Workflow completion: "You've committed but haven't pushed. Want me to create a PR?"
- Suggestions appear as dismissible cards in the UI, not as interruptions

### 4.5 Collaborative Editing

Real-time collaborative editing where the agent and user can work on the same file simultaneously.

- Agent proposes changes as highlighted diffs in the editor
- User can accept, reject, or modify each change inline
- Conflict resolution when both edit the same region
- Change attribution: color-coded by author (user vs. agent)

### 4.6 Guided Tutorials & Explanations

The agent can generate step-by-step walkthroughs of its actions for educational purposes.

- "Explain mode": the agent narrates what it's doing and why at each step
- Generated tutorials saved as markdown in the knowledge base
- Difficulty-adaptive: adjusts explanation depth based on user's demonstrated expertise
- Code annotations: the agent can add temporary inline comments explaining generated code

---

## 5. Visualization & Observability

### 5.1 Execution Timeline

A horizontal timeline view showing every action the agent has taken during a task.

- Each tool call is a node on the timeline with duration, status (success/failure), and a preview of the result
- Nodes are color-coded by tool type (shell=blue, browser=green, code=yellow, ML=purple)
- Click a node to expand full input/output details
- Filter by tool type, status, or time range
- Parallel tool calls shown as vertically stacked nodes at the same time position

### 5.2 Live Resource Monitor

Real-time dashboard showing system resource usage during agent execution.

- CPU, memory, disk I/O, network usage
- Per-process breakdown (Rust core, Python sidecar, headless Chrome, child processes)
- GPU utilization during ML training (CUDA/ROCm metrics from PyTorch)
- Historical chart with markers for each agent action
- Alerts when resources approach configured thresholds

### 5.3 Memory Visualization

Interactive visualization of the agent's memory state.

- **Embedding space**: 2D projection (UMAP/t-SNE) of stored embeddings, colored by source type (message, tool result, task)
- **Retrieval heatmap**: which memories were retrieved for each planning step
- **Memory timeline**: chronological view of all stored items with search and filter
- **Cluster detection**: automatically identify topic clusters in stored knowledge

### 5.4 Plan Visualization

Render the agent's current plan as an interactive directed graph.

- Nodes represent planned actions; edges represent dependencies
- Completed nodes are checked; in-progress nodes pulse; failed nodes are flagged
- Subtask expansion: click a node to see its child plan
- Critical path highlighting: show which steps are blocking completion

### 5.5 Diff & Change Visualization

When the agent modifies files, show rich diffs in the UI.

- Side-by-side and inline diff views
- Syntax-highlighted diffs for all supported languages
- Cumulative diff: show all changes made during a task as a single unified diff
- Before/after file tree comparison when the agent creates or deletes files

### 5.6 Training Dashboard

Dedicated view for monitoring ML model training in real time.

- Loss curves (training and validation) updating live
- Metric charts (accuracy, F1, BLEU, etc.) per epoch
- Learning rate schedule visualization
- GPU memory and utilization graphs
- Checkpoint timeline with performance annotations
- Early stopping indicator with configurable patience
- Model comparison table across training runs

---

## 6. Platform & Integration

### 6.1 Plugin / Skill System

A standardized plugin architecture allowing third-party tool and skill development.

- Plugin manifest (JSON/TOML) declaring tool name, schema, permission tier, and entry point
- Plugins can be written in Rust (native), Python (sidecar-hosted), or JavaScript (sandboxed)
- Plugin marketplace: discover, install, and update plugins from a central registry
- MCP (Model Context Protocol) compatibility for interoperability with other agent frameworks

### 6.2 Project Profiles

Configurable per-project settings that the agent loads when working in a specific directory.

- `.pi-assistant.toml` in the project root
- Defines: allowed tools, permission overrides, preferred LLM provider, custom tool configurations
- Auto-detected when the agent opens a project directory
- Shareable across teams via version control

### 6.3 iOS Client

Extend mobile support beyond Android with a native iOS client.

- SwiftUI interface matching the Android client's functionality
- Same WebSocket protocol, same state synchronization
- Siri Shortcuts integration for common agent commands
- Widget for home screen status monitoring

### 6.4 Web Client

Browser-based client for accessing Pi-Assistant remotely without installing the desktop app.

- Connects via WebSocket to the Rust core (same protocol as mobile)
- Progressive Web App (PWA) for installability
- Responsive design for desktop and mobile browsers
- End-to-end encryption for remote access over the internet

### 6.5 CI/CD Integration

Allow the agent to participate in CI/CD pipelines as an automated reviewer or fixer.

- GitHub Actions / GitLab CI integration: trigger the agent on PR events
- Automated code review: the agent comments on PRs with suggestions
- Auto-fix mode: the agent can push fix commits for failing CI checks (with approval)
- Pipeline status monitoring: the agent tracks build/test status and notifies the user

### 6.6 Calendar & Task Manager Integration

Connect with external productivity tools so the agent can schedule and track work.

- Google Calendar / CalDAV: schedule tasks, set reminders, block focus time
- Todoist / Linear / Jira: create issues, update status, link to agent tasks
- Bidirectional sync: changes in external tools reflect in the agent's task state

---

## 7. Safety & Reliability

### 7.1 Sandboxed Execution Profiles

Define execution profiles with strict resource and access boundaries.

- Network policies: allowlist/denylist of domains, ports, and protocols
- Filesystem policies: read-only mounts, writable scratch directories, size quotas
- Process policies: max child processes, CPU time limits, memory caps
- Profiles are composable: "web-dev" = "shell-safe" + "browser-localhost" + "npm-install"

### 7.2 Rollback & Undo

Every file modification and shell command is recorded in a transaction log that supports rollback.

- Automatic filesystem snapshots before destructive operations
- "Undo last action" command that reverses the most recent tool execution
- "Undo task" command that reverses all changes from a completed task
- Git-aware rollback: if changes were committed, create a revert commit

### 7.3 Anomaly Detection

Monitor agent behavior for patterns that indicate something has gone wrong.

- Token budget runaway: alert if the agent is consuming tokens much faster than expected
- Loop detection: flag if the agent is repeating the same tool call with the same parameters
- Output size monitoring: alert if a tool produces unexpectedly large output
- Behavioral baseline: learn normal patterns per task type and flag deviations

### 7.4 Audit & Compliance

Comprehensive logging for environments that require auditability.

- Immutable audit log of all agent actions, decisions, and user approvals
- Exportable reports (PDF, JSON) for compliance review
- Data retention policies with configurable auto-purge
- PII detection and redaction in stored memories

---

## 8. Performance & Scalability

### 8.1 Streaming Responses

Stream LLM planner output token-by-token to the UI instead of waiting for the complete response.

- Partial plan display: show reasoning as it's generated
- Early tool dispatch: begin executing the first tool call as soon as it's fully parsed, before the rest of the plan is complete
- Cancellation: user can stop generation mid-stream if the plan is heading in the wrong direction

### 8.2 Embedding Cache & Batch Processing

Optimize the memory system for larger knowledge bases.

- LRU cache for frequently accessed embeddings
- Batch embedding generation: process multiple texts in a single sidecar call
- Background re-indexing: update embeddings when the underlying model changes without blocking the agent
- Quantized embeddings (int8) to reduce storage and speed up similarity search

### 8.3 Model Quantization & Optimization

Reduce resource requirements for local model inference.

- GGUF/GGML quantized model support via llama.cpp integration
- ONNX Runtime for optimized inference on CPU
- Dynamic model loading: load models on demand, unload when idle
- Model distillation: train smaller task-specific models from larger teacher models

### 8.4 Persistent Sidecar Pool

Keep multiple Python sidecar instances warm for parallel tool execution.

- Pool size configurable based on available system resources
- Affinity routing: send training tasks to GPU-equipped sidecars, inference to CPU sidecars
- Health checking and automatic restart of unhealthy sidecars
- Graceful scaling: spin up additional sidecars under load, wind down when idle
