# Pi-Assistant Project Recipes

# ── Setup ─────────────────────────────────────────────────────────────────────

# Sync all dependencies (pip, npm, cargo)
sync:
	npm install
	cd sidecars/ml && uv sync
	cd sidecars/logic && uv sync
	cargo fetch

# ── Development ───────────────────────────────────────────────────────────────

# Run the desktop application in development mode
dev:
	npm run tauri dev

# Run the Android application (debug)
android:
	cd android && ./gradlew installDebug

# ── Quality ───────────────────────────────────────────────────────────────────

# Run all linters (Rust, Python, Frontend)
lint:
	cargo clippy --workspace -- -D warnings
	uv run ruff check .
	npm run lint

# Run all tests (Rust, Python, Frontend)
test:
	cargo test --workspace
	uv run pytest
	npm test

# ── Production ───────────────────────────────────────────────────────────────

# Build the production application
build:
	npm run tauri build
