//! Sidecar lifecycle management and request/response routing.
//!
//! Spawns Python sidecar, routes NDJSON requests/responses via correlation IDs.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Request sent to the Python sidecar.
#[derive(Debug, Serialize)]
pub struct IpcRequest {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
}

/// Response from the Python sidecar.
#[derive(Debug, Deserialize)]
pub struct IpcResponse {
    pub id: String,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<IpcError>,
    #[serde(default)]
    pub progress: Option<serde_json::Value>,
}

/// Error from the sidecar.
#[derive(Debug, Deserialize)]
pub struct IpcError {
    pub code: String,
    pub message: String,
}

/// Progress update from the sidecar.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub request_id: String,
    pub data: serde_json::Value,
}

/// Handle for the Python sidecar process.
pub struct SidecarHandle {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<serde_json::Value>>>>>,
    progress_tx: mpsc::Sender<ProgressUpdate>,
    progress_rx: Option<mpsc::Receiver<ProgressUpdate>>,
    python_path: String,
    sidecar_dir: String,
    sidecar_module: String,
}

impl SidecarHandle {
    /// Create a new sidecar handle (not yet started).
    pub fn new() -> Self {
        let (progress_tx, progress_rx) = mpsc::channel(256);
        Self {
            child: None,
            stdin: None,
            pending: Arc::new(Mutex::new(HashMap::new())),
            progress_tx,
            progress_rx: Some(progress_rx),
            python_path: "python3".to_string(),
            sidecar_dir: "sidecar".to_string(),
            sidecar_module: "pi_sidecar".to_string(),
        }
    }

    /// Take the progress receiver (can only be called once).
    pub fn take_progress_rx(&mut self) -> Option<mpsc::Receiver<ProgressUpdate>> {
        self.progress_rx.take()
    }

    /// Configure the Python executable path.
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// Configure the sidecar directory name.
    pub fn with_sidecar_dir(mut self, dir: impl Into<String>) -> Self {
        self.sidecar_dir = dir.into();
        self
    }

    /// Configure the sidecar module name.
    pub fn with_sidecar_module(mut self, module: impl Into<String>) -> Self {
        self.sidecar_module = module.into();
        self
    }

    /// Start the sidecar process.
    pub async fn start(&mut self) -> Result<()> {
        if self.child.is_some() {
            return Ok(()); // Already running
        }

        info!(python = %self.python_path, dir = %self.sidecar_dir, module = %self.sidecar_module, "Starting Python sidecar");

        // Resolve absolute paths for development
        let cwd = std::env::current_dir().unwrap_or_default();
        let sidecars_root = if cwd.join("sidecars").exists() {
            cwd.join("sidecars")
        } else {
            cwd.join("../sidecars")
        };

        let sidecar_base = sidecars_root.join(&self.sidecar_dir);
        let shared_base = sidecars_root.join("shared");

        let sidecar_src = sidecar_base.join("src");
        let shared_src = shared_base.join("src");

        // Detect venv python (relative to sidecar_base)
        let venv_python = sidecar_base.join(".venv/bin/python");
        if venv_python.exists() {
            info!("Using venv python: {:?}", venv_python);
            self.python_path = venv_python.to_string_lossy().to_string();
        } else {
            info!("Using system python: {}", self.python_path);
        }

        // Combine PYTHONPATH
        let python_path = format!(
            "{}:{}",
            sidecar_src.to_string_lossy(),
            shared_src.to_string_lossy()
        );

        info!("Setting PYTHONPATH to: {}", python_path);

        let mut child = Command::new(&self.python_path)
            .arg("-m")
            .arg(&self.sidecar_module)
            .env("PYTHONPATH", python_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Log to console
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdout"))?;

        self.stdin = Some(stdin);
        self.child = Some(child);

        // Spawn stdout reader task
        let pending = self.pending.clone();
        let progress_tx = self.progress_tx.clone();

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                debug!(line = %line, "Sidecar output");

                match serde_json::from_str::<IpcResponse>(&line) {
                    Ok(response) => {
                        if let Some(progress) = response.progress {
                            // Progress update
                            let _ = progress_tx
                                .send(ProgressUpdate {
                                    request_id: response.id.clone(),
                                    data: progress,
                                })
                                .await;
                        } else {
                            // Final response
                            let mut pending = pending.lock().await;
                            if let Some(tx) = pending.remove(&response.id) {
                                let result = if let Some(err) = response.error {
                                    Err(anyhow!("{}: {}", err.code, err.message))
                                } else {
                                    Ok(response.result.unwrap_or(serde_json::Value::Null))
                                };
                                let _ = tx.send(result);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, line = %line, "Failed to parse sidecar response");
                    }
                }
            }

            info!("Sidecar stdout closed");
        });

        // Health check (5s timeout)
        let health = tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            self.send_raw_request("health.ping", serde_json::json!({})),
        )
        .await
        .map_err(|_| anyhow!("Health check timed out"))??;

        info!(response = ?health, "Sidecar health check passed");

        Ok(())
    }

    /// Stop the sidecar process.
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            // Try graceful shutdown first (best effort, no retry)
            if let Some(stdin) = self.stdin.as_mut() {
                let id = Uuid::new_v4().to_string();
                let request = IpcRequest {
                    id,
                    method: "lifecycle.shutdown".into(),
                    params: serde_json::json!({}),
                };
                if let Ok(line) = serde_json::to_string(&request) {
                    let _ = stdin.write_all(format!("{}\n", line).as_bytes()).await;
                    let _ = stdin.flush().await;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Force kill if still running
            let _ = child.kill().await;
            info!("Sidecar stopped");
        }
        self.stdin = None;
        Ok(())
    }

    /// Send a request to the sidecar and await the response.
    /// automatically restarts the sidecar if it is not running or if the request fails.
    pub async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // 1. Ensure sidecar is healthy before sending
        if !self.is_alive() {
            warn!("Sidecar is dead or not started, attempting to start...");
            // Clean up old child if necessary
            let _ = self.stop().await;
            self.start().await?;
        }

        // 2. Try raw request first
        match self.send_raw_request(method, params.clone()).await {
            Ok(res) => Ok(res),
            Err(e) => {
                warn!(error = %e, "Request failed, triggering restart...");
                // Restart
                let _ = self.stop().await;
                self.start().await?;

                // Retry once
                self.send_raw_request(method, params).await
            }
        }
    }

    /// Internal method to send request without retry logic
    async fn send_raw_request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();

        self.pending.lock().await.insert(id.clone(), tx);

        let request = IpcRequest {
            id: id.clone(),
            method: method.to_string(),
            params,
        };
        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("Sidecar not started"))?;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        // Wait for response with timeout
        tokio::time::timeout(std::time::Duration::from_secs(300), rx)
            .await
            .map_err(|_| {
                let pending = self.pending.clone();
                let id = id.clone();
                tokio::spawn(async move {
                    pending.lock().await.remove(&id);
                });
                anyhow!("Sidecar request timed out after 300s: {method}")
            })?
            .map_err(|_| anyhow!("Sidecar response channel closed"))?
    }

    /// Check if the sidecar process is actually running (not exited).
    pub fn is_alive(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(Some(_)) => false, // Exited
                Ok(None) => true,     // Still running
                Err(_) => false,      // Error checking
            }
        } else {
            false
        }
    }
}

impl Default for SidecarHandle {
    fn default() -> Self {
        Self::new()
    }
}
