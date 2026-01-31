use crate::mcp::config::McpServerConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, info};

// ── JSON-RPC Types ───────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    result: Option<Value>,
    error: Option<Value>,
    id: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcNotification {
    jsonrpc: String,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

// ── MCP Client ───────────────────────────────────────────────────────

pub struct McpClient {
    name: String,
    _process: Mutex<Child>,
    stdin: Mutex<tokio::process::ChildStdin>,
    // For simplicity, we'll use a request counter and just wait for the next line
    // In a real robust implementation, we'd need a message loop map like IPC
    reader: Mutex<BufReader<tokio::process::ChildStdout>>,
    next_id: Mutex<u64>,
}

impl McpClient {
    pub async fn new(name: &str, config: &McpServerConfig) -> Result<Self> {
        info!("Starting MCP server: {} ({})", name, config.command);

        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args);
        cmd.envs(&config.env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit()); // Log stderr to parent stderr

        let mut child = cmd.spawn().context("Failed to spawn MCP server")?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to open stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to open stdout"))?;
        let reader = BufReader::new(stdout);

        let client = Self {
            name: name.to_string(),
            _process: Mutex::new(child),
            stdin: Mutex::new(stdin),
            reader: Mutex::new(reader),
            next_id: Mutex::new(0),
        };

        // Initialize handshake
        client.initialize().await?;

        Ok(client)
    }

    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = {
            let mut guard = self.next_id.lock().await;
            *guard += 1;
            *guard
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: Some(id),
        };

        let json = serde_json::to_string(&request)?;
        debug!("MCP TX [{}]: {}", self.name, json);

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(json.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
        drop(stdin); // Release lock

        // Read response (Blocking logic for simplicity - assumes sync response order)
        // TODO: Handle out-of-order responses or notifications in a real loop
        let mut reader = self.reader.lock().await;
        let mut line = String::new();

        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).await?;
            if bytes == 0 {
                return Err(anyhow::anyhow!(
                    "MCP server {} closed connection",
                    self.name
                ));
            }

            debug!("MCP RX [{}]: {}", self.name, line.trim());

            if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                if response.id == Some(id) {
                    if let Some(error) = response.error {
                        return Err(anyhow::anyhow!("MCP Error: {:?}", error));
                    }
                    return Ok(response.result.unwrap_or(Value::Null));
                }
            }
            // Ignore other messages (logging/notifications) for now
        }
    }

    async fn initialize(&self) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": "0.1.0",
            "capabilities": {
                "roots": { "listChanged": false },
                "sampling": {}
            },
            "clientInfo": {
                "name": "pi-assistant",
                "version": "0.1.0"
            }
        });

        let result = self.request("initialize", Some(params)).await?;
        debug!("MCP Initialized: {:?}", result);

        // Send 'notifications/initialized'
        let notification = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let json = serde_json::to_string(&notification)?;
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(json.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;

        Ok(())
    }

    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let response = self.request("tools/list", None).await?;

        #[derive(Deserialize)]
        struct ToolsListResponse {
            tools: Vec<McpToolInfo>,
        }

        let list: ToolsListResponse = serde_json::from_value(response)?;
        Ok(list.tools)
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": args
        });

        let response = self.request("tools/call", Some(params)).await?;

        #[derive(Deserialize)]
        struct ToolContent {
            #[serde(rename = "type")]
            kind: String,
            text: Option<String>,
        }

        #[derive(Deserialize)]
        struct ToolCallResponse {
            content: Vec<ToolContent>,
            #[serde(rename = "isError")]
            is_error: Option<bool>,
        }

        let result: ToolCallResponse = serde_json::from_value(response)?;

        if result.is_error == Some(true) {
            let error_text = result
                .content
                .iter()
                .map(|c| c.text.clone().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n");
            return Err(anyhow::anyhow!("Tool execution failed: {}", error_text));
        }

        // Combine all text content
        let output = result
            .content
            .into_iter()
            .filter(|c| c.kind == "text")
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        Ok(output)
    }
}
