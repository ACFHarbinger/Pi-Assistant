use super::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use openapiv3::OpenAPI;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Cache entry with TTL.
struct CacheEntry {
    response: Value,
    status: u16,
    stored_at: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.stored_at.elapsed() > self.ttl
    }
}

/// API integration tool for HTTP requests.
pub struct ApiTool {
    client: Client,
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    /// Rate limit cooldown per domain.
    rate_limits: Arc<Mutex<HashMap<String, Instant>>>,
}

impl ApiTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Pi-Assistant/0.1.0")
            .build()
            .unwrap_or_default();

        Self {
            client,
            cache: Arc::new(Mutex::new(HashMap::new())),
            rate_limits: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn build_cache_key(method: &str, url: &str, body: Option<&Value>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(method.as_bytes());
        hasher.update(b":");
        hasher.update(url.as_bytes());
        if let Some(b) = body {
            hasher.update(b":");
            hasher.update(b.to_string().as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    fn extract_domain(url: &str) -> String {
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_default()
    }

    async fn check_rate_limit(&self, url: &str) -> Option<Duration> {
        let domain = Self::extract_domain(url);
        let limits = self.rate_limits.lock().await;
        if let Some(resume_at) = limits.get(&domain) {
            let now = Instant::now();
            if *resume_at > now {
                return Some(*resume_at - now);
            }
        }
        None
    }

    async fn prune_expired_cache(&self) {
        let mut cache = self.cache.lock().await;
        cache.retain(|_, entry| !entry.is_expired());
    }

    async fn ingest_spec(&self, url: &str) -> Result<ToolResult> {
        info!(url = %url, "Ingesting OpenAPI spec");

        // Fetch spec
        let response = self.client.get(url).send().await?;
        if !response.status().is_success() {
            return Ok(ToolResult::error(format!(
                "Failed to fetch spec: HTTP {}",
                response.status()
            )));
        }

        let content = response.text().await?;

        // Try parsing as JSON first, then YAML
        let openapi: OpenAPI = if let Ok(json_spec) = serde_json::from_str(&content) {
            json_spec
        } else {
            serde_yaml::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse spec as JSON or YAML: {}", e))?
        };

        let mut output = Vec::new();
        output.push(format!(
            "OpenAPI Spec: {} ({})",
            openapi.info.title, openapi.info.version
        ));
        if let Some(desc) = &openapi.info.description {
            output.push(format!("Description: {}", desc));
        }
        output.push(String::new());
        output.push("Endpoints:".to_string());

        for (path, item) in openapi.paths.iter() {
            let item = match item {
                openapiv3::ReferenceOr::Item(i) => i,
                openapiv3::ReferenceOr::Reference { reference } => {
                    // Simple handling for references by just listing them
                    output.push(format!("  {} (Ref: {})", path, reference));
                    continue;
                }
            };

            // Helper to format operation
            let format_op = |method: &str, op: &Option<openapiv3::Operation>| {
                if let Some(o) = op {
                    let summary = o.summary.as_deref().unwrap_or("No summary");
                    format!("  {:4} {} - {}", method, path, summary)
                } else {
                    String::new()
                }
            };

            if let Some(_) = &item.get {
                output.push(format_op("GET", &item.get));
            }
            if let Some(_) = &item.post {
                output.push(format_op("POST", &item.post));
            }
            if let Some(_) = &item.put {
                output.push(format_op("PUT", &item.put));
            }
            if let Some(_) = &item.delete {
                output.push(format_op("DEL", &item.delete));
            }
            if let Some(_) = &item.patch {
                output.push(format_op("PTCH", &item.patch));
            }
        }

        // Clean up empty lines
        let output_str = output
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult::success(output_str).with_data(json!({
            "title": openapi.info.title,
            "version": openapi.info.version,
            "description": openapi.info.description,
            "servers": openapi.servers,
        })))
    }

    async fn request(
        &self,
        method_str: &str,
        url: &str,
        headers: Option<&Value>,
        body: Option<&Value>,
        auth: Option<&Value>,
        cache_ttl: Option<u64>,
        timeout_secs: Option<u64>,
    ) -> Result<ToolResult> {
        info!(method = %method_str, url = %url, "Making HTTP request");

        // Check rate limit
        if let Some(wait) = self.check_rate_limit(url).await {
            return Ok(ToolResult::error(format!(
                "Rate limited for domain '{}'. Retry after {:.0}s.",
                Self::extract_domain(url),
                wait.as_secs_f64()
            )));
        }

        let method = match method_str.to_uppercase().as_str() {
            "GET" => Method::GET,
            "POST" => Method::POST,
            "PUT" => Method::PUT,
            "PATCH" => Method::PATCH,
            "DELETE" => Method::DELETE,
            "HEAD" => Method::HEAD,
            "OPTIONS" => Method::OPTIONS,
            _ => {
                return Ok(ToolResult::error(format!(
                    "Unsupported HTTP method: {}",
                    method_str
                )))
            }
        };

        // Check cache for GET requests
        let cache_key = Self::build_cache_key(method_str, url, body);
        if method == Method::GET {
            if let Some(ttl) = cache_ttl {
                if ttl > 0 {
                    let cache = self.cache.lock().await;
                    if let Some(entry) = cache.get(&cache_key) {
                        if !entry.is_expired() {
                            return Ok(ToolResult::success(format!(
                                "HTTP {} (cached)\n\n{}",
                                entry.status,
                                serde_json::to_string_pretty(&entry.response)
                                    .unwrap_or_else(|_| entry.response.to_string())
                            ))
                            .with_data(json!({
                                "status": entry.status,
                                "body": entry.response,
                                "cached": true,
                            })));
                        }
                    }
                }
            }
        }

        // Prune expired cache entries
        self.prune_expired_cache().await;

        // Build request
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(30));
        let mut req_builder = self.client.request(method.clone(), url).timeout(timeout);

        // Add headers
        if let Some(hdrs) = headers {
            if let Some(obj) = hdrs.as_object() {
                for (key, value) in obj {
                    if let Some(val_str) = value.as_str() {
                        req_builder = req_builder.header(key.as_str(), val_str);
                    }
                }
            }
        }

        // Add authentication
        if let Some(auth_config) = auth {
            let auth_type = auth_config
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("bearer");

            match auth_type {
                "bearer" => {
                    let token = auth_config
                        .get("token")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing 'token' for bearer auth"))?;
                    req_builder = req_builder.bearer_auth(token);
                }
                "basic" => {
                    let username = auth_config
                        .get("username")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let password = auth_config.get("password").and_then(|v| v.as_str());
                    req_builder = req_builder.basic_auth(username, password);
                }
                "api_key" => {
                    let key = auth_config
                        .get("key")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing 'key' for api_key auth"))?;
                    let header_name = auth_config
                        .get("header_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("X-API-Key");
                    req_builder = req_builder.header(header_name, key);
                }
                _ => {
                    return Ok(ToolResult::error(format!(
                        "Unknown auth type: {}",
                        auth_type
                    )));
                }
            }
        }

        // Add body
        if let Some(body_val) = body {
            if let Some(body_str) = body_val.as_str() {
                req_builder = req_builder.body(body_str.to_string());
            } else {
                req_builder = req_builder.json(body_val);
            }
        }

        // Execute request
        let start = Instant::now();
        let response = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult::error(format!("Request failed: {}", e)));
            }
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let status = response.status().as_u16();
        let status_text = response
            .status()
            .canonical_reason()
            .unwrap_or("")
            .to_string();

        // Handle rate limiting
        if status == 429 {
            if let Some(retry_after) = response.headers().get("retry-after") {
                if let Ok(secs_str) = retry_after.to_str() {
                    if let Ok(secs) = secs_str.parse::<u64>() {
                        let domain = Self::extract_domain(url);
                        let resume_at = Instant::now() + Duration::from_secs(secs);
                        self.rate_limits.lock().await.insert(domain, resume_at);
                        warn!(url = %url, retry_after = secs, "Rate limited");
                    }
                }
            }
        }

        // Collect response headers
        let resp_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let content_type = resp_headers
            .get("content-type")
            .cloned()
            .unwrap_or_default();

        // Read body
        let body_text = match response.text().await {
            Ok(t) => t,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to read response body: {}",
                    e
                )));
            }
        };

        // Try to parse as JSON
        let body_value: Value = if content_type.contains("json") {
            serde_json::from_str(&body_text).unwrap_or(Value::String(body_text.clone()))
        } else {
            Value::String(body_text.clone())
        };

        // Cache GET responses if TTL specified
        if method == Method::GET {
            if let Some(ttl) = cache_ttl {
                if ttl > 0 && status >= 200 && status < 300 {
                    let mut cache = self.cache.lock().await;
                    cache.insert(
                        cache_key,
                        CacheEntry {
                            response: body_value.clone(),
                            status,
                            stored_at: Instant::now(),
                            ttl: Duration::from_secs(ttl),
                        },
                    );
                }
            }
        }

        // Format output
        let body_preview = if body_text.len() > 2000 {
            format!(
                "{}...\n(truncated, {} bytes total)",
                &body_text[..2000],
                body_text.len()
            )
        } else {
            body_text
        };

        let output = format!(
            "HTTP {} {} ({}ms)\nContent-Type: {}\n\n{}",
            status, status_text, duration_ms, content_type, body_preview
        );

        let success = status >= 200 && status < 400;

        Ok(ToolResult {
            success,
            output,
            error: if success {
                None
            } else {
                Some(format!("HTTP {}", status))
            },
            data: Some(json!({
                "status": status,
                "status_text": status_text,
                "headers": resp_headers,
                "body": body_value,
                "duration_ms": duration_ms,
                "cached": false,
            })),
        })
    }
}

#[async_trait]
impl Tool for ApiTool {
    fn name(&self) -> &str {
        "api"
    }

    fn description(&self) -> &str {
        "Make HTTP requests to external APIs. Supports OpenAPI spec ingestion, GET, POST, PUT, PATCH, DELETE with authentication, caching, and rate limit awareness."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["action", "url"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["request", "ingest_spec"],
                    "description": "API action to perform"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                    "description": "HTTP method (default: GET)"
                },
                "url": {
                    "type": "string",
                    "description": "Full URL to request or spec URL"
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers as key-value pairs"
                },
                "body": {
                    "description": "Request body (string or JSON object)"
                },
                "auth": {
                    "type": "object",
                    "description": "Authentication: { type: 'bearer'|'basic'|'api_key', token?, username?, password?, key?, header_name? }"
                },
                "cache_ttl_secs": {
                    "type": "integer",
                    "description": "Cache TTL in seconds for GET requests (default: 0 = no cache)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30)"
                }
            }
        })
    }

    async fn execute(&self, params: Value, _context: ToolContext) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "ingest_spec" => {
                let url = params
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'url' parameter"))?;
                self.ingest_spec(url).await
            }
            "request" => {
                let url = params
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'url' parameter"))?;
                let method = params
                    .get("method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("GET");
                let headers = params.get("headers");
                let body = params.get("body");
                let auth = params.get("auth");
                let cache_ttl = params.get("cache_ttl_secs").and_then(|v| v.as_u64());
                let timeout_secs = params.get("timeout_secs").and_then(|v| v.as_u64());

                self.request(method, url, headers, body, auth, cache_ttl, timeout_secs)
                    .await
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
