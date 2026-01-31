//! Browser tool: headless Chrome automation via chromiumoxide.

use super::{PermissionTier, Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::{Browser, BrowserConfig, Page};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Browser automation tool.
pub struct BrowserTool {
    browser: Arc<Mutex<Option<Browser>>>,
    current_page: Arc<Mutex<Option<Page>>>,
    allowed_domains: Vec<String>,
}

impl BrowserTool {
    pub fn new() -> Self {
        Self {
            browser: Arc::new(Mutex::new(None)),
            current_page: Arc::new(Mutex::new(None)),
            allowed_domains: vec!["localhost".to_string(), "127.0.0.1".to_string()],
        }
    }

    /// Add an allowed domain.
    pub fn allow_domain(&mut self, domain: impl Into<String>) {
        self.allowed_domains.push(domain.into());
    }

    async fn ensure_browser(&self) -> Result<()> {
        let mut browser = self.browser.lock().await;
        if browser.is_none() {
            info!("Launching headless Chrome");
            let config = BrowserConfig::builder()
                .build()
                .map_err(|e| anyhow::anyhow!("Browser config error: {}", e))?;

            let (new_browser, mut handler) = Browser::launch(config).await?;

            // Spawn handler task to process browser events
            tokio::spawn(async move {
                while let Some(event) = handler.next().await {
                    if event.is_err() {
                        break;
                    }
                }
            });

            *browser = Some(new_browser);
        }
        Ok(())
    }

    fn is_url_allowed(&self, url: &str) -> bool {
        if let Ok(parsed) = url::Url::parse(url) {
            if let Some(host) = parsed.host_str() {
                return self
                    .allowed_domains
                    .iter()
                    .any(|d| host == d || host.ends_with(&format!(".{}", d)));
            }
        }
        false
    }

    async fn navigate(&self, url: &str) -> Result<ToolResult> {
        if !self.is_url_allowed(url) {
            return Ok(ToolResult::error(format!("Domain not allowed: {}", url)));
        }

        self.ensure_browser().await?;
        let browser = self.browser.lock().await;
        let browser = browser
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No browser"))?;

        let page = browser.new_page(url).await?;
        page.wait_for_navigation().await?;

        let title = page.get_title().await?.unwrap_or_default();

        let mut current = self.current_page.lock().await;
        *current = Some(page);

        Ok(ToolResult::success(format!(
            "Navigated to: {} ({})",
            url, title
        )))
    }

    async fn extract_text(&self) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let text: String = page
            .evaluate("document.body.innerText")
            .await?
            .into_value()?;
        Ok(ToolResult::success(text))
    }

    async fn extract_html(&self) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let html: String = page
            .evaluate("document.documentElement.outerHTML")
            .await?
            .into_value()?;
        Ok(ToolResult::success(html))
    }

    async fn screenshot(&self) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let screenshot = page
            .screenshot(
                chromiumoxide::page::ScreenshotParams::builder()
                    .format(CaptureScreenshotFormat::Png)
                    .build(),
            )
            .await?;

        let b64 = base64::engine::general_purpose::STANDARD.encode(&screenshot);
        Ok(
            ToolResult::success(format!("Screenshot captured ({} bytes)", screenshot.len()))
                .with_data(serde_json::json!({ "base64": b64 })),
        )
    }

    async fn click(&self, selector: &str) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let element = page.find_element(selector).await?;
        element.click().await?;

        Ok(ToolResult::success(format!("Clicked: {}", selector)))
    }

    async fn fill(&self, selector: &str, value: &str) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let element = page.find_element(selector).await?;
        element.click().await?;
        element.type_str(value).await?;

        Ok(ToolResult::success(format!(
            "Filled {} with text",
            selector
        )))
    }

    async fn evaluate(&self, script: &str) -> Result<ToolResult> {
        let page = self.current_page.lock().await;
        let page = page
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No page open"))?;

        let result: serde_json::Value = page.evaluate(script).await?.into_value()?;
        Ok(ToolResult::success(serde_json::to_string_pretty(&result)?))
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn description(&self) -> &str {
        "Control a headless browser. Actions: navigate, extract_text, extract_html, screenshot, click, fill, evaluate."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "extract_text", "extract_html", "screenshot", "click", "fill", "evaluate"],
                    "description": "Browser action to perform"
                },
                "url": { "type": "string", "description": "URL for navigate action" },
                "selector": { "type": "string", "description": "CSS selector for click/fill" },
                "value": { "type": "string", "description": "Value for fill action" },
                "script": { "type": "string", "description": "JavaScript for evaluate" }
            }
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        info!(action = action, "Browser action");

        match action {
            "navigate" => {
                let url = params
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'url'"))?;
                self.navigate(url).await
            }
            "extract_text" => self.extract_text().await,
            "extract_html" => self.extract_html().await,
            "screenshot" => self.screenshot().await,
            "click" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'selector'"))?;
                self.click(selector).await
            }
            "fill" => {
                let selector = params
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'selector'"))?;
                let value = params
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'value'"))?;
                self.fill(selector, value).await
            }
            "evaluate" => {
                let script = params
                    .get("script")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'script'"))?;
                self.evaluate(script).await
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}

impl Default for BrowserTool {
    fn default() -> Self {
        Self::new()
    }
}
