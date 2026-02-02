//! Drawing & Diagram tool using Mermaid syntax rendered in the Live Canvas.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tracing::info;

use crate::tools::canvas::CanvasStateManager;
use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};

/// Tool for generating and rendering diagrams via Mermaid syntax.
pub struct DiagramTool {
    app_handle: AppHandle,
    state_manager: Arc<CanvasStateManager>,
    /// Version counter for diagram artifacts.
    version: Arc<tokio::sync::Mutex<u32>>,
}

impl DiagramTool {
    pub fn new(app_handle: AppHandle, state_manager: Arc<CanvasStateManager>) -> Self {
        Self {
            app_handle,
            state_manager,
            version: Arc::new(tokio::sync::Mutex::new(0)),
        }
    }

    fn wrap_mermaid_html(mermaid_code: &str, title: Option<&str>) -> String {
        let title_text = title.unwrap_or("Diagram");
        let title_html = title.map(|t| format!("<h2>{}</h2>", t)).unwrap_or_default();

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title_text}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0; padding: 20px;
            display: flex; flex-direction: column;
            justify-content: center; align-items: center;
            min-height: 100vh;
            background: #1a1a2e; color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .mermaid {{ max-width: 100%; overflow: auto; }}
        h2 {{ text-align: center; margin-bottom: 16px; font-size: 16px; opacity: 0.7; }}
    </style>
</head>
<body>
    {title_html}
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            themeVariables: {{
                primaryColor: '#6366f1',
                primaryTextColor: '#fff',
                primaryBorderColor: '#818cf8',
                lineColor: '#94a3b8',
                secondaryColor: '#1e293b',
                tertiaryColor: '#0f172a'
            }}
        }});

        window.addEventListener('message', function(event) {{
            if (event.data && event.data.type === 'pi-canvas-eval') {{
                try {{
                    eval(event.data.code);
                }} catch (e) {{
                    console.error('Eval error:', e);
                }}
            }}
        }});
    </script>
</body>
</html>"#
        )
    }

    async fn render(&self, mermaid_code: &str, title: Option<&str>) -> anyhow::Result<ToolResult> {
        info!(title = ?title, code_len = mermaid_code.len(), "Rendering Mermaid diagram");

        let html = Self::wrap_mermaid_html(mermaid_code, title);

        self.app_handle.emit("canvas-push", &html)?;
        self.state_manager.save(&html).await?;

        let mut ver = self.version.lock().await;
        *ver += 1;

        Ok(ToolResult::success(format!(
            "Rendered diagram v{} ({} chars of Mermaid) in Live Canvas",
            *ver,
            mermaid_code.len()
        ))
        .with_data(json!({
            "version": *ver,
            "mermaid_length": mermaid_code.len(),
        })))
    }

    async fn export(&self, format: &str) -> anyhow::Result<ToolResult> {
        info!(format = format, "Exporting diagram");

        let js_code = match format {
            "svg" => {
                r#"
                (function() {
                    var svg = document.querySelector('.mermaid svg');
                    if (svg) {
                        var serializer = new XMLSerializer();
                        window.__piExportResult = serializer.serializeToString(svg);
                        document.title = 'EXPORT_READY:svg';
                    }
                })();
                "#
            }
            "png" => {
                r#"
                (function() {
                    var svg = document.querySelector('.mermaid svg');
                    if (svg) {
                        var canvas = document.createElement('canvas');
                        var ctx = canvas.getContext('2d');
                        var data = new XMLSerializer().serializeToString(svg);
                        var blob = new Blob([data], {type: 'image/svg+xml'});
                        var url = URL.createObjectURL(blob);
                        var img = new Image();
                        img.onload = function() {
                            canvas.width = img.width * 2;
                            canvas.height = img.height * 2;
                            ctx.scale(2, 2);
                            ctx.drawImage(img, 0, 0);
                            window.__piExportResult = canvas.toDataURL('image/png');
                            document.title = 'EXPORT_READY:png';
                            URL.revokeObjectURL(url);
                        };
                        img.src = url;
                    }
                })();
                "#
            }
            _ => {
                return Ok(ToolResult::error(format!(
                    "Unsupported format: {}. Use 'svg' or 'png'.",
                    format
                )));
            }
        };

        self.app_handle.emit("canvas-eval", js_code)?;

        Ok(ToolResult::success(format!(
            "Export command sent for {} format. Result will be available in the canvas.",
            format
        )))
    }
}

#[async_trait]
impl Tool for DiagramTool {
    fn name(&self) -> &str {
        "diagram"
    }

    fn description(&self) -> &str {
        "Generate and render diagrams using Mermaid syntax in the Live Canvas. Supports flowcharts, sequence diagrams, class diagrams, ER diagrams, state diagrams, Gantt charts, and more. Actions: render, export."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["render", "export"],
                    "description": "Diagram action to perform"
                },
                "code": {
                    "type": "string",
                    "description": "Mermaid diagram code (required for 'render')"
                },
                "title": {
                    "type": "string",
                    "description": "Optional title displayed above the diagram"
                },
                "format": {
                    "type": "string",
                    "enum": ["svg", "png"],
                    "description": "Export format (required for 'export', default: 'svg')"
                }
            }
        })
    }

    async fn execute(&self, params: Value, _context: ToolContext) -> anyhow::Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "render" => {
                let code = params
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'code' for render action"))?;
                let title = params.get("title").and_then(|v| v.as_str());
                self.render(code, title).await
            }
            "export" => {
                let format = params
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("svg");
                self.export(format).await
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}
