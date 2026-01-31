use axum::{extract::Query, response::Html, routing::get, Router};
use serde::Deserialize;
use std::error::Error;
use std::sync::Arc;
use tauri_plugin_shell::ShellExt;
use tokio::sync::{oneshot, Mutex};

const ANTIGRAVITY_GOOGLE_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const ANTIGRAVITY_GOOGLE_CLIENT_SECRET: &str = "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";
const ANTIGRAVITY_REDIRECT_PORT: u16 = 51121;
const ANTIGRAVITY_CALLBACK_PATH: &str = "/oauth-callback";

#[derive(Deserialize)]
pub struct AuthCallback {
    pub code: String,
}

#[tauri::command]
pub async fn start_oauth(
    app_handle: tauri::AppHandle,
    provider: String,
    client_id: Option<String>,
) -> Result<String, String> {
    // 1. Determine Auth URL
    let is_antigravity = provider == "antigravity"
        || ((provider == "gemini" || provider == "google")
            && client_id.as_deref().unwrap_or("").is_empty());

    let (auth_base, actual_client_id, scope) = match provider.as_str() {
        "google" | "gemini" | "antigravity" => {
            if is_antigravity {
                (
                    "https://accounts.google.com/o/oauth2/v2/auth",
                    ANTIGRAVITY_GOOGLE_CLIENT_ID.to_string(),
                    "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/cclog https://www.googleapis.com/auth/experimentsandconfigs".to_string(),
                )
            } else {
                (
                    "https://accounts.google.com/o/oauth2/v2/auth",
                    client_id.clone().unwrap_or_default(),
                    "openid email profile https://www.googleapis.com/auth/generative-language"
                        .to_string(),
                )
            }
        }
        "anthropic" => (
            "https://auth.anthropic.com/oauth2/auth",
            client_id.clone().unwrap_or_default(),
            "all".to_string(),
        ),
        _ => return Err("Unsupported provider".into()),
    };

    // Check if client_id is still empty
    if actual_client_id.is_empty() {
        return Err(format!(
            "Client ID is required for provider '{}' when not using internal credentials.",
            provider
        ));
    }

    // 2. Setup loopback server
    let (tx, rx) = oneshot::channel();
    let tx_shared = Arc::new(Mutex::new(Some(tx)));

    let target_callback_path = if is_antigravity {
        ANTIGRAVITY_CALLBACK_PATH
    } else {
        "/callback"
    };

    let app = Router::new().route(
        target_callback_path,
        get(move |Query(params): Query<AuthCallback>| {
            let tx = tx_shared.clone();
            async move {
                if let Some(chan) = tx.lock().await.take() {
                    let _ = chan.send(params.code);
                }
                Html(
                    "<html><body style='font-family:sans-serif;text-align:center;padding:50px;'>
                <h1 style='color:#0ea5e9;'>Authentication Successful</h1>
                <p>Pi-Assistant has received your credentials. You can close this tab now.</p>
                </body></html>",
                )
            }
        }),
    );

    let target_port = if is_antigravity {
        ANTIGRAVITY_REDIRECT_PORT
    } else {
        5678
    };
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], target_port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(_) => {
            if is_antigravity {
                return Err(format!(
                    "Antigravity OAuth requires port {} to be available.",
                    ANTIGRAVITY_REDIRECT_PORT
                ));
            }
            // If port 5678 is taken, try a random port
            let addr_any = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
            tokio::net::TcpListener::bind(addr_any)
                .await
                .map_err(|e| e.to_string())?
        }
    };

    let local_addr = listener.local_addr().map_err(|e| e.to_string())?;
    let port = local_addr.port();

    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // 3. Open browser
    let redirect_uri = format!("http://localhost:{}{}", port, target_callback_path);
    let full_url = format!(
        "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state=pi_auth&access_type=offline&prompt=consent",
        auth_base,
        actual_client_id,
        url::form_urlencoded::byte_serialize(redirect_uri.as_bytes()).collect::<String>(),
        url::form_urlencoded::byte_serialize(scope.as_bytes()).collect::<String>()
    );

    app_handle
        .shell()
        .open(full_url, None)
        .map_err(|e| e.to_string())?;

    // 4. Wait for code (with timeout)
    match tokio::time::timeout(std::time::Duration::from_secs(300), rx).await {
        Ok(Ok(code)) => Ok(code),
        Ok(Err(_)) => Err("Channel closed".into()),
        Err(_) => Err("Authentication timed out".into()),
    }
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
}

#[tauri::command]
pub async fn exchange_oauth_code(
    provider: String,
    code: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    redirect_uri: Option<String>,
) -> Result<(), String> {
    let is_antigravity = provider == "antigravity"
        || ((provider == "gemini" || provider == "google")
            && client_id.as_deref().unwrap_or("").is_empty());

    let (token_url, actual_client_id, actual_client_secret, actual_redirect_uri) =
        match provider.as_str() {
            "google" | "gemini" | "antigravity" => {
                if is_antigravity {
                    (
                        "https://oauth2.googleapis.com/token",
                        ANTIGRAVITY_GOOGLE_CLIENT_ID.to_string(),
                        ANTIGRAVITY_GOOGLE_CLIENT_SECRET.to_string(),
                        format!(
                            "http://localhost:{}{}",
                            ANTIGRAVITY_REDIRECT_PORT, ANTIGRAVITY_CALLBACK_PATH
                        ),
                    )
                } else {
                    (
                        "https://oauth2.googleapis.com/token",
                        client_id.unwrap_or_default(),
                        client_secret.unwrap_or_default(),
                        redirect_uri.unwrap_or_default(),
                    )
                }
            }
            "anthropic" => (
                "https://auth.anthropic.com/oauth2/token",
                client_id.unwrap_or_default(),
                client_secret.unwrap_or_default(),
                redirect_uri.unwrap_or_default(),
            ),
            _ => return Err("Unsupported provider".into()),
        };

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();
    let mut params = HashMap::new();
    params.insert("client_id", actual_client_id.clone());
    params.insert("client_secret", actual_client_secret.clone());
    params.insert("code", code.clone());
    params.insert("grant_type", "authorization_code".to_string());
    params.insert("redirect_uri", actual_redirect_uri.clone());

    let res = client
        .post(token_url)
        .form(&params)
        .send()
        .await
        .map_err(|e| {
            let mut msg = format!("Request failed: {}", e);
            if let Some(s) = e.source() {
                msg.push_str(&format!("\nCaused by: {}", s));
            }
            msg
        });

    let res = match res {
        Ok(r) => r,
        Err(reqwest_err) => {
            println!("Reqwest failed, trying curl fallback: {}", reqwest_err);
            // Fallback to curl
            let status = std::process::Command::new("curl")
                .env_remove("HTTP_PROXY")
                .env_remove("http_proxy")
                .env_remove("HTTPS_PROXY")
                .env_remove("https_proxy")
                .env_remove("ALL_PROXY")
                .env_remove("all_proxy")
                .arg("-v")
                .arg("-4")
                .arg("-X")
                .arg("POST")
                .arg(&token_url)
                .arg("-d")
                .arg(format!("code={}", code))
                .arg("-d")
                .arg(format!("client_id={}", actual_client_id))
                .arg("-d")
                .arg(format!("client_secret={}", actual_client_secret))
                .arg("-d")
                .arg(format!("redirect_uri={}", actual_redirect_uri))
                .arg("-d")
                .arg("grant_type=authorization_code")
                .output()
                .map_err(|e| {
                    format!(
                        "Curl fallback execution failed: {}\nOriginal error: {}",
                        e, reqwest_err
                    )
                })?;

            if !status.status.success() {
                let stderr = String::from_utf8_lossy(&status.stderr);
                return Err(format!(
                    "Curl failed with status {}: {}\nOriginal: {}",
                    status.status, stderr, reqwest_err
                ));
            }

            let stdout = String::from_utf8_lossy(&status.stdout);
            let tokens: TokenResponse = serde_json::from_str(&stdout).map_err(|e| {
                format!("Failed to parse curl response: {}\nResponse: {}", e, stdout)
            })?;

            crate::commands::config::save_api_key(
                format!("{}_oauth", provider),
                tokens.access_token,
            )
            .await?;
            if let Some(refresh) = tokens.refresh_token {
                crate::commands::config::save_api_key(format!("{}_refresh", provider), refresh)
                    .await?;
            }
            return Ok(());
        }
    };

    let status = res.status();
    if !status.is_success() {
        let err = res.text().await.unwrap_or_default();
        return Err(format!("Token exchange failed ({}): {}", status, err));
    }

    let tokens: TokenResponse = res.json().await.map_err(|e| e.to_string())?;

    // Save tokens to secrets.json using the existing save_api_key logic (or similar)
    // Actually we can just call the save_api_key logic here internally
    crate::commands::config::save_api_key(format!("{}_oauth", provider), tokens.access_token)
        .await?;

    if let Some(refresh) = tokens.refresh_token {
        crate::commands::config::save_api_key(format!("{}_refresh", provider), refresh).await?;
    }

    Ok(())
}

use std::collections::HashMap;
