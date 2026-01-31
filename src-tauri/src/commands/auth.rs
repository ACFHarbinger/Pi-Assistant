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

#[allow(deprecated)]
#[tauri::command]
pub async fn start_oauth(
    app_handle: tauri::AppHandle,
    provider: String,
    client_id: Option<String>,
) -> Result<String, String> {
    // 1. Determine Auth URL
    let is_antigravity = provider == "antigravity"
        || provider == "anthropic"
        || ((provider == "gemini" || provider == "google")
            && client_id.as_deref().unwrap_or("").is_empty());

    let (auth_base, actual_client_id, scope) = match provider.as_str() {
        "google" | "gemini" | "antigravity" | "anthropic" => {
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
    code_verifier: Option<String>,
    state: Option<String>,
) -> Result<(), String> {
    println!(
        "Auth Debug: provider={}, code_len={}, verifier={:?}, state={:?}",
        provider,
        code.len(),
        code_verifier,
        state
    );
    let is_antigravity = provider == "antigravity"
        || provider == "anthropic"
        || ((provider == "gemini" || provider == "google")
            && client_id.as_deref().unwrap_or("").is_empty());

    let (token_url, actual_client_id, actual_client_secret, actual_redirect_uri) =
        match provider.as_str() {
            "google" | "gemini" | "antigravity" | "anthropic" => {
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
            _ => return Err("Unsupported provider".into()),
        };

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();
    let res = if provider == "anthropic" {
        // Fallback to FORM encoding as JSON seems to trigger "client_secret missing" spuriously
        let mut params = HashMap::new();
        params.insert("client_id", actual_client_id.clone());
        params.insert("client_secret", actual_client_secret.clone()); // Should be empty string
        params.insert("code", code.clone());
        params.insert("grant_type", "authorization_code".to_string());
        params.insert("redirect_uri", actual_redirect_uri.clone());
        if let Some(verifier) = code_verifier.as_ref() {
            params.insert("code_verifier", verifier.clone());
        }
        if let Some(s) = state.as_ref() {
            params.insert("state", s.clone());
        }

        client.post(token_url).form(&params).send().await
    } else {
        let mut params = HashMap::new();
        params.insert("client_id", actual_client_id.clone());
        params.insert("client_secret", actual_client_secret.clone());
        params.insert("code", code.clone());
        params.insert("grant_type", "authorization_code".to_string());
        if let Some(verifier) = code_verifier.as_ref() {
            params.insert("code_verifier", verifier.clone());
        }
        params.insert("redirect_uri", actual_redirect_uri.clone());

        client.post(token_url).form(&params).send().await
    }
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
                .arg(token_url)
                .arg("-d")
                .arg(format!("code={}", code))
                .arg("-d")
                .arg(format!("client_id={}", actual_client_id))
                .arg("-d")
                .arg(format!("client_secret={}", actual_client_secret))
                .arg("-d")
                .arg(format!("redirect_uri={}", actual_redirect_uri))
                .arg("-d")
                .arg(format!(
                    "code_verifier={}",
                    code_verifier.unwrap_or_default()
                ))
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

// ── Claude Pro/Max OAuth Constants ──────────────────────────────────
const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const CLAUDE_AUTH_URL: &str = "https://claude.ai/oauth/authorize";
const CLAUDE_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const CLAUDE_SCOPE: &str = "org:create_api_key user:profile user:inference";

fn generate_pkce() -> (String, String) {
    use rand::Rng;
    use sha2::{Digest, Sha256};

    let mut rng = rand::thread_rng();
    let verifier: String = (0..43)
        .map(|_| {
            let idx = rng.gen_range(0..66);
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"[idx] as char
        })
        .collect();

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let hash = hasher.finalize();

    use base64::Engine;
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash);

    (verifier, challenge)
}

#[allow(deprecated)]
#[tauri::command]
pub async fn start_claude_oauth(app_handle: tauri::AppHandle) -> Result<(), String> {
    let (verifier, challenge) = generate_pkce();

    // 1. Start loopback server
    let (tx, rx) = oneshot::channel();
    let tx_shared = Arc::new(Mutex::new(Some(tx)));

    let app = Router::new().route(
        "/callback",
        get(move |Query(params): Query<AuthCallback>| {
            let tx = tx_shared.clone();
            async move {
                if let Some(chan) = tx.lock().await.take() {
                    let _ = chan.send(params.code);
                }
                Html(
                    "<html><body style='font-family:sans-serif;text-align:center;padding:50px;background:#18181b;color:#fff;'>
                    <h1 style='color:#0ea5e9;'>Claude Pro/Max Connected</h1>
                    <p>Pi-Assistant has received your credentials. You can close this tab now.</p>
                    </body></html>",
                )
            }
        }),
    );

    // Bind to localhost on random port
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 0)); // localhost resolves to 127.0.0.1
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("Failed to bind loopback server: {}", e))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();

    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // 2. Open browser
    // Use "localhost" not "127.0.0.1" — Anthropic's OAuth validates redirect_uri
    // and only allows http://localhost for native app PKCE flows (RFC 8252).
    let redirect_uri = format!("http://localhost:{}/callback", port);
    let state = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..32)
            .map(|_| {
                let idx = rng.gen_range(0..36);
                b"abcdefghijklmnopqrstuvwxyz0123456789"[idx] as char
            })
            .collect::<String>()
    };
    let full_url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
        CLAUDE_AUTH_URL,
        CLAUDE_CLIENT_ID,
        url::form_urlencoded::byte_serialize(redirect_uri.as_bytes()).collect::<String>(),
        url::form_urlencoded::byte_serialize(CLAUDE_SCOPE.as_bytes()).collect::<String>(),
        challenge,
        state,
    );

    app_handle
        .shell()
        .open(&full_url, None)
        .map_err(|e| format!("Failed to open browser: {}", e))?;

    // 3. Wait for auth code
    let code = match tokio::time::timeout(std::time::Duration::from_secs(300), rx).await {
        Ok(Ok(code)) => code,
        Ok(Err(_)) => return Err("Auth channel closed unexpectedly".into()),
        Err(_) => return Err("Authentication timed out (5 minutes)".into()),
    };

    // 4. Exchange code for tokens (Anthropic requires JSON body, not form-encoded)
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();

    let body = serde_json::json!({
        "grant_type": "authorization_code",
        "code": code,
        "state": state,
        "client_id": CLAUDE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier
    });

    let res = client
        .post(CLAUDE_TOKEN_URL)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Token exchange request failed: {}", e))?;

    let status = res.status();
    if !status.is_success() {
        let err = res.text().await.unwrap_or_default();
        return Err(format!("Token exchange failed ({}): {}", status, err));
    }

    #[derive(serde::Deserialize)]
    struct ClaudeTokenResponse {
        access_token: String,
        refresh_token: Option<String>,
        expires_in: Option<u64>,
    }

    let tokens: ClaudeTokenResponse = res
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {}", e))?;

    // 5. Store tokens
    crate::commands::config::save_api_key("claude_max_oauth".into(), tokens.access_token).await?;

    if let Some(refresh) = tokens.refresh_token {
        crate::commands::config::save_api_key("claude_max_refresh".into(), refresh).await?;
    }

    if let Some(expires_in) = tokens.expires_in {
        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            + expires_in;
        crate::commands::config::save_api_key(
            "claude_max_expires_at".into(),
            expires_at.to_string(),
        )
        .await?;
    }

    Ok(())
}

#[tauri::command]
pub async fn refresh_claude_token() -> Result<(), String> {
    let config_dir = dirs::home_dir()
        .ok_or("No home directory")?
        .join(".pi-assistant");
    let secrets_path = config_dir.join("secrets.json");

    let content = tokio::fs::read_to_string(&secrets_path)
        .await
        .map_err(|e| format!("Failed to read secrets: {}", e))?;
    let secrets: HashMap<String, String> =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse secrets: {}", e))?;

    let refresh_token = secrets
        .get("claude_max_refresh")
        .ok_or("No Claude refresh token found")?;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();

    let mut params = HashMap::new();
    params.insert("grant_type", "refresh_token".to_string());
    params.insert("refresh_token", refresh_token.clone());
    params.insert("client_id", CLAUDE_CLIENT_ID.to_string());

    let res = client
        .post(CLAUDE_TOKEN_URL)
        .form(&params)
        .send()
        .await
        .map_err(|e| format!("Refresh request failed: {}", e))?;

    let status = res.status();
    if !status.is_success() {
        let err = res.text().await.unwrap_or_default();
        return Err(format!("Token refresh failed ({}): {}", status, err));
    }

    #[derive(serde::Deserialize)]
    struct RefreshResponse {
        access_token: String,
        refresh_token: Option<String>,
        expires_in: Option<u64>,
    }

    let tokens: RefreshResponse = res
        .json()
        .await
        .map_err(|e| format!("Failed to parse refresh response: {}", e))?;

    crate::commands::config::save_api_key("claude_max_oauth".into(), tokens.access_token).await?;

    if let Some(refresh) = tokens.refresh_token {
        crate::commands::config::save_api_key("claude_max_refresh".into(), refresh).await?;
    }

    if let Some(expires_in) = tokens.expires_in {
        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            + expires_in;
        crate::commands::config::save_api_key(
            "claude_max_expires_at".into(),
            expires_at.to_string(),
        )
        .await?;
    }

    Ok(())
}

#[tauri::command]
pub async fn check_claude_auth() -> Result<bool, String> {
    let config_dir = dirs::home_dir()
        .ok_or("No home directory")?
        .join(".pi-assistant");
    let secrets_path = config_dir.join("secrets.json");

    if !secrets_path.exists() {
        return Ok(false);
    }

    let content = tokio::fs::read_to_string(&secrets_path)
        .await
        .map_err(|e| e.to_string())?;
    let secrets: HashMap<String, String> = serde_json::from_str(&content).unwrap_or_default();

    Ok(secrets.contains_key("claude_max_oauth"))
}

#[tauri::command]
pub async fn disconnect_claude_auth() -> Result<(), String> {
    let config_dir = dirs::home_dir()
        .ok_or("No home directory")?
        .join(".pi-assistant");
    let secrets_path = config_dir.join("secrets.json");

    if !secrets_path.exists() {
        return Ok(());
    }

    let content = tokio::fs::read_to_string(&secrets_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut secrets: HashMap<String, String> = serde_json::from_str(&content).unwrap_or_default();

    secrets.remove("claude_max_oauth");
    secrets.remove("claude_max_refresh");
    secrets.remove("claude_max_expires_at");

    let json = serde_json::to_string_pretty(&secrets).map_err(|e| e.to_string())?;
    tokio::fs::write(&secrets_path, json)
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub async fn check_provider_auth(provider: String) -> Result<bool, String> {
    let config_dir = dirs::home_dir()
        .ok_or("No home directory")?
        .join(".pi-assistant");
    let secrets_path = config_dir.join("secrets.json");

    if !secrets_path.exists() {
        return Ok(false);
    }

    let content = tokio::fs::read_to_string(&secrets_path)
        .await
        .map_err(|e| e.to_string())?;
    let secrets: HashMap<String, String> = serde_json::from_str(&content).unwrap_or_default();

    Ok(secrets.contains_key(&format!("{}_oauth", provider)))
}

#[tauri::command]
pub async fn disconnect_provider_auth(provider: String) -> Result<(), String> {
    let config_dir = dirs::home_dir()
        .ok_or("No home directory")?
        .join(".pi-assistant");
    let secrets_path = config_dir.join("secrets.json");

    if !secrets_path.exists() {
        return Ok(());
    }

    let content = tokio::fs::read_to_string(&secrets_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut secrets: HashMap<String, String> = serde_json::from_str(&content).unwrap_or_default();

    secrets.remove(&format!("{}_oauth", provider));
    secrets.remove(&format!("{}_refresh", provider));

    let json = serde_json::to_string_pretty(&secrets).map_err(|e| e.to_string())?;
    tokio::fs::write(&secrets_path, json)
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}
