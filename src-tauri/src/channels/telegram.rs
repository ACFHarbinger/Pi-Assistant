//! Telegram bot channel integration.
//!
//! Uses teloxide to connect to the Telegram Bot API and route
//! messages to the Pi-Assistant agent. Supports text, photo, voice,
//! audio, and document messages with automatic transcription of
//! voice/audio via the sidecar STT pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use teloxide::net::Download;
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{error, info, warn};

use super::{Channel, ChannelResponse, MediaAttachment, MediaType};
use crate::ipc::SidecarHandle;
use pi_core::agent_types::AgentCommand;

/// Telegram bot channel.
pub struct TelegramChannel {
    /// Bot token
    token: String,
    /// Allowed user IDs (empty = allow all)
    allowed_users: Arc<RwLock<Vec<u64>>>,
    /// Running state
    running: Arc<AtomicBool>,
    /// Message sender to forward messages to agent
    message_tx: mpsc::Sender<AgentCommand>,
    /// Shutdown signal sender
    shutdown_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
    /// Sidecar handle for voice transcription
    ml_sidecar: Arc<Mutex<SidecarHandle>>,
    /// Media download directory
    media_dir: PathBuf,
}

impl TelegramChannel {
    /// Create a new Telegram channel.
    pub fn new(
        token: String,
        message_tx: mpsc::Sender<AgentCommand>,
        ml_sidecar: Arc<Mutex<SidecarHandle>>,
    ) -> Self {
        let media_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".pi-assistant")
            .join("media")
            .join("telegram");

        Self {
            token,
            allowed_users: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            message_tx,
            shutdown_tx: Arc::new(Mutex::new(None)),
            ml_sidecar,
            media_dir,
        }
    }

    /// Add an allowed user ID.
    pub async fn allow_user(&self, user_id: u64) {
        let mut users = self.allowed_users.write().await;
        if !users.contains(&user_id) {
            users.push(user_id);
            info!("Allowed Telegram user: {}", user_id);
        }
    }

    /// Download a file from Telegram by file_id to the media directory.
    async fn download_file(
        bot: &Bot,
        file_id: &str,
        media_dir: &PathBuf,
        ext: &str,
    ) -> Option<PathBuf> {
        // Ensure media dir exists
        if let Err(e) = tokio::fs::create_dir_all(media_dir).await {
            error!("Failed to create media directory: {}", e);
            return None;
        }

        let file = match bot.get_file(file_id).await {
            Ok(f) => f,
            Err(e) => {
                error!("Failed to get file info from Telegram: {}", e);
                return None;
            }
        };

        let file_name = format!(
            "{}_{}.{}",
            chrono_timestamp(),
            &file_id[..8.min(file_id.len())],
            ext
        );
        let dest_path = media_dir.join(&file_name);

        let mut dest_file = match tokio::fs::File::create(&dest_path).await {
            Ok(f) => f,
            Err(e) => {
                error!("Failed to create download file: {}", e);
                return None;
            }
        };

        if let Err(e) = bot.download_file(&file.path, &mut dest_file).await {
            error!("Failed to download file from Telegram: {}", e);
            let _ = tokio::fs::remove_file(&dest_path).await;
            return None;
        }

        info!("Downloaded Telegram file to {:?}", dest_path);
        Some(dest_path)
    }

    /// Transcribe an audio file using the sidecar STT.
    async fn transcribe_audio(
        sidecar: &Arc<Mutex<SidecarHandle>>,
        audio_path: &str,
    ) -> Option<String> {
        let mut sidecar = sidecar.lock().await;
        match sidecar
            .request(
                "voice.transcribe",
                serde_json::json!({
                    "audio_path": audio_path,
                    "model_size": "base",
                    "device": "cpu"
                }),
            )
            .await
        {
            Ok(result) => result
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            Err(e) => {
                warn!("STT transcription failed: {}", e);
                None
            }
        }
    }

    /// Handle an incoming message.
    async fn handle_message(
        bot: Bot,
        msg: Message,
        allowed_users: Arc<RwLock<Vec<u64>>>,
        message_tx: mpsc::Sender<AgentCommand>,
        ml_sidecar: Arc<Mutex<SidecarHandle>>,
        media_dir: PathBuf,
    ) -> ResponseResult<()> {
        // Get sender info
        let sender = match &msg.from {
            Some(user) => user,
            None => {
                warn!("Message without sender, ignoring");
                return Ok(());
            }
        };

        // Check if user is allowed
        let users = allowed_users.read().await;
        if !users.is_empty() && !users.contains(&sender.id.0) {
            warn!(
                "Unauthorized user {} attempted to send message",
                sender.id.0
            );
            bot.send_message(
                msg.chat.id,
                "⚠️ You are not authorized to use this bot. Contact the owner to request access.",
            )
            .await?;
            return Ok(());
        }
        drop(users);

        let sender_id = sender.id.0.to_string();
        let sender_name = Some(sender.full_name());
        let chat_id = msg.chat.id.0.to_string();
        let msg_id = msg.id.0.to_string();
        let _is_private = msg.chat.is_private();
        let _timestamp = msg.date.timestamp();

        // Determine message content and media
        let mut text = String::new();
        let mut media = Vec::new();
        let mut media_maps: Vec<HashMap<String, String>> = Vec::new();

        // Text content
        if let Some(t) = msg.text() {
            text = t.to_string();
        }

        // Caption (for photo/document/video messages)
        if let Some(caption) = msg.caption() {
            if text.is_empty() {
                text = caption.to_string();
            }
        }

        // Voice message
        if let Some(voice) = msg.voice() {
            if let Some(path) = Self::download_file(&bot, &voice.file.id, &media_dir, "ogg").await {
                let path_str = path.to_string_lossy().to_string();

                // Transcribe the voice message
                if let Some(transcription) = Self::transcribe_audio(&ml_sidecar, &path_str).await {
                    text = format!("[Voice message]: {}", transcription);
                } else {
                    text = "[Voice message]: (transcription failed)".to_string();
                }

                let attachment = MediaAttachment {
                    media_type: MediaType::Voice,
                    file_path: path_str.clone(),
                    file_name: None,
                    mime_type: voice.mime_type.clone().map(|m| m.to_string()),
                };
                media.push(attachment);

                let mut m = HashMap::new();
                m.insert("type".to_string(), "voice".to_string());
                m.insert("file_path".to_string(), path_str);
                media_maps.push(m);
            }
        }

        // Audio file
        if let Some(audio) = msg.audio() {
            if let Some(path) = Self::download_file(&bot, &audio.file.id, &media_dir, "mp3").await {
                let path_str = path.to_string_lossy().to_string();

                // Transcribe the audio
                if let Some(transcription) = Self::transcribe_audio(&ml_sidecar, &path_str).await {
                    if text.is_empty() {
                        text = format!("[Audio message]: {}", transcription);
                    } else {
                        text.push_str(&format!("\n[Audio transcription]: {}", transcription));
                    }
                }

                let file_name = audio
                    .file_name
                    .clone()
                    .unwrap_or_else(|| "audio.mp3".to_string());

                let attachment = MediaAttachment {
                    media_type: MediaType::Audio,
                    file_path: path_str.clone(),
                    file_name: Some(file_name.clone()),
                    mime_type: audio.mime_type.clone().map(|m| m.to_string()),
                };
                media.push(attachment);

                let mut m = HashMap::new();
                m.insert("type".to_string(), "audio".to_string());
                m.insert("file_path".to_string(), path_str);
                m.insert("file_name".to_string(), file_name);
                media_maps.push(m);
            }
        }

        // Photo (take the largest resolution)
        if let Some(photos) = msg.photo() {
            if let Some(photo) = photos.last() {
                if let Some(path) =
                    Self::download_file(&bot, &photo.file.id, &media_dir, "jpg").await
                {
                    let path_str = path.to_string_lossy().to_string();

                    if text.is_empty() {
                        text = format!("[Photo attached: {}]", path_str);
                    } else {
                        text.push_str(&format!("\n[Photo attached: {}]", path_str));
                    }

                    let attachment = MediaAttachment {
                        media_type: MediaType::Photo,
                        file_path: path_str.clone(),
                        file_name: None,
                        mime_type: Some("image/jpeg".to_string()),
                    };
                    media.push(attachment);

                    let mut m = HashMap::new();
                    m.insert("type".to_string(), "photo".to_string());
                    m.insert("file_path".to_string(), path_str);
                    media_maps.push(m);
                }
            }
        }

        // Document
        if let Some(doc) = msg.document() {
            let ext = doc
                .file_name
                .as_deref()
                .and_then(|n| n.rsplit('.').next())
                .unwrap_or("bin");
            if let Some(path) = Self::download_file(&bot, &doc.file.id, &media_dir, ext).await {
                let path_str = path.to_string_lossy().to_string();
                let file_name = doc
                    .file_name
                    .clone()
                    .unwrap_or_else(|| format!("document.{}", ext));

                if text.is_empty() {
                    text = format!("[Document attached: {}]", file_name);
                } else {
                    text.push_str(&format!("\n[Document attached: {}]", file_name));
                }

                let attachment = MediaAttachment {
                    media_type: MediaType::Document,
                    file_path: path_str.clone(),
                    file_name: Some(file_name.clone()),
                    mime_type: doc.mime_type.clone().map(|m| m.to_string()),
                };
                media.push(attachment);

                let mut m = HashMap::new();
                m.insert("type".to_string(), "document".to_string());
                m.insert("file_path".to_string(), path_str);
                m.insert("file_name".to_string(), file_name);
                media_maps.push(m);
            }
        }

        // Skip if no content at all
        if text.is_empty() && media.is_empty() {
            return Ok(());
        }

        // Forward to agent
        if let Err(e) = message_tx
            .send(AgentCommand::ChannelMessage {
                id: msg_id,
                channel: "telegram".to_string(),
                sender_id,
                sender_name,
                text,
                chat_id,
                media: media_maps,
            })
            .await
        {
            error!("Failed to forward Telegram message: {}", e);
        }

        Ok(())
    }
}

/// Generate a simple timestamp string for file naming.
fn chrono_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn start(&self) -> anyhow::Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(()); // Already running
        }

        info!("Starting Telegram bot...");

        let bot = Bot::new(&self.token);
        let allowed_users = self.allowed_users.clone();
        let message_tx = self.message_tx.clone();
        let running = self.running.clone();
        let ml_sidecar = self.ml_sidecar.clone();
        let media_dir = self.media_dir.clone();

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.lock().await = Some(shutdown_tx);

        running.store(true, Ordering::SeqCst);

        // Spawn the bot listener
        tokio::spawn(async move {
            let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
                let allowed = allowed_users.clone();
                let tx = message_tx.clone();
                let sc = ml_sidecar.clone();
                let md = media_dir.clone();
                async move { Self::handle_message(bot, msg, allowed, tx, sc, md).await }
            });

            let mut dispatcher = Dispatcher::builder(bot, handler)
                .enable_ctrlc_handler()
                .build();

            tokio::select! {
                _ = dispatcher.dispatch() => {
                    info!("Telegram dispatcher stopped");
                }
                _ = shutdown_rx.recv() => {
                    info!("Telegram bot shutdown requested");
                }
            }

            running.store(false, Ordering::SeqCst);
        });

        info!("Telegram bot started");
        Ok(())
    }

    async fn stop(&self) -> anyhow::Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _: Result<(), mpsc::error::SendError<()>> = tx.send(()).await;
        }

        self.running.store(false, Ordering::SeqCst);
        info!("Telegram bot stopped");
        Ok(())
    }

    async fn send(&self, response: ChannelResponse) -> anyhow::Result<()> {
        let bot = Bot::new(&self.token);
        let chat_id = ChatId(response.chat_id.parse::<i64>()?);

        let mut msg = bot.send_message(chat_id, &response.text);
        msg = msg.parse_mode(ParseMode::MarkdownV2);

        if let Some(reply_to) = response.reply_to {
            if let Ok(msg_id) = reply_to.parse::<i32>() {
                msg =
                    msg.reply_parameters(teloxide::types::ReplyParameters::new(MessageId(msg_id)));
            }
        }

        msg.await?;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
