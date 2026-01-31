//! Messaging channels for Pi-Assistant.
//!
//! This module provides integrations with various messaging platforms
//! (Telegram, Discord, etc.) to allow the agent to receive tasks and
//! respond through multiple channels.

pub mod discord;
pub mod telegram;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type of media attachment received from a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaType {
    Photo,
    Voice,
    Audio,
    Document,
    Video,
}

/// A media attachment from a channel message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaAttachment {
    /// Type of media
    pub media_type: MediaType,
    /// Local file path (after download)
    pub file_path: String,
    /// Original file name, if available
    pub file_name: Option<String>,
    /// MIME type, if available
    pub mime_type: Option<String>,
}

/// A message received from a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMessage {
    /// Unique message ID from the channel
    pub id: String,
    /// Channel type (telegram, discord, etc.)
    pub channel: String,
    /// Sender identifier
    pub sender_id: String,
    /// Sender display name
    pub sender_name: Option<String>,
    /// Message content (text)
    pub text: String,
    /// Timestamp
    pub timestamp: i64,
    /// Chat/conversation ID
    pub chat_id: String,
    /// Whether this is a private/DM conversation
    pub is_private: bool,
    /// Media attachments
    #[serde(default)]
    pub media: Vec<MediaAttachment>,
}

/// Response to send back to a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelResponse {
    /// Target chat/conversation ID
    pub chat_id: String,
    /// Response text
    pub text: String,
    /// Optional reply-to message ID
    pub reply_to: Option<String>,
}

/// Trait for messaging channel implementations.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Get the channel name identifier.
    fn name(&self) -> &str;

    /// Start the channel listener.
    async fn start(&self) -> anyhow::Result<()>;

    /// Stop the channel listener.
    async fn stop(&self) -> anyhow::Result<()>;

    /// Send a response to the channel.
    async fn send(&self, response: ChannelResponse) -> anyhow::Result<()>;

    /// Check if the channel is running.
    fn is_running(&self) -> bool;
}

/// Manages multiple communication channels.
pub struct ChannelManager {
    channels: Arc<RwLock<HashMap<String, Box<dyn Channel>>>>,
}

use std::collections::HashMap;
use tokio::sync::RwLock;

impl ChannelManager {
    /// Create a new channel manager.
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a channel.
    pub async fn add_channel(&self, channel: Box<dyn Channel>) {
        let mut channels = self.channels.write().await;
        channels.insert(channel.name().to_string(), channel);
    }

    /// Start a channel.
    pub async fn start_channel(&self, name: &str) -> anyhow::Result<()> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(name) {
            channel.start().await?;
        }
        Ok(())
    }

    /// Stop a channel.
    pub async fn stop_channel(&self, name: &str) -> anyhow::Result<()> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(name) {
            channel.stop().await?;
        }
        Ok(())
    }

    /// Send a response to a channel.
    pub async fn send_response(
        &self,
        channel_name: &str,
        response: ChannelResponse,
    ) -> anyhow::Result<()> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(channel_name) {
            channel.send(response).await?;
        }
        Ok(())
    }

    /// Get channel status.
    pub async fn is_running(&self, name: &str) -> bool {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(name) {
            channel.is_running()
        } else {
            false
        }
    }
}
