//! Messaging channels for Pi-Assistant.
//!
//! This module provides integrations with various messaging platforms
//! (Telegram, Discord, etc.) to allow the agent to receive tasks and
//! respond through multiple channels.

pub mod telegram;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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
