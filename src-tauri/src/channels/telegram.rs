//! Telegram bot channel integration.
//!
//! Uses teloxide to connect to the Telegram Bot API and route
//! messages to the Pi-Assistant agent.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{error, info, warn};

use super::{Channel, ChannelMessage, ChannelResponse};
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
}

impl TelegramChannel {
    /// Create a new Telegram channel.
    pub fn new(token: String, message_tx: mpsc::Sender<AgentCommand>) -> Self {
        Self {
            token,
            allowed_users: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            message_tx,
            shutdown_tx: Arc::new(Mutex::new(None)),
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

    /// Handle an incoming message.
    async fn handle_message(
        bot: Bot,
        msg: Message,
        allowed_users: Arc<RwLock<Vec<u64>>>,
        message_tx: mpsc::Sender<AgentCommand>,
    ) -> ResponseResult<()> {
        // Check if message has text
        let text = match msg.text() {
            Some(t) => t.to_string(),
            None => {
                // Skip non-text messages for now
                return Ok(());
            }
        };

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

        // Create channel message
        let channel_msg = ChannelMessage {
            id: msg.id.0.to_string(),
            channel: "telegram".to_string(),
            sender_id: sender.id.0.to_string(),
            sender_name: Some(sender.full_name()),
            text,
            timestamp: msg.date.timestamp(),
            chat_id: msg.chat.id.0.to_string(),
            is_private: msg.chat.is_private(),
        };

        // Forward to agent
        if let Err(e) = message_tx
            .send(AgentCommand::ChannelMessage {
                id: channel_msg.id,
                channel: channel_msg.channel,
                sender_id: channel_msg.sender_id,
                sender_name: channel_msg.sender_name,
                text: channel_msg.text,
                chat_id: channel_msg.chat_id,
            })
            .await
        {
            error!("Failed to forward Telegram message: {}", e);
        }

        Ok(())
    }
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

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.lock().await = Some(shutdown_tx);

        running.store(true, Ordering::SeqCst);

        // Spawn the bot listener
        tokio::spawn(async move {
            let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
                let allowed = allowed_users.clone();
                let tx = message_tx.clone();
                async move { Self::handle_message(bot, msg, allowed, tx).await }
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
