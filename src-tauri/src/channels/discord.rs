//! Discord bot channel integration.
//!
//! Uses serenity to connect to the Discord API and route
//! messages to the Pi-Assistant agent.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serenity::all::{ChannelId, GatewayIntents, Message, Ready};
use serenity::prelude::*;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

use super::{Channel, ChannelMessage, ChannelResponse};
use pi_core::agent_types::AgentCommand;

struct Handler {
    message_tx: mpsc::Sender<AgentCommand>,
}

#[async_trait]
impl EventHandler for Handler {
    async def message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }

        let is_dm = msg.guild_id.is_none();
        
        // Only respond if it's a DM or the bot is mentioned
        let bot_user_id = ctx.cache.current_user().id;
        let is_mentioned = msg.mentions.iter().any(|u| u.id == bot_user_id);

        if !is_dm && !is_mentioned {
            return;
        }

        // Strip mention from text if present
        let mut text = msg.content.clone();
        if is_mentioned {
            let mention = format!("<@{}>", bot_user_id);
            let mention_nickname = format!("<@!{}>", bot_user_id);
            text = text.replace(&mention, "").replace(&mention_nickname, "").trim().to_string();
        }

        // Create channel message
        let channel_msg = ChannelMessage {
            id: msg.id.to_string(),
            channel: "discord".to_string(),
            sender_id: msg.author.id.to_string(),
            sender_name: Some(msg.author.name.clone()),
            text,
            timestamp: msg.timestamp.unix_timestamp(),
            chat_id: msg.channel_id.to_string(),
            is_private: is_dm,
        };

        // Forward to agent
        if let Err(e) = self.message_tx
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
            error!("Failed to forward Discord message: {}", e);
        }
    }

    async fn ready(&self, _: Context, ready: Ready) {
        info!("Discord bot '{}' is connected!", ready.user.name);
    }
}

/// Discord bot channel.
pub struct DiscordChannel {
    /// Bot token
    token: String,
    /// Running state
    running: Arc<AtomicBool>,
    /// Message sender to forward messages to agent
    message_tx: mpsc::Sender<AgentCommand>,
    /// Shutdown signal sender
    shard_manager: Arc<Mutex<Option<Arc<serenity::all::ShardManager>>>>,
}

impl DiscordChannel {
    /// Create a new Discord channel.
    pub fn new(token: String, message_tx: mpsc::Sender<AgentCommand>) -> Self {
        Self {
            token,
            running: Arc::new(AtomicBool::new(false)),
            message_tx,
            shard_manager: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> &str {
        "discord"
    }

    async fn start(&self) -> anyhow::Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(()); // Already running
        }

        info!("Starting Discord bot...");

        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let handler = Handler {
            message_tx: self.message_tx.clone(),
        };

        let mut client = Client::builder(&self.token, intents)
            .event_handler(handler)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Discord client: {}", e))?;

        let shard_manager = client.shard_manager.clone();
        *self.shard_manager.lock().await = Some(shard_manager);
        
        let running = self.running.clone();
        running.store(true, Ordering::SeqCst);

        tokio::spawn(async move {
            if let Err(why) = client.start().await {
                error!("Discord client error: {:?}", why);
            }
            running.store(false, Ordering::SeqCst);
        });

        info!("Discord bot started");
        Ok(())
    }

    async fn stop(&self) -> anyhow::Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        if let Some(shard_manager) = self.shard_manager.lock().await.take() {
            shard_manager.shutdown_all().await;
        }

        self.running.store(false, Ordering::SeqCst);
        info!("Discord bot stopped");
        Ok(())
    }

    async fn send(&self, response: ChannelResponse) -> anyhow::Result<()> {
        let http = serenity::http::Http::new(&self.token);
        let channel_id: u64 = response.chat_id.parse()?;
        let channel = ChannelId::new(channel_id);

        channel.say(&http, &response.text).await?;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
