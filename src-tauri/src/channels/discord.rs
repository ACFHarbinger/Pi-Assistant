//! Discord bot channel integration.
//!
//! Uses serenity to connect to the Discord API and route
//! messages to the Pi-Assistant agent.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serenity::all::{
    ChannelId, Command, CommandInteraction, CommandOptionType, CreateCommand, CreateCommandOption,
    CreateInteractionResponse, CreateInteractionResponseMessage, GatewayIntents, Interaction,
    Message, Ready,
};
use serenity::prelude::*;
use tokio::sync::{mpsc, watch, Mutex};
use tracing::{error, info};

use super::{Channel, ChannelMessage, ChannelResponse};
use pi_core::agent_types::{AgentCommand, AgentState};

struct Handler {
    message_tx: mpsc::Sender<AgentCommand>,
    bot_id: Mutex<Option<serenity::all::UserId>>,
    agent_state_rx: watch::Receiver<AgentState>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, _ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }

        let is_dm = msg.guild_id.is_none();

        let bot_user_id = {
            let id = self.bot_id.lock().await;
            id.unwrap_or(msg.author.id) // Fallback if not ready
        };
        let is_mentioned = msg.mentions.iter().any(|u| u.id == bot_user_id);

        if !is_dm && !is_mentioned {
            return;
        }

        // Strip mention from text if present
        let mut text = msg.content.clone();
        if is_mentioned {
            let mention = format!("<@{}>", bot_user_id);
            let mention_nickname = format!("<@!{}>", bot_user_id);
            text = text
                .replace(&mention, "")
                .replace(&mention_nickname, "")
                .trim()
                .to_string();
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
            media: vec![],
        };

        // Forward to agent
        if let Err(e) = self
            .message_tx
            .send(AgentCommand::ChannelMessage {
                id: channel_msg.id,
                channel: channel_msg.channel,
                sender_id: channel_msg.sender_id,
                sender_name: channel_msg.sender_name,
                text: channel_msg.text,
                chat_id: channel_msg.chat_id,
                media: vec![],
            })
            .await
        {
            error!("Failed to forward Discord message: {}", e);
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Command(command) = interaction {
            let response_content = self.handle_slash_command(&command).await;

            let response = CreateInteractionResponse::Message(
                CreateInteractionResponseMessage::new().content(response_content),
            );

            if let Err(e) = command.create_response(&ctx.http, response).await {
                error!("Failed to respond to slash command: {}", e);
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        info!("Discord bot '{}' is connected!", ready.user.name);
        *self.bot_id.lock().await = Some(ready.user.id);

        // Register global slash commands
        let commands = vec![
            CreateCommand::new("ask")
                .description("Send a message to Pi-Assistant")
                .add_option(
                    CreateCommandOption::new(CommandOptionType::String, "message", "Your message")
                        .required(true),
                ),
            CreateCommand::new("status").description("Get the current agent status"),
            CreateCommand::new("stop").description("Stop the currently running agent task"),
        ];

        for command in commands {
            if let Err(e) = Command::create_global_command(&ctx.http, command).await {
                error!("Failed to register slash command: {}", e);
            }
        }

        info!("Slash commands registered successfully");
    }
}

impl Handler {
    async fn handle_slash_command(&self, command: &CommandInteraction) -> String {
        match command.data.name.as_str() {
            "ask" => {
                let message = command
                    .data
                    .options
                    .first()
                    .and_then(|opt| opt.value.as_str())
                    .unwrap_or("Hello!");

                let _ = self
                    .message_tx
                    .send(AgentCommand::ChannelMessage {
                        id: command.id.to_string(),
                        channel: "discord".to_string(),
                        sender_id: command.user.id.to_string(),
                        sender_name: Some(command.user.name.clone()),
                        text: message.to_string(),
                        chat_id: command.channel_id.to_string(),
                        media: vec![],
                    })
                    .await;

                format!("📨 Message sent to Pi: \"{}\"", message)
            }
            "status" => {
                let state = self.agent_state_rx.borrow().clone();
                match state {
                    AgentState::Idle => "🟢 Agent is idle and ready.".to_string(),
                    AgentState::Running { iteration, .. } => {
                        format!("🔄 Agent is running (iteration {}).", iteration)
                    }
                    AgentState::Paused { .. } => "⏸️ Agent is paused.".to_string(),
                    AgentState::Stopped { reason, .. } => {
                        format!("🛑 Agent stopped: {:?}", reason)
                    }
                    _ => "❓ Unknown state.".to_string(),
                }
            }
            "stop" => {
                let _ = self
                    .message_tx
                    .send(AgentCommand::Stop { agent_id: None })
                    .await;
                "🛑 Stop command sent to agent.".to_string()
            }
            _ => "❓ Unknown command.".to_string(),
        }
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
    /// Agent state receiver for status queries
    agent_state_rx: watch::Receiver<AgentState>,
    /// Shutdown signal sender
    shard_manager: Arc<Mutex<Option<Arc<serenity::all::ShardManager>>>>,
}

impl DiscordChannel {
    /// Create a new Discord channel.
    pub fn new(
        token: String,
        message_tx: mpsc::Sender<AgentCommand>,
        agent_state_rx: watch::Receiver<AgentState>,
    ) -> Self {
        Self {
            token,
            running: Arc::new(AtomicBool::new(false)),
            message_tx,
            agent_state_rx,
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
            bot_id: Mutex::new(None),
            agent_state_rx: self.agent_state_rx.clone(),
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
