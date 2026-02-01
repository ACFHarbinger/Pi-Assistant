use anyhow::Result;
use pi_core::agent_types::{CostConfig, TokenUsage};

pub struct CostManager {
    usage: TokenUsage,
    config: CostConfig,
}

impl CostManager {
    pub fn new(config: CostConfig) -> Self {
        Self {
            usage: TokenUsage::default(),
            config,
        }
    }

    pub fn add_usage(&mut self, usage: &TokenUsage) {
        self.usage.prompt_tokens += usage.prompt_tokens;
        self.usage.completion_tokens += usage.completion_tokens;
        self.usage.total_tokens += usage.total_tokens;
    }

    pub fn check_budget(&self) -> Result<()> {
        if let Some(limit) = self.config.max_tokens_per_session {
            if self.usage.total_tokens >= limit {
                anyhow::bail!(
                    "Token budget exceeded: {} >= {}",
                    self.usage.total_tokens,
                    limit
                );
            }
        }
        Ok(())
    }

    pub fn get_usage(&self) -> TokenUsage {
        self.usage.clone()
    }

    /// Estimate cost in USD based on provider rates.
    pub fn estimate_cost(&self, provider: &str, model: Option<&str>) -> f64 {
        // Rates per 1M tokens (approximate defaults)
        let (prompt_rate, completion_rate) = match provider {
            "openai" => {
                match model {
                    Some(m) if m.contains("gpt-4") => (30.0, 60.0), // GPT-4
                    _ => (0.50, 1.50),                              // GPT-3.5 Turbo
                }
            }
            "anthropic" => (3.0, 15.0), // Claude 3.5 Sonnet
            "local" | "ollama" => (0.0, 0.0),
            _ => (0.0, 0.0),
        };

        let prompt_cost = (self.usage.prompt_tokens as f64 / 1_000_000.0) * prompt_rate;
        let completion_cost = (self.usage.completion_tokens as f64 / 1_000_000.0) * completion_rate;

        prompt_cost + completion_cost
    }
}
