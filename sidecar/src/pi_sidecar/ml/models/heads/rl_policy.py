"""
RL Policy Head.

Actor-Critic head for reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .base import Head, HeadConfig, register_head

__all__ = ["RLPolicyHeadConfig", "RLPolicyHead", "PolicyOutput"]


class PolicyOutput(NamedTuple):
    """Output from RL policy head."""
    action_logits: torch.Tensor  # For discrete; mean for continuous
    action_std: torch.Tensor | None  # For continuous actions
    value: torch.Tensor  # State value estimate


@dataclass
class RLPolicyHeadConfig(HeadConfig):
    """Configuration for RL policy head."""
    action_dim: int = 4
    continuous: bool = False  # Discrete vs continuous actions
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    separate_value_head: bool = True


@register_head("rl_policy")
class RLPolicyHead(Head):
    """
    Actor-Critic head for reinforcement learning.
    
    Supports:
    - Discrete actions (Categorical distribution)
    - Continuous actions (Gaussian distribution)
    - Shared or separate value network
    """
    
    def __init__(self, config: RLPolicyHeadConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        # Actor network
        actor_layers = []
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            actor_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),  # Tanh is standard for RL
            ])
            current_dim = hidden_dim
        
        if config.continuous:
            # Mean and log_std for Gaussian policy
            actor_layers.append(nn.Linear(current_dim, config.action_dim))
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        else:
            # Logits for Categorical policy
            actor_layers.append(nn.Linear(current_dim, config.action_dim))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic (value) network
        if config.separate_value_head:
            critic_layers = []
            current_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                critic_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.Tanh(),
                ])
                current_dim = hidden_dim
            critic_layers.append(nn.Linear(current_dim, 1))
            self.critic = nn.Sequential(*critic_layers)
        else:
            # Shared features, just a final layer
            self.critic = nn.Linear(
                config.hidden_dims[-1] if config.hidden_dims else config.input_dim,
                1,
            )
    
    def forward(
        self,
        features: torch.Tensor,
        **kwargs: Any,
    ) -> PolicyOutput:
        """
        Args:
            features: (batch, dim) backbone features
        
        Returns:
            PolicyOutput with action distribution params and value
        """
        # Pool if sequence
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        # Actor output
        action_output = self.actor(features)
        
        if self.cfg.continuous:
            action_mean = action_output
            # Clamp log_std for numerical stability
            log_std = self.log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
            action_std = log_std.exp().expand_as(action_mean)
        else:
            action_mean = action_output  # These are logits
            action_std = None
        
        # Critic output
        value = self.critic(features).squeeze(-1)
        
        return PolicyOutput(
            action_logits=action_mean,
            action_std=action_std,
            value=value,
        )
    
    def get_action(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            (action, log_prob, value)
        """
        output = self.forward(features)
        
        if self.cfg.continuous:
            dist = Normal(output.action_logits, output.action_std)
            if deterministic:
                action = output.action_logits
            else:
                action = dist.rsample()  # Reparameterized sample
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = Categorical(logits=output.action_logits)
            if deterministic:
                action = output.action_logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, output.value
