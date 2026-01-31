"""
Regression Head.

For continuous value prediction tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from .base import Head, HeadConfig, register_head

__all__ = ["RegressionHeadConfig", "RegressionHead"]


@dataclass
class RegressionHeadConfig(HeadConfig):
    """Configuration for regression head."""
    output_dim: int = 1
    pool_type: Literal["mean", "last", "none"] = "last"
    output_activation: Literal["none", "sigmoid", "tanh", "softplus"] = "none"


@register_head("regression")
class RegressionHead(Head):
    """
    Regression head for continuous outputs.
    
    Supports:
    - Single-target and multi-target regression
    - Various output activations for bounded outputs
    - Sequence-to-value and sequence-to-sequence
    """
    
    def __init__(self, config: RegressionHeadConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        layers = []
        current_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, config.output_dim))
        self.regressor = nn.Sequential(*layers)
        
        self.pool_type = config.pool_type
        self.activation = self._get_activation(config.output_activation)
    
    def _get_activation(self, name: str) -> nn.Module | None:
        if name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "softplus":
            return nn.Softplus()
        return None
    
    def forward(
        self,
        features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq, dim) or (batch, dim)
        
        Returns:
            (batch, output_dim) or (batch, seq, output_dim) depending on pool_type
        """
        # Pool sequence dimension if present
        if features.dim() == 3:
            if self.pool_type == "mean":
                features = features.mean(dim=1)
            elif self.pool_type == "last":
                features = features[:, -1]
            # "none" keeps the sequence for seq2seq
        
        output = self.regressor(features)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
