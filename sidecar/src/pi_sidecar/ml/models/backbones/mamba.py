"""
Mamba Backbone.

State-space model backbone for long-range sequence modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .base import Backbone, BackboneConfig, register_backbone

__all__ = ["MambaBackboneConfig", "MambaBackbone"]


@dataclass
class MambaBackboneConfig(BackboneConfig):
    """Configuration for Mamba backbone."""
    state_dim: int = 16  # SSM state expansion factor
    conv_size: int = 4   # Local convolution width
    expand_factor: int = 2


class MambaBlock(nn.Module):
    """Simplified Mamba block (S6-style selective SSM)."""
    
    def __init__(self, dim: int, state_dim: int, conv_size: int, expand_factor: int) -> None:
        super().__init__()
        inner_dim = dim * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        
        # Conv1d for local context
        self.conv = nn.Conv1d(
            inner_dim, inner_dim, conv_size,
            padding=conv_size - 1, groups=inner_dim
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(inner_dim, state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(state_dim, inner_dim, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        
        self.state_dim = state_dim
        self.inner_dim = inner_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, dim)
        Returns:
            (batch, seq, dim)
        """
        batch, seq_len, _ = x.shape
        
        # Split into main path and gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        
        # Local convolution
        x_conv = self.conv(x_main.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = nn.functional.silu(x_conv)
        
        # SSM (simplified - full implementation would use scan)
        # This is a placeholder that maintains the interface
        y = x_conv * nn.functional.silu(z)
        
        return self.out_proj(y)


@register_backbone("mamba")
class MambaBackbone(Backbone):
    """
    Mamba (State-Space Model) backbone.
    
    Suitable for:
    - Long-range sequence modeling
    - Language modeling
    - Time series with long dependencies
    """
    
    def __init__(self, config: MambaBackboneConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        input_dim = config.input_dim or config.hidden_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                config.hidden_dim,
                config.state_dim,
                config.conv_size,
                config.expand_factor,
            )
            for _ in range(config.num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, features)
        Returns:
            (batch, seq, hidden_dim)
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + x
        
        x = self.final_norm(x)
        return x
    
    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim
