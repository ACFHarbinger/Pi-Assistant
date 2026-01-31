"""
Transformer Backbone.

Encoder-only transformer for sequence modeling tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .base import Backbone, BackboneConfig, register_backbone

__all__ = ["TransformerBackboneConfig", "TransformerBackbone"]


@dataclass
class TransformerBackboneConfig(BackboneConfig):
    """Configuration for Transformer backbone."""
    num_heads: int = 8
    ff_dim: int = 1024
    max_seq_len: int = 512
    prenorm: bool = True


@register_backbone("transformer")
class TransformerBackbone(Backbone):
    """
    Encoder-only Transformer backbone.
    
    Suitable for:
    - Sequence classification
    - Token classification
    - Feature extraction from sequential data
    """
    
    def __init__(self, config: TransformerBackboneConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        # Input projection
        input_dim = config.input_dim or config.hidden_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=config.prenorm,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, features)
            attention_mask: Optional (batch, seq) boolean mask
        
        Returns:
            (batch, seq, hidden_dim) or (batch, hidden_dim) if pooled
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Convert attention mask to transformer format if provided
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask  # Transformer expects True for masked
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        return x
    
    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim
