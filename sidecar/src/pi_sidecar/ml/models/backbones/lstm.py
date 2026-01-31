"""
LSTM Backbone.

Bidirectional LSTM for sequential feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import Backbone, BackboneConfig, register_backbone

__all__ = ["LSTMBackboneConfig", "LSTMBackbone"]


@dataclass
class LSTMBackboneConfig(BackboneConfig):
    """Configuration for LSTM backbone."""
    bidirectional: bool = True
    proj_size: int = 0  # Projection size (0 = disabled)


@register_backbone("lstm")
class LSTMBackbone(Backbone):
    """
    Stacked BiLSTM backbone.
    
    Suitable for:
    - Time series modeling
    - Sequence labeling
    - Any task requiring sequential context
    """
    
    def __init__(self, config: LSTMBackboneConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        input_dim = config.input_dim or config.hidden_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0,
            proj_size=config.proj_size,
        )
        
        # Output dim depends on bidirectional and proj_size
        lstm_out_dim = (
            config.proj_size if config.proj_size > 0 else config.hidden_dim
        )
        if config.bidirectional:
            lstm_out_dim *= 2
        
        self._output_dim = lstm_out_dim
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, features)
            lengths: Optional (batch,) tensor of sequence lengths
        
        Returns:
            (batch, seq, hidden_dim * num_directions)
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        
        if lengths is not None:
            # Pack for efficiency
            x = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)
        
        x = self.norm(x)
        return x
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
