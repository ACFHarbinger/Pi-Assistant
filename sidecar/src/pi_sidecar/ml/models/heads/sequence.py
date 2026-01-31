"""
Sequence Head.

For sequence-to-sequence tasks like translation, summarization, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .base import Head, HeadConfig, register_head

__all__ = ["SequenceHeadConfig", "SequenceHead"]


@dataclass
class SequenceHeadConfig(HeadConfig):
    """Configuration for sequence head."""
    vocab_size: int = 50000
    tie_weights: bool = True  # Tie with encoder embeddings


@register_head("sequence")
class SequenceHead(Head):
    """
    Sequence prediction head for token-level outputs.
    
    Suitable for:
    - Language modeling
    - Token classification (NER, POS)
    - Sequence-to-sequence generation
    """
    
    def __init__(self, config: SequenceHeadConfig) -> None:
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
        
        self.proj = nn.Sequential(*layers) if layers else nn.Identity()
        self.lm_head = nn.Linear(current_dim, config.vocab_size, bias=False)
    
    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq, dim)
            return_logits: If True, return raw logits
        
        Returns:
            (batch, seq, vocab_size) logits or probabilities
        """
        features = self.proj(features)
        logits = self.lm_head(features)
        
        if return_logits:
            return logits
        
        return torch.softmax(logits, dim=-1)
    
    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """Tie output weights to input embeddings."""
        self.lm_head.weight = embedding_weight
