"""
Classification Head.

For multi-class and multi-label classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Head, HeadConfig, register_head

__all__ = ["ClassificationHeadConfig", "ClassificationHead"]


@dataclass
class ClassificationHeadConfig(HeadConfig):
    """Configuration for classification head."""
    num_classes: int = 10
    pool_type: Literal["cls", "mean", "max", "none"] = "mean"
    multi_label: bool = False


@register_head("classification")
class ClassificationHead(Head):
    """
    Classification head for categorical outputs.
    
    Supports:
    - Multi-class (softmax)
    - Multi-label (sigmoid)
    - Various pooling strategies
    """
    
    def __init__(self, config: ClassificationHeadConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        layers = []
        current_dim = config.input_dim
        
        # Optional MLP layers
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)
        self.pool_type = config.pool_type
        self.multi_label = config.multi_label
    
    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq, dim) or (batch, dim)
            return_logits: If True, return raw logits; else return probabilities
        
        Returns:
            (batch, num_classes) logits or probabilities
        """
        # Pool sequence dimension if present
        if features.dim() == 3:
            if self.pool_type == "cls":
                features = features[:, 0]  # First token
            elif self.pool_type == "mean":
                features = features.mean(dim=1)
            elif self.pool_type == "max":
                features = features.max(dim=1).values
            # "none" keeps the sequence
        
        logits = self.classifier(features)
        
        if return_logits:
            return logits
        
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
