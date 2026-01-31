"""
Base Head Classes.

Heads are task-specific layers that attach to backbone outputs
for classification, regression, RL policies, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

__all__ = ["HeadConfig", "Head", "HEAD_REGISTRY", "register_head"]


@dataclass
class HeadConfig:
    """Configuration for head modules."""
    input_dim: int = 256  # Must match backbone.output_dim
    output_dim: int = 10  # Task-specific (num_classes, action_dim, etc.)
    hidden_dims: tuple[int, ...] = ()  # Optional MLP layers
    dropout: float = 0.1
    extra: dict[str, Any] = field(default_factory=dict)


class Head(nn.Module, ABC):
    """
    Abstract base class for task-specific heads.
    
    Heads take backbone features and produce task-specific outputs.
    """
    
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass producing task-specific output.
        
        Args:
            features: Output from backbone (batch, ..., backbone_dim)
            **kwargs: Task-specific arguments
        
        Returns:
            Task-specific output tensor
        """
        pass


# Global registry for head classes
HEAD_REGISTRY: dict[str, type[Head]] = {}


def register_head(name: str):
    """Decorator to register a head class."""
    def decorator(cls: type[Head]) -> type[Head]:
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator
