"""
Base Backbone Classes.

Backbones are task-agnostic feature extractors that can be composed
with task-specific heads for classification, regression, RL, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

__all__ = ["BackboneConfig", "Backbone", "BACKBONE_REGISTRY", "register_backbone"]


@dataclass
class BackboneConfig:
    """Configuration for backbone models."""
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    input_dim: int | None = None  # Set dynamically based on data
    extra: dict[str, Any] = field(default_factory=dict)


class Backbone(nn.Module, ABC):
    """
    Abstract base class for task-agnostic backbones.
    
    Backbones extract features from raw inputs and produce
    a fixed-size representation that heads can use.
    """
    
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass producing feature representation.
        
        Args:
            x: Input tensor of shape (batch, seq, features) or (batch, features)
            **kwargs: Additional arguments (e.g., attention mask)
        
        Returns:
            Feature tensor suitable for downstream heads
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the backbone's output features."""
        pass
    
    def freeze(self) -> None:
        """Freeze all backbone parameters for transfer learning."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True


# Global registry for backbone classes
BACKBONE_REGISTRY: dict[str, type[Backbone]] = {}


def register_backbone(name: str):
    """Decorator to register a backbone class."""
    def decorator(cls: type[Backbone]) -> type[Backbone]:
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator
