"""
Composed Model Factory.

Builds complete models by combining backbones with task-specific heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .backbones import BACKBONE_REGISTRY, Backbone, BackboneConfig
from .heads import HEAD_REGISTRY, Head, HeadConfig

__all__ = ["ComposedModel", "ComposedModelConfig", "build_model"]


@dataclass
class ComposedModelConfig:
    """Configuration for composed models."""
    backbone: str  # Name in BACKBONE_REGISTRY
    head: str  # Name in HEAD_REGISTRY
    backbone_config: dict[str, Any]
    head_config: dict[str, Any]
    freeze_backbone: bool = False


class ComposedModel(nn.Module):
    """
    A complete model composed of a backbone and head.
    
    This is the standard way to build task-specific models:
    - Select a backbone (e.g., transformer, lstm, mamba)
    - Attach a head (e.g., classification, regression, rl_policy)
    - Optionally freeze the backbone for transfer learning
    """
    
    def __init__(
        self,
        backbone: Backbone,
        head: Head,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        
        if freeze_backbone:
            self.backbone.freeze()
    
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """
        Forward pass through backbone and head.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to both backbone and head
        
        Returns:
            Head output (task-specific)
        """
        features = self.backbone(x, **kwargs)
        return self.head(features, **kwargs)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone for transfer learning."""
        self.backbone.freeze()
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        self.backbone.unfreeze()


def build_model(
    backbone_name: str,
    head_name: str,
    backbone_config: dict[str, Any] | None = None,
    head_config: dict[str, Any] | None = None,
    freeze_backbone: bool = False,
) -> ComposedModel:
    """
    Build a complete model from registry.
    
    Args:
        backbone_name: Name of backbone in BACKBONE_REGISTRY
        head_name: Name of head in HEAD_REGISTRY
        backbone_config: Configuration dict for backbone
        head_config: Configuration dict for head
        freeze_backbone: Whether to freeze backbone parameters
    
    Returns:
        ComposedModel ready for training
    
    Example:
        model = build_model(
            backbone_name="transformer",
            head_name="classification",
            backbone_config={"hidden_dim": 512, "num_layers": 6},
            head_config={"num_classes": 10},
        )
    """
    backbone_config = backbone_config or {}
    head_config = head_config or {}
    
    # Get classes from registries
    if backbone_name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {available}")
    
    if head_name not in HEAD_REGISTRY:
        available = list(HEAD_REGISTRY.keys())
        raise ValueError(f"Unknown head '{head_name}'. Available: {available}")
    
    backbone_cls = BACKBONE_REGISTRY[backbone_name]
    head_cls = HEAD_REGISTRY[head_name]
    
    # Find the config class for backbone
    # Look for a config class with matching name pattern
    config_class = BackboneConfig
    for attr_name in dir(backbone_cls):
        if "Config" in attr_name:
            config_class = getattr(backbone_cls, attr_name, BackboneConfig)
            break
    
    # Build components
    backbone = backbone_cls(config_class(**backbone_config))
    
    # Ensure head knows backbone output dim
    head_config.setdefault("input_dim", backbone.output_dim)
    
    head_config_class = HeadConfig
    for attr_name in dir(head_cls):
        if "Config" in attr_name:
            head_config_class = getattr(head_cls, attr_name, HeadConfig)
            break
    
    head = head_cls(head_config_class(**head_config))
    
    return ComposedModel(backbone, head, freeze_backbone=freeze_backbone)
