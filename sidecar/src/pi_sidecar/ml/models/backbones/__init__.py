"""
Backbones Package.

Task-agnostic feature extractors that can be composed with task-specific heads.
"""

from .base import BACKBONE_REGISTRY, Backbone, BackboneConfig, register_backbone
from .conv import ConvBackbone, ConvBackboneConfig
from .lstm import LSTMBackbone, LSTMBackboneConfig
from .mamba import MambaBackbone, MambaBackboneConfig
from .transformer import TransformerBackbone, TransformerBackboneConfig

__all__ = [
    # Base
    "Backbone",
    "BackboneConfig",
    "BACKBONE_REGISTRY",
    "register_backbone",
    # Implementations
    "TransformerBackbone",
    "TransformerBackboneConfig",
    "LSTMBackbone",
    "LSTMBackboneConfig",
    "MambaBackbone",
    "MambaBackboneConfig",
    "ConvBackbone",
    "ConvBackboneConfig",
]
