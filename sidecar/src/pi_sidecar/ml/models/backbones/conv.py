"""
Convolutional Backbone.

1D/2D ConvNets for spatial/temporal feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from .base import Backbone, BackboneConfig, register_backbone

__all__ = ["ConvBackboneConfig", "ConvBackbone"]


@dataclass
class ConvBackboneConfig(BackboneConfig):
    """Configuration for Convolutional backbone."""
    kernel_sizes: tuple[int, ...] = (7, 5, 3, 3)
    channels: tuple[int, ...] = (64, 128, 256, 256)
    conv_type: Literal["1d", "2d"] = "1d"
    pool_type: Literal["max", "avg"] = "max"


@register_backbone("conv")
class ConvBackbone(Backbone):
    """
    Convolutional backbone for spatial/temporal data.
    
    Suitable for:
    - Image classification
    - Audio processing
    - Time series with local patterns
    """
    
    def __init__(self, config: ConvBackboneConfig) -> None:
        super().__init__(config)
        self.cfg = config
        
        input_dim = config.input_dim or 1
        conv_class = nn.Conv1d if config.conv_type == "1d" else nn.Conv2d
        norm_class = nn.BatchNorm1d if config.conv_type == "1d" else nn.BatchNorm2d
        pool_class = (
            nn.MaxPool1d if config.conv_type == "1d" else nn.MaxPool2d
        ) if config.pool_type == "max" else (
            nn.AvgPool1d if config.conv_type == "1d" else nn.AvgPool2d
        )
        
        layers = []
        in_channels = input_dim
        
        for i, (kernel_size, out_channels) in enumerate(
            zip(config.kernel_sizes, config.channels)
        ):
            layers.extend([
                conv_class(
                    in_channels, out_channels, kernel_size,
                    padding=kernel_size // 2
                ),
                norm_class(out_channels),
                nn.GELU(),
                pool_class(2),
                nn.Dropout(config.dropout),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self._output_dim = config.channels[-1] if config.channels else config.hidden_dim
        
        # Optional projection to hidden_dim
        if self._output_dim != config.hidden_dim:
            self.proj = nn.Linear(self._output_dim, config.hidden_dim)
            self._output_dim = config.hidden_dim
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length) for 1D or (batch, channels, height, width) for 2D
        Returns:
            (batch, output_dim) after global pooling
        """
        x = self.conv_layers(x)
        
        # Global average pooling
        if self.cfg.conv_type == "1d":
            x = x.mean(dim=-1)  # (batch, channels)
        else:
            x = x.mean(dim=(-2, -1))  # (batch, channels)
        
        if self.proj is not None:
            x = self.proj(x)
        
        return x
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
