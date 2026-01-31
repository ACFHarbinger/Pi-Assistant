
from __future__ import annotations

from typing import Any

import torch.nn as nn

from pi_sidecar.ml.models.convolutional.cnn import CNN
from pi_sidecar.ml.models.convolutional.resnet import ResNet
from pi_sidecar.ml.models.convolutional.capsule import CapsuleNet
from pi_sidecar.ml.models.convolutional.dcign import DCIGN
from pi_sidecar.ml.models.convolutional.dcn import DCN
from pi_sidecar.ml.models.convolutional.deconv import DeconvNet
from pi_sidecar.ml.models.factories.base import NeuralComponentFactory


class ConvolutionalFactory(NeuralComponentFactory):
    """Factory for convolutional networks."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get convolutional model by name."""
        name = name.lower()
        if "resnet" in name:
            return ResNet(**kwargs)
        elif "capsule" in name:
            return CapsuleNet(**kwargs)
        elif "dcign" in name:
            return DCIGN(**kwargs)
        elif "deconv" in name:
            return DeconvNet(**kwargs)
        elif "dcn" in name:
            return DCN(**kwargs)
        elif "cnn" in name:
            return CNN(**kwargs)
        else:
            raise ValueError(
                f"Unknown convolutional model: {name}. "
                f"Available: cnn, resnet, capsule, dcign, deconv, dcn"
            )
