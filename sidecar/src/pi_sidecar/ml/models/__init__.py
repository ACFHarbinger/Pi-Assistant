from __future__ import annotations

from pi_sidecar.ml.models.base import BaseModel, BaseEncoder, BaseDecoder, ModelProtocol
from pi_sidecar.ml.models.registry import MODEL_REGISTRY, get_model, register_model
from pi_sidecar.ml.models.composed import ComposedModel, build_model
from pi_sidecar.ml.models.time_series import TimeSeriesBackbone

# Flattened Architecture Modules
from .attention import *  # noqa: F403
from .autoencoders import *  # noqa: F403
from .backbones import *  # noqa: F403
from .competitive import *  # noqa: F403
from .convolutional import *  # noqa: F403
from .general import *  # noqa: F403
from .heads import *  # noqa: F403
from .memory import *  # noqa: F403
from .probabilistic import *  # noqa: F403
from .recurrent import *  # noqa: F403
from .spiking import *  # noqa: F403

__all__ = [
    "BaseModel",
    "BaseEncoder",
    "BaseDecoder",
    "ModelProtocol",
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
    "ComposedModel",
    "build_model",
    "TimeSeriesBackbone",
]
