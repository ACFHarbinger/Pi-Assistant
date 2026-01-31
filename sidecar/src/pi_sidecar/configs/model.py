from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer."""

    model_id: str
    model: Any
    tokenizer: Any
    metadata: dict = field(default_factory=dict)