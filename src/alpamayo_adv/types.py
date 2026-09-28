from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InferenceOutput:
    """Common output shared by the lightweight and official backends."""

    reasoning: str
    trajectory: torch.Tensor
    visual_features: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)
