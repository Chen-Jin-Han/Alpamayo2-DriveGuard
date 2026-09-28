from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    return None


class FeatureCapture(AbstractContextManager["FeatureCapture"]):
    """Capture a module output without changing the model forward path."""

    def __init__(self, module: torch.nn.Module, detach: bool = True) -> None:
        self.module = module
        self.detach = detach
        self.value: torch.Tensor | None = None
        self._handle: Any = None

    def __enter__(self) -> "FeatureCapture":
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                self.value = tensor.detach() if self.detach else tensor

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
