from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_REQUIRED_SAMPLE_KEYS = {
    "image_frames",
    "camera_indices",
    "ego_history_xyz",
    "ego_history_rot",
}


def load_official_sample(clip_id: str, t0_us: int = 5_100_000) -> dict[str, Any]:
    """Load and select the exact official six-camera/four-frame trajectory profile."""
    try:
        from alpamayo2_super.input_profiles import select_task_input
        from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    except ImportError as exc:
        raise RuntimeError("Install the pinned third_party/alpamayo2 environment first") from exc
    source = load_physical_aiavdataset(clip_id=clip_id, t0_us=t0_us, num_frames=4)
    return select_task_input(source, "trajectory")


def load_cached_sample(path: str | Path) -> dict[str, Any]:
    """Load a previously materialized official input without network credentials."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"cached PhysicalAI sample does not exist: {path}")
    sample = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sample, dict):
        raise TypeError(f"cached PhysicalAI sample must be a dict, got {type(sample).__name__}")
    missing = sorted(_REQUIRED_SAMPLE_KEYS.difference(sample))
    if missing:
        raise ValueError(f"cached PhysicalAI sample is missing required keys: {missing}")
    frames = sample["image_frames"]
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise ValueError("cached image_frames must have shape [camera, frame, channel, height, width]")
    return sample
