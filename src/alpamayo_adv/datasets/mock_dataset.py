from __future__ import annotations

from typing import Any

import torch


def make_mock_sample(
    cameras: int = 6,
    frames: int = 4,
    height: int = 96,
    width: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic synthetic clip for code-path tests, never for model claims."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
    )
    clips = []
    for camera in range(cameras):
        camera_frames = []
        for frame in range(frames):
            base = torch.stack(
                [
                    (xx + camera / max(cameras, 1) * 0.15).remainder(1.0),
                    (yy + frame / max(frames, 1) * 0.10).remainder(1.0),
                    0.5 + 0.25 * torch.sin((xx + yy) * 6.28 + frame * 0.2),
                ]
            )
            noise = torch.rand(base.shape, generator=generator) * 0.01
            camera_frames.append((base + noise).clamp(0, 1))
        clips.append(torch.stack(camera_frames))
    image_frames = torch.stack(clips)
    horizon = 64
    steps = torch.arange(1, horizon + 1, dtype=torch.float32)
    gt = torch.stack([steps * 0.11, torch.zeros_like(steps), torch.zeros_like(steps)], dim=-1)
    return {
        "image_frames": image_frames,
        "camera_indices": torch.tensor([0, 1, 2, 3, 5, 6][:cameras]),
        "ego_future_xyz": gt,
        "clip_id": "synthetic-sanity-only",
    }


def make_official_synthetic_sample(
    height: int = 224,
    width: int = 384,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a release-shape synthetic sample for official model smoke tests only."""
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES

    sample = make_mock_sample(cameras=6, frames=4, height=height, width=width, seed=seed)
    camera_ids = (0, 1, 2, 3, 5, 6)
    names_by_id = {value: key for key, value in CAMERA_NAMES_TO_INDICES.items()}
    sample["image_frames"] = (sample["image_frames"] * 255).round().to(torch.uint8)
    sample["camera_indices"] = torch.tensor(camera_ids, dtype=torch.int64)
    sample["camera_names"] = [names_by_id[index] for index in camera_ids]
    sample["ego_history_xyz"] = torch.zeros(1, 1, 16, 3)
    sample["ego_history_rot"] = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 16, 1, 1)
    sample["clip_id"] = "synthetic-official-smoke-only"
    return sample
