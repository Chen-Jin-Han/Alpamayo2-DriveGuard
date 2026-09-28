from __future__ import annotations

import torch


def epsilon_from_config(value: float | int | str) -> float:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def project_linf(candidate: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = (candidate - clean).clamp(-epsilon, epsilon)
    return (clean + delta).clamp(0.0, 1.0)


def perturbation_stats(clean: torch.Tensor, adv: torch.Tensor) -> dict[str, float]:
    delta = (adv - clean).detach()
    return {
        "linf": float(delta.abs().max().item()),
        "l2_mean": float(delta.square().flatten(2).sum(-1).sqrt().mean().item()),
        "pixel_min": float(adv.min().item()),
        "pixel_max": float(adv.max().item()),
    }


def attack_mask_like(
    frames: torch.Tensor,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
) -> torch.Tensor:
    mask = torch.zeros_like(frames)
    cameras = list(range(frames.shape[0])) if attacked_cameras is None else attacked_cameras
    times = list(range(frames.shape[1])) if attacked_frames is None else attacked_frames
    for index in cameras:
        if index < 0 or index >= frames.shape[0]:
            raise ValueError(f"camera index {index} is outside [0, {frames.shape[0]})")
    for index in times:
        if index < 0 or index >= frames.shape[1]:
            raise ValueError(f"frame index {index} is outside [0, {frames.shape[1]})")
    for camera in cameras:
        for time in times:
            mask[camera, time] = 1
    return mask


def camera_mask_like(frames: torch.Tensor, attacked_cameras: list[int] | None) -> torch.Tensor:
    return attack_mask_like(frames, attacked_cameras=attacked_cameras)
