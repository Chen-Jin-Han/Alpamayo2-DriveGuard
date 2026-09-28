from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


AttackKind = Literal["fgsm", "pgd", "temporal"]


@dataclass(frozen=True)
class SurrogateAttackInfo:
    kind: str
    epsilon: float
    step_size: float
    iterations: int
    attacked_cameras: tuple[int, ...]
    attacked_frames: tuple[int, ...]
    shared_across_cameras: bool


class ResNet18Surrogate(torch.nn.Module):
    """Frozen ImageNet ResNet-18 used only to craft transferable perturbations."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        x = (images - self.mean) / self.std
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        features = torch.flatten(model.avgpool(x), 1)
        return model.fc(features), features


def _attack_mask(
    cameras: int,
    frames: int,
    attacked_cameras: tuple[int, ...],
    attacked_frames: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(cameras, 1, 1, 1, 1, device=device)
    mask[list(attacked_cameras)] = 1
    frame_mask = torch.zeros(1, frames, 1, 1, 1, device=device)
    frame_mask[:, list(attacked_frames)] = 1
    return mask * frame_mask


def _normalize_kind(kind: str) -> AttackKind:
    if kind not in {"fgsm", "pgd", "temporal"}:
        raise ValueError(f"unsupported surrogate attack: {kind}")
    return kind  # type: ignore[return-value]


def generate_resnet18_transfer_attack(
    clean_uint8: torch.Tensor,
    kind: str,
    epsilon: float,
    step_size: float,
    iterations: int,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
    seed: int = 42,
    resize: tuple[int, int] = (224, 384),
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, SurrogateAttackInfo]:
    """Craft on a frozen image model, then lift the perturbation to native resolution.

    This is explicitly a transfer baseline: Alpamayo weights are never differentiated.
    Temporal mode learns one time-indexed perturbation shared across attacked cameras.
    """
    kind = _normalize_kind(kind)
    if clean_uint8.dtype != torch.uint8 or clean_uint8.ndim != 5:
        raise ValueError("clean_uint8 must be uint8 [camera, frame, channel, height, width]")
    cameras, frames, channels, height, width = clean_uint8.shape
    if channels != 3:
        raise ValueError("expected RGB inputs")
    attacked = tuple(range(cameras)) if attacked_cameras is None else tuple(attacked_cameras)
    if not attacked or min(attacked) < 0 or max(attacked) >= cameras:
        raise ValueError(f"attacked_cameras must be within [0, {cameras - 1}]")
    attacked_time = tuple(range(frames)) if attacked_frames is None else tuple(attacked_frames)
    if not attacked_time or min(attacked_time) < 0 or max(attacked_time) >= frames:
        raise ValueError(f"attacked_frames must be within [0, {frames - 1}]")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    target_device = torch.device(device)
    torch.manual_seed(seed)
    clean = clean_uint8.float().div(255).reshape(cameras * frames, channels, height, width)
    small = F.interpolate(clean, size=resize, mode="bilinear", align_corners=False, antialias=True)
    small = small.reshape(cameras, frames, channels, *resize).to(target_device)
    model = ResNet18Surrogate().to(target_device)
    mask = _attack_mask(cameras, frames, attacked, attacked_time, target_device)

    with torch.no_grad():
        clean_logits, clean_features = model(small.flatten(0, 1))
        clean_features = clean_features.reshape(cameras, frames, -1)
        pseudo_labels = clean_logits.argmax(dim=-1)

    shared = kind == "temporal"
    delta_shape = (1, frames, channels, *resize) if shared else small.shape
    generator = torch.Generator(device=target_device).manual_seed(seed)
    if kind == "fgsm":
        delta = torch.zeros(delta_shape, device=target_device)
        attack_iterations = 1
        attack_step = epsilon
    else:
        delta = (
            torch.rand(delta_shape, generator=generator, device=target_device) * 2 - 1
        ) * epsilon
        attack_iterations = iterations
        attack_step = step_size

    for _ in range(attack_iterations):
        delta = delta.detach().requires_grad_(True)
        expanded = delta.expand(cameras, -1, -1, -1, -1) if shared else delta
        adv = (small + expanded * mask).clamp(0, 1)
        logits, features = model(adv.flatten(0, 1))
        features = features.reshape(cameras, frames, -1)
        classification = F.cross_entropy(logits, pseudo_labels)
        feature_shift = 1 - F.cosine_similarity(
            features.flatten(0, 1), clean_features.flatten(0, 1), dim=-1
        ).mean()
        temporal_shift = (
            (features[:, 1:] - features[:, :-1])
            - (clean_features[:, 1:] - clean_features[:, :-1])
        ).square().mean()
        if kind == "fgsm":
            objective = classification
        elif kind == "pgd":
            objective = feature_shift + 0.1 * classification
        else:
            objective = feature_shift + temporal_shift + 0.1 * classification
        gradient = torch.autograd.grad(objective, delta, only_inputs=True)[0]
        delta = (delta + attack_step * gradient.sign()).clamp(-epsilon, epsilon)

    expanded = delta.detach().expand(cameras, -1, -1, -1, -1) if shared else delta.detach()
    expanded = expanded * mask
    large_delta = F.interpolate(
        expanded.flatten(0, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(cameras, frames, channels, height, width)

    clean_i16 = clean_uint8.to(device=target_device, dtype=torch.int16)
    candidate = torch.round(clean_i16.float() + large_delta * 255).to(torch.int16)
    epsilon_pixels = int(round(epsilon * 255))
    lower = (clean_i16 - epsilon_pixels).clamp(0, 255)
    upper = (clean_i16 + epsilon_pixels).clamp(0, 255)
    adv_uint8 = torch.maximum(lower, torch.minimum(upper, candidate)).to(torch.uint8).cpu()
    info = SurrogateAttackInfo(
        kind=kind,
        epsilon=epsilon,
        step_size=attack_step,
        iterations=attack_iterations,
        attacked_cameras=attacked,
        attacked_frames=attacked_time,
        shared_across_cameras=shared,
    )
    return adv_uint8, info
