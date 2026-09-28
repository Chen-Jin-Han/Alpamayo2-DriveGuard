from __future__ import annotations

from collections.abc import Callable

import torch

from .base_attack import attack_mask_like, project_linf

Objective = Callable[[torch.Tensor], torch.Tensor]


def random_noise(
    clean: torch.Tensor,
    epsilon: float,
    seed: int = 42,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    delta = (torch.rand(clean.shape, generator=generator, device=clean.device) * 2 - 1) * epsilon
    delta = delta * attack_mask_like(clean, attacked_cameras, attacked_frames)
    return project_linf(clean + delta, clean, epsilon).detach()


def fgsm(
    clean: torch.Tensor,
    objective: Objective,
    epsilon: float,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
) -> torch.Tensor:
    adv = clean.detach().clone().requires_grad_(True)
    loss = objective(adv)
    gradient = torch.autograd.grad(loss, adv, only_inputs=True)[0]
    update = epsilon * gradient.sign() * attack_mask_like(
        clean, attacked_cameras, attacked_frames
    )
    return project_linf(clean + update, clean, epsilon).detach()


def pgd(
    clean: torch.Tensor,
    objective: Objective,
    epsilon: float,
    step_size: float,
    iterations: int,
    random_start: bool = True,
    seed: int = 42,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
) -> torch.Tensor:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    mask = attack_mask_like(clean, attacked_cameras, attacked_frames)
    if random_start and epsilon > 0:
        generator = torch.Generator(device=clean.device).manual_seed(seed)
        delta = (torch.rand(clean.shape, generator=generator, device=clean.device) * 2 - 1)
        adv = project_linf(clean + delta * epsilon * mask, clean, epsilon)
    else:
        adv = clean.detach().clone()
    for _ in range(iterations):
        adv = adv.detach().requires_grad_(True)
        gradient = torch.autograd.grad(objective(adv), adv, only_inputs=True)[0]
        adv = project_linf(adv + step_size * gradient.sign() * mask, clean, epsilon)
    return adv.detach()


def temporal_pgd(
    clean: torch.Tensor,
    objective: Objective,
    epsilon: float,
    step_size: float,
    iterations: int,
    shared_across_cameras: bool = False,
    seed: int = 42,
    attacked_cameras: list[int] | None = None,
    attacked_frames: list[int] | None = None,
) -> torch.Tensor:
    """Optimize a time-indexed perturbation, optionally universal across cameras."""
    mask = attack_mask_like(clean, attacked_cameras, attacked_frames)
    delta_shape = (1, *clean.shape[1:]) if shared_across_cameras else clean.shape
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    delta = (torch.rand(delta_shape, generator=generator, device=clean.device) * 2 - 1) * epsilon
    for _ in range(iterations):
        delta = delta.detach().requires_grad_(True)
        adv = (clean + delta.expand_as(clean) * mask).clamp(0, 1)
        gradient = torch.autograd.grad(objective(adv), delta, only_inputs=True)[0]
        delta = (delta + step_size * gradient.sign()).clamp(-epsilon, epsilon)
    return project_linf(clean + delta.detach().expand_as(clean) * mask, clean, epsilon).detach()
