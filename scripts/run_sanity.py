#!/usr/bin/env python3
from __future__ import annotations

import torch

from alpamayo_adv.attacks import pgd
from alpamayo_adv.datasets import make_mock_sample
from alpamayo_adv.models import MockAlpamayoWrapper


def main() -> None:
    data = make_mock_sample(height=32, width=48)
    clean = data["image_frames"]
    wrapper = MockAlpamayoWrapper()
    clean_out = wrapper.forward_clean(data)

    def objective(frames: torch.Tensor) -> torch.Tensor:
        return (wrapper.differentiable_forward(frames).visual_features - clean_out.visual_features).square().mean()

    epsilon_zero = pgd(clean, objective, epsilon=0.0, step_size=0.0, iterations=2)
    assert torch.equal(clean, epsilon_zero), "epsilon=0 must reproduce clean pixels exactly"
    epsilon = 16 / 255
    adv = pgd(clean, objective, epsilon=epsilon, step_size=2 / 255, iterations=3)
    assert float((adv - clean).abs().max()) <= epsilon + 1e-6
    assert 0 <= float(adv.min()) <= float(adv.max()) <= 1
    assert adv.shape == clean.shape
    print("sanity checks passed: epsilon=0, Linf budget, legal pixels, camera/time order")


if __name__ == "__main__":
    main()

