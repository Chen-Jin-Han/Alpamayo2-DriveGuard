from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _image(tensor: torch.Tensor):
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def save_comparison(
    clean_frames: torch.Tensor,
    adv_frames: torch.Tensor,
    clean_traj: torch.Tensor,
    adv_traj: torch.Tensor,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diff = (adv_frames[0, -1] - clean_frames[0, -1]).abs()
    scale = max(float(diff.max().item()), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(_image(clean_frames[0, -1])); axes[0].set_title("clean")
    axes[1].imshow(_image(adv_frames[0, -1])); axes[1].set_title("adversarial")
    axes[2].imshow(_image(diff / scale)); axes[2].set_title("|delta| normalized")
    axes[3].plot(clean_traj[:, 0].detach().cpu(), clean_traj[:, 1].detach().cpu(), label="clean")
    axes[3].plot(adv_traj[:, 0].detach().cpu(), adv_traj[:, 1].detach().cpu(), label="adv")
    axes[3].axis("equal"); axes[3].legend(); axes[3].set_title("trajectory")
    for axis in axes[:3]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
