from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def gaussian_denoise(frames: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([1.0, 2.0, 1.0], device=frames.device, dtype=frames.dtype)
    kernel = (kernel[:, None] * kernel[None, :]) / 16.0
    weight = kernel.expand(3, 1, 3, 3)
    flat = frames.flatten(0, 1)
    return F.conv2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), weight, groups=3).view_as(frames)


def temporal_smooth(frames: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    result = frames.clone()
    for index in range(1, frames.shape[1]):
        result[:, index] = alpha * frames[:, index] + (1.0 - alpha) * result[:, index - 1]
    return result.clamp(0, 1)


def jpeg_compress(frames: torch.Tensor, quality: int = 75) -> torch.Tensor:
    restored = []
    for camera in frames.detach().cpu():
        restored_camera = []
        for frame in camera:
            array = (frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            buffer = BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
            restored_camera.append(torch.from_numpy(decoded).permute(2, 0, 1))
        restored.append(torch.stack(restored_camera))
    return torch.stack(restored).to(frames.device)
