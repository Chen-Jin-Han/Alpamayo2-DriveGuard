from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alpamayo_adv.types import InferenceOutput
from .feature_hooks import FeatureCapture


def validate_image_frames(frames: torch.Tensor) -> torch.Tensor:
    """Validate `[camera, frame, channel, height, width]` pixel-space input."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise ValueError("image_frames must have shape [camera, frame, channel, height, width]")
    if frames.shape[2] != 3:
        raise ValueError("image_frames channel dimension must be RGB (3)")
    result = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()
    if not torch.isfinite(result).all():
        raise ValueError("image_frames contain NaN or infinity")
    if result.min().item() < 0.0 or result.max().item() > 1.0:
        raise ValueError("image_frames must be in pixel range [0, 1]")
    return result


class AlpamayoWrapper(ABC):
    """Unified interface; attacks are inserted before official image preprocessing."""

    @abstractmethod
    def preprocess(self, data: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def forward_clean(self, data: dict[str, Any]) -> InferenceOutput: ...

    def forward_adv(self, data: dict[str, Any], adversarial_frames: torch.Tensor) -> InferenceOutput:
        attacked = dict(data)
        attacked["image_frames"] = validate_image_frames(adversarial_frames)
        return self.forward_clean(attacked)

    @staticmethod
    def get_reasoning(output: InferenceOutput) -> str:
        return output.reasoning

    @staticmethod
    def get_trajectory(output: InferenceOutput) -> torch.Tensor:
        return output.trajectory

    @staticmethod
    def get_visual_features(output: InferenceOutput) -> torch.Tensor:
        return output.visual_features

    def differentiable_forward(self, frames: torch.Tensor) -> InferenceOutput:
        raise RuntimeError(
            "This backend does not expose a differentiable pixel-to-output path. "
            "Use random noise, or configure an explicitly labeled differentiable surrogate."
        )


class MockAlpamayoWrapper(AlpamayoWrapper):
    """Deterministic differentiable toy VLA used only for pipeline verification."""

    def __init__(self, horizon: int = 64, device: str | torch.device = "cpu") -> None:
        self.horizon = horizon
        self.device = torch.device(device)

    def preprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        result["image_frames"] = validate_image_frames(data["image_frames"]).to(self.device)
        return result

    def _features(self, frames: torch.Tensor) -> torch.Tensor:
        means = frames.mean(dim=(-1, -2))
        stds = frames.std(dim=(-1, -2), unbiased=False)
        dx = (frames[..., 1:] - frames[..., :-1]).abs().mean(dim=(-1, -2, -3), keepdim=False)
        dy = (frames[..., 1:, :] - frames[..., :-1, :]).abs().mean(
            dim=(-1, -2, -3), keepdim=False
        )
        return torch.cat([means, stds, dx[..., None], dy[..., None]], dim=-1)

    def _trajectory(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(0, 1))
        steps = torch.arange(1, self.horizon + 1, device=features.device, dtype=features.dtype)
        speed = 0.08 + 0.06 * pooled[0]
        lateral = 0.8 * (pooled[2] - pooled[0])
        x = steps * speed
        y = lateral * torch.sin(steps / 12.0)
        z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _reason(features: torch.Tensor) -> str:
        pooled = features.detach().mean(dim=(0, 1))
        brightness = "bright" if pooled[0:3].mean().item() >= 0.5 else "dim"
        direction = "right" if (pooled[2] - pooled[0]).item() >= 0 else "left"
        texture = "complex" if pooled[6:8].mean().item() >= 0.12 else "clear"
        return f"Mock CoC: {brightness} {texture} scene; maintain caution and track {direction}."

    def differentiable_forward(self, frames: torch.Tensor) -> InferenceOutput:
        frames = validate_image_frames(frames).to(self.device)
        features = self._features(frames)
        return InferenceOutput(
            reasoning=self._reason(features),
            trajectory=self._trajectory(features),
            visual_features=features,
            extra={"backend": "mock", "verified_official_result": False},
        )

    def forward_clean(self, data: dict[str, Any]) -> InferenceOutput:
        prepared = self.preprocess(data)
        return self.differentiable_forward(prepared["image_frames"])


class OfficialAlpamayoWrapper(AlpamayoWrapper):
    """Thin adapter over the pinned NVIDIA release; does not reimplement Alpamayo."""

    def __init__(
        self,
        model_id: str = "nvidia/Alpamayo2-Super",
        device_map: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        diffusion_steps: int = 10,
        seed: int = 42,
        offload_folder: str | None = None,
        max_memory: dict[Any, str] | None = None,
    ) -> None:
        try:
            from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
        except ImportError as exc:
            raise RuntimeError(
                "Official backend unavailable. Add third_party/alpamayo2/src to PYTHONPATH "
                "and install its locked environment."
            ) from exc
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        load_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map}
        if offload_folder is not None:
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = offload_folder
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        self.model = Alpamayo2Super.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.device = next(self.model.get_input_embeddings().parameters()).device
        if self.device.type == "meta":
            raise RuntimeError("Input embeddings remained on the meta device after checkpoint load")

    def preprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        result["image_frames"] = validate_image_frames(data["image_frames"])
        return result

    def _vision_module(self) -> torch.nn.Module:
        root = self.model.vlm.model
        for name in ("visual", "vision_model", "vision_tower"):
            module = getattr(root, name, None)
            if isinstance(module, torch.nn.Module):
                return module
        raise RuntimeError("Could not locate the official VLM vision module for feature capture")

    def forward_clean(self, data: dict[str, Any]) -> InferenceOutput:
        from alpamayo2_super import helper

        prepared = self.preprocess(data)
        model_inputs = helper.prepare_model_inputs(prepared, self.model.config, self.model.tokenizer)
        model_inputs = helper.to_device(model_inputs, self.device)
        torch.cuda.manual_seed_all(self.seed)
        with FeatureCapture(self._vision_module(), detach=True) as capture:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _pred_rot, _logprob, extra = self.model.sample_trajectories_from_data(
                    data=model_inputs,
                    num_traj_samples=1,
                    diffusion_kwargs={"inference_step": self.diffusion_steps},
                    return_extra=True,
                )
        if capture.value is None:
            raise RuntimeError("Official vision hook did not observe a tensor output")
        cot = str(extra["cot"].reshape(-1)[0])
        trajectory = pred_xyz[0, 0, 0]
        return InferenceOutput(
            reasoning=cot,
            trajectory=trajectory,
            visual_features=capture.value,
            extra={"backend": "official", "verified_official_result": True},
        )


def official_source_path(project_root: Path) -> Path:
    return project_root / "third_party" / "alpamayo2" / "src"
