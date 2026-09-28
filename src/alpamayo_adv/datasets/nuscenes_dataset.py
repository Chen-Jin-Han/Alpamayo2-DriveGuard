from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


DEFAULT_CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class NuScenesSequenceConfig:
    dataroot: str
    version: str = "v1.0-mini"
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    frames: int = 4
    max_samples: int | None = 100
    scene_keywords: tuple[str, ...] = ()
    sampling_strategy: str = "sequential"
    resize_height: int | None = None
    resize_width: int | None = None


def scene_matches(description: str, keywords: Iterable[str]) -> bool:
    requested = tuple(keyword.strip().lower() for keyword in keywords if keyword.strip())
    if not requested:
        return True
    normalized = description.lower()
    return any(keyword in normalized for keyword in requested)


def select_samples(
    samples: list[dict[str, Any]], max_samples: int | None, strategy: str
) -> list[dict[str, Any]]:
    """Select deterministically, optionally round-robin across scene tokens."""
    if strategy not in {"sequential", "round_robin"}:
        raise ValueError("sampling_strategy must be sequential or round_robin")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    if strategy == "sequential":
        return samples if max_samples is None else samples[:max_samples]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["scene_token"], []).append(sample)
    selected: list[dict[str, Any]] = []
    depth = 0
    while True:
        added = False
        for group in grouped.values():
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if max_samples is not None and len(selected) >= max_samples:
                    return selected
        if not added:
            return selected
        depth += 1


def _camera_history(nusc: Any, token: str, frames: int) -> list[dict[str, Any]]:
    records = []
    current = token
    while current and len(records) < frames:
        record = nusc.get("sample_data", current)
        records.append(record)
        current = record["prev"]
    if not records:
        raise ValueError("camera sample has no sample_data records")
    records.reverse()
    while len(records) < frames:
        records.insert(0, records[0])
    return records


def _load_rgb(path: Path, height: int | None, width: int | None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if height is not None and width is not None:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


class NuScenesSequenceDataset(torch.utils.data.Dataset):
    """nuScenes sample adapter yielding `[camera, time, RGB, H, W]` uint8 frames."""

    def __init__(self, config: NuScenesSequenceConfig) -> None:
        if config.frames < 1:
            raise ValueError("frames must be positive")
        if (config.resize_height is None) != (config.resize_width is None):
            raise ValueError("resize_height and resize_width must be set together")
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError("Install nuscenes-devkit to use NuScenesSequenceDataset") from exc
        self.config = config
        self.nusc = NuScenes(version=config.version, dataroot=config.dataroot, verbose=False)
        eligible = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if scene_matches(scene.get("description", ""), config.scene_keywords):
                eligible.append(sample)
        self.samples = select_samples(eligible, config.max_samples, config.sampling_strategy)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        camera_tensors = []
        timestamps = []
        for camera in self.config.cameras:
            if camera not in sample["data"]:
                raise KeyError(f"sample {sample['token']} does not contain {camera}")
            records = _camera_history(self.nusc, sample["data"][camera], self.config.frames)
            frames = [
                _load_rgb(
                    Path(self.config.dataroot) / record["filename"],
                    self.config.resize_height,
                    self.config.resize_width,
                )
                for record in records
            ]
            camera_tensors.append(torch.stack(frames))
            timestamps.append([int(record["timestamp"]) for record in records])
        scene = self.nusc.get("scene", sample["scene_token"])
        return {
            "image_frames": torch.stack(camera_tensors),
            "camera_names": list(self.config.cameras),
            "camera_indices": torch.arange(len(self.config.cameras), dtype=torch.long),
            "absolute_timestamps": torch.tensor(timestamps, dtype=torch.long),
            "sample_token": sample["token"],
            "scene_token": sample["scene_token"],
            "scene_name": scene.get("name", ""),
            "scene_description": scene.get("description", ""),
            "dataset_source": "nuScenes",
        }
