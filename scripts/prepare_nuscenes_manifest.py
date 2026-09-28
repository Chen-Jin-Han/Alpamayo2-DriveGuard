from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from alpamayo_adv.datasets.nuscenes_dataset import (
    DEFAULT_CAMERAS,
    NuScenesSequenceConfig,
    NuScenesSequenceDataset,
    _camera_history,
)


SCENARIO_KEYWORDS = {
    "night": ("night",),
    "rain": ("rain", "wet"),
    "intersection": ("intersection", "crossing", "junction"),
    "highway": ("highway", "expressway"),
    "vehicle_following": ("following",),
    "pedestrian": ("pedestrian", "peds", "ped ", "jaywalker", "crosswalk"),
    "lane_change": ("overtaking", "lane change", "turn left", "turn right"),
    "occlusion_proxy": ("parked truck", "parked cars", "difficult lighting"),
    "dense_complex_proxy": ("busy", "many peds", "parking lot", "cars, truck"),
    "construction": ("construction",),
    "dense_traffic": ("dense traffic", "heavy traffic"),
}


def scenario_tags(description: str) -> list[str]:
    normalized = description.lower()
    tags = [
        name
        for name, keywords in SCENARIO_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return tags or ["unspecified"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inventory a deterministic nuScenes sequence subset."
    )
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--sampling-strategy", choices=("sequential", "round_robin"), default="round_robin"
    )
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--resize-width", type=int, default=384)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/nuscenes_100_manifest.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    config = NuScenesSequenceConfig(
        dataroot=str(args.dataroot),
        version=args.version,
        cameras=DEFAULT_CAMERAS,
        frames=args.frames,
        max_samples=args.samples,
        sampling_strategy=args.sampling_strategy,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
    )
    dataset = NuScenesSequenceDataset(config)
    if len(dataset) != args.samples:
        raise RuntimeError(
            f"requested {args.samples} samples but selected {len(dataset)} from {args.version}"
        )

    missing_files: list[str] = []
    entries: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    padded_histories = 0

    for index, sample in enumerate(dataset.samples):
        scene = dataset.nusc.get("scene", sample["scene_token"])
        description = scene.get("description", "")
        tags = scenario_tags(description)
        scene_counts[scene.get("name", "")] += 1
        tag_counts.update(tags)
        camera_records: dict[str, list[dict[str, Any]]] = {}
        for camera in config.cameras:
            records = _camera_history(dataset.nusc, sample["data"][camera], config.frames)
            filenames = [record["filename"] for record in records]
            if len(set(filenames)) < len(filenames):
                padded_histories += 1
            for filename in filenames:
                if not (args.dataroot / filename).is_file():
                    missing_files.append(filename)
            camera_records[camera] = [
                {
                    "filename": record["filename"],
                    "timestamp": int(record["timestamp"]),
                    "is_key_frame": bool(record.get("is_key_frame", False)),
                }
                for record in records
            ]
        entries.append(
            {
                "index": index,
                "sample_token": sample["token"],
                "timestamp": int(sample["timestamp"]),
                "scene_token": sample["scene_token"],
                "scene_name": scene.get("name", ""),
                "scene_description": description,
                "scenario_tags": tags,
                "cameras": camera_records,
            }
        )

    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} referenced camera files are missing; first: {preview}"
        )

    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    probes = []
    expected_shape = [
        len(config.cameras),
        config.frames,
        3,
        args.resize_height,
        args.resize_width,
    ]
    for index in probe_indices:
        item = dataset[index]
        actual_shape = list(item["image_frames"].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"sample {index} shape {actual_shape} does not match {expected_shape}"
            )
        probes.append(
            {
                "index": index,
                "sample_token": item["sample_token"],
                "image_shape": actual_shape,
                "dtype": str(item["image_frames"].dtype),
                "pixel_min": int(item["image_frames"].min()),
                "pixel_max": int(item["image_frames"].max()),
                "timestamps_shape": list(item["absolute_timestamps"].shape),
            }
        )

    payload = {
        "schema_version": 1,
        "dataset": "nuScenes",
        "version": args.version,
        "dataroot": str(args.dataroot.resolve()),
        "selection": args.sampling_strategy,
        "sample_count": len(dataset),
        "scene_count": len(scene_counts),
        "camera_order": list(config.cameras),
        "frames_per_camera": config.frames,
        "resize_hw": [args.resize_height, args.resize_width],
        "referenced_camera_file_count": len(dataset) * len(config.cameras) * config.frames,
        "missing_camera_file_count": 0,
        "padded_camera_histories": padded_histories,
        "scene_sample_counts": dict(sorted(scene_counts.items())),
        "scenario_tag_counts": dict(sorted(tag_counts.items())),
        "decoded_probes": probes,
        "samples": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "sample_count": len(dataset),
                "scene_count": len(scene_counts),
                "probe_shapes": [probe["image_shape"] for probe in probes],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
