#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT = Path.cwd().resolve()
THRESHOLDS = {
    "feature_distance": 0.05,
    "reasoning_similarity": 0.8,
    "trajectory_deviation": 0.5,
    "ade_increase": 0.5,
    "fde_increase": 1.0,
}


def base(sample_index: int) -> dict:
    return {
        "backend": "official",
        "device": "cuda",
        "device_map": "auto",
        "offload_folder": str(PROJECT / ".offload/alpamayo2"),
        "max_memory": {0: "18GiB", "cpu": "400GiB"},
        "seed": 42,
        "model_id": "nvidia/Alpamayo2-Super",
        "diffusion_steps": 10,
        "data": {
            "source": "cached",
            "path": str(PROJECT / f"data/physicalai/validation_sample_{sample_index}.pt"),
            "sample_index": sample_index,
        },
        "success_thresholds": THRESHOLDS,
        "defense": {"name": "none"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize official multi-sample YAML configs")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output-dir", default="configs/generated")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        clean = base(index)
        clean["attack"] = {"name": "clean", "epsilon": 0}
        clean["output_dir"] = f"outputs/official_physicalai_clean_sample{index}"
        if index == 0:
            clean["output_dir"] = "outputs/official_physicalai_clean"
        clean_path = output_dir / f"official_clean_sample{index}.yaml"
        clean_path.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")

        temporal = base(index)
        temporal["attack"] = {
            "name": "precomputed",
            "label": "temporal_resnet18_transfer",
            "path": str(
                PROJECT / f"data/physicalai/attacks/temporal_eps16_sample{index}_cam6.pt"
            ),
            "epsilon": "16/255",
            "attacked_cameras": [0, 1, 2, 3, 4, 5],
        }
        temporal["output_dir"] = f"outputs/official_physicalai_temporal_16_sample{index}"
        if index == 0:
            temporal["attack"]["path"] = str(
                PROJECT / "data/physicalai/attacks/temporal_eps16_cam6.pt"
            )
            temporal["output_dir"] = "outputs/official_physicalai_temporal_16"
        temporal_path = output_dir / f"official_temporal16_sample{index}.yaml"
        temporal_path.write_text(yaml.safe_dump(temporal, sort_keys=False), encoding="utf-8")
    print(f"generated={args.count * 2} output_dir={output_dir}")


if __name__ == "__main__":
    main()
