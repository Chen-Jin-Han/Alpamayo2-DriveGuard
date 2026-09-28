from __future__ import annotations

import argparse
import csv
import json
import statistics
import math
from pathlib import Path
from typing import Any

import torch

from alpamayo_adv.attacks import fgsm, pgd, random_noise, temporal_pgd
from alpamayo_adv.datasets.nuscenes_dataset import (
    DEFAULT_CAMERAS,
    NuScenesSequenceConfig,
    NuScenesSequenceDataset,
)
from alpamayo_adv.experiment import _fgsm_objective, _objective
from alpamayo_adv.metrics import attack_success_metrics, evaluate_pair
from alpamayo_adv.models import MockAlpamayoWrapper


SUPPORTED_ATTACKS = ("random", "fgsm", "pgd", "temporal")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["attack"]), int(row["epsilon_pixels"])), []).append(row)
    summaries = []
    excluded = {
        "sample_index", "sample_token", "scene_name", "attack", "epsilon_pixels", "seed"
    }
    for (attack, epsilon), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "attack": attack,
            "epsilon_pixels": epsilon,
            "sample_count": len(group),
            "scene_count": len({row["scene_name"] for row in group}),
        }
        numeric_keys = sorted(
            key
            for key, value in group[0].items()
            if key not in excluded and isinstance(value, (int, float))
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in group]
            record[f"{key}_mean"] = statistics.fmean(values)
            record[f"{key}_median"] = statistics.median(values)
            record[f"{key}_std"] = statistics.pstdev(values)
            margin = 1.96 * record[f"{key}_std"] / math.sqrt(len(values))
            record[f"{key}_ci95_low"] = record[f"{key}_mean"] - margin
            record[f"{key}_ci95_high"] = record[f"{key}_mean"] + margin
        summaries.append(record)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable CPU-only mock attack prototype on nuScenes sequences."
    )
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--sampling-strategy", choices=("sequential", "round_robin"), default="round_robin"
    )
    parser.add_argument("--resize-height", type=int, default=112)
    parser.add_argument("--resize-width", type=int, default=192)
    parser.add_argument("--attacks", default="random,fgsm,pgd,temporal")
    parser.add_argument("--epsilon-values", default="16")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=4.0)
    parser.add_argument("--attacked-cameras", default="")
    parser.add_argument("--attacked-frames", default="")
    parser.add_argument("--shared-across-cameras", action="store_true")
    parser.add_argument("--objective-feature", type=float, default=1.0)
    parser.add_argument("--objective-trajectory", type=float, default=1.0)
    parser.add_argument("--objective-temporal", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nuscenes_prototype"))
    return parser.parse_args()


def read_completed(path: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["sample_index"]), str(row["attack"]), int(row["epsilon_pixels"]))
        completed[key] = row
    return completed


def main() -> None:
    args = parse_args()
    attacks = parse_csv(args.attacks)
    unsupported = sorted(set(attacks) - set(SUPPORTED_ATTACKS))
    if unsupported:
        raise ValueError(f"unsupported attacks: {unsupported}")
    epsilon_values = [int(value) for value in parse_csv(args.epsilon_values)]
    attacked_cameras = (
        [int(value) for value in parse_csv(args.attacked_cameras)]
        if args.attacked_cameras
        else None
    )
    attacked_frames = (
        [int(value) for value in parse_csv(args.attacked_frames)]
        if args.attacked_frames
        else None
    )
    if any(value < 0 or value > 16 for value in epsilon_values):
        raise ValueError("epsilon values must be integer pixels in [0, 16]")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    torch.set_num_threads(max(1, args.threads))

    dataset = NuScenesSequenceDataset(
        NuScenesSequenceConfig(
            dataroot=str(args.dataroot),
            version=args.version,
            cameras=DEFAULT_CAMERAS,
            frames=args.frames,
            max_samples=args.samples,
            sampling_strategy=args.sampling_strategy,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
        )
    )
    if len(dataset) != args.samples:
        raise RuntimeError(f"requested {args.samples} samples, selected {len(dataset)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "sample_metrics.jsonl"
    completed = read_completed(rows_path)
    wrapper = MockAlpamayoWrapper(device="cpu")
    thresholds = {
        "feature_distance": 0.05,
        "reasoning_similarity": 0.8,
        "reasoning_entity_f1": 0.8,
        "reasoning_action_f1": 1.0,
        "trajectory_deviation": 0.5,
    }

    for sample_index in range(len(dataset)):
        item = dataset[sample_index]
        data = wrapper.preprocess(item)
        clean_frames = data["image_frames"]
        clean = wrapper.forward_clean(data)
        for epsilon_pixels in epsilon_values:
            epsilon = epsilon_pixels / 255.0
            for attack in attacks:
                key = (sample_index, attack, epsilon_pixels)
                if key in completed:
                    continue
                seed = args.seed + sample_index
                if attack == "random":
                    adv_frames = random_noise(
                        clean_frames, epsilon, seed, attacked_cameras, attacked_frames
                    )
                elif attack == "fgsm":
                    adv_frames = fgsm(
                        clean_frames,
                        _fgsm_objective(wrapper, clean),
                        epsilon,
                        attacked_cameras,
                        attacked_frames,
                    )
                elif attack == "pgd":
                    adv_frames = pgd(
                        clean_frames,
                        _objective(
                            wrapper,
                            clean,
                            {
                                "feature": args.objective_feature,
                                "trajectory": args.objective_trajectory,
                                "temporal": args.objective_temporal,
                            },
                        ),
                        epsilon,
                        args.step_size / 255.0,
                        args.iterations,
                        True,
                        seed,
                        attacked_cameras,
                        attacked_frames,
                    )
                else:
                    adv_frames = temporal_pgd(
                        clean_frames,
                        _objective(
                            wrapper,
                            clean,
                            {
                                "feature": args.objective_feature,
                                "trajectory": args.objective_trajectory,
                                "temporal": args.objective_temporal,
                            },
                        ),
                        epsilon,
                        args.step_size / 255.0,
                        args.iterations,
                        args.shared_across_cameras,
                        seed,
                        attacked_cameras,
                        attacked_frames,
                    )
                adv = wrapper.forward_adv(data, adv_frames)
                metrics = evaluate_pair(clean_frames, adv_frames, clean, adv)
                metrics.update(attack_success_metrics(metrics, thresholds))
                if metrics["linf"] > epsilon + 1e-6:
                    raise AssertionError(f"L-infinity budget violated for {key}")
                row = {
                    "sample_index": sample_index,
                    "sample_token": item["sample_token"],
                    "scene_name": item["scene_name"],
                    "attack": attack,
                    "epsilon_pixels": epsilon_pixels,
                    "seed": args.seed,
                    **metrics,
                }
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                completed[key] = row
        print(f"completed_sample={sample_index + 1}/{len(dataset)}", flush=True)

    rows = list(completed.values())
    expected = len(dataset) * len(attacks) * len(epsilon_values)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} result rows, found {len(rows)}")
    summaries = summarize(rows)
    payload = {
        "status": "completed",
        "result_scope": "mock_pipeline_nuscenes_prototype_not_alpamayo_evidence",
        "dataset": args.version,
        "sample_count": len(dataset),
        "camera_order": list(DEFAULT_CAMERAS),
        "frames_per_camera": args.frames,
        "sampling_strategy": args.sampling_strategy,
        "resize_hw": [args.resize_height, args.resize_width],
        "attacks": attacks,
        "epsilon_pixels": epsilon_values,
        "iterations": args.iterations,
        "step_size_pixels": args.step_size,
        "attacked_cameras": attacked_cameras,
        "attacked_frames": attacked_frames,
        "shared_across_cameras": args.shared_across_cameras,
        "objective_weights": {
            "feature": args.objective_feature,
            "trajectory": args.objective_trajectory,
            "temporal": args.objective_temporal,
        },
        "seed": args.seed,
        "result_row_count": len(rows),
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if summaries:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
