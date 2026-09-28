from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.run_nuscenes_prototype import summarize
except ModuleNotFoundError:
    # Direct execution sets sys.path[0] to this scripts directory.
    from run_nuscenes_prototype import summarize


MAIN_RUNS = (
    "nuscenes_prototype_balanced_100",
    "nuscenes_prototype_balanced_seed43",
    "nuscenes_prototype_balanced_seed44",
)


def pearson(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = [(float(row[left]), float(row[right])) for row in rows if left in row and right in row]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - xm) * (y - ym) for x, y in pairs)
    scale_x = sum((x - xm) ** 2 for x in xs) ** 0.5
    scale_y = sum((y - ym) ** 2 for y in ys) ** 0.5
    return numerator / (scale_x * scale_y) if scale_x and scale_y else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_records(outputs: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(outputs.glob("nuscenes_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("sample_count") != 100:
            continue
        cameras = payload.get("attacked_cameras")
        frames = payload.get("attacked_frames")
        weights = payload.get("objective_weights", {})
        for item in payload["summary"]:
            records.append(
                {
                    "run": path.parent.name,
                    "attack": item["attack"],
                    "epsilon_pixels": item["epsilon_pixels"],
                    "seed": payload.get("seed", 42),
                    "sample_count": item["sample_count"],
                    "scene_count": item["scene_count"],
                    "temporal_frames": payload["frames_per_camera"],
                    "attacked_camera_count": 6 if cameras is None else len(cameras),
                    "attacked_frame_count": payload["frames_per_camera"] if frames is None else len(frames),
                    "shared_across_cameras": payload.get("shared_across_cameras", False),
                    "objective_feature": weights.get("feature", 1.0),
                    "objective_trajectory": weights.get("trajectory", 1.0),
                    "objective_temporal": weights.get("temporal", 0.25),
                    **{key: value for key, value in item.items() if key not in {"attack", "epsilon_pixels"}},
                }
            )
    return records


def line_plot(
    rows: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str, path: Path
) -> None:
    points = sorted((float(row[x_key]), float(row[y_key])) for row in rows)
    if not points:
        return
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed nuScenes mock prototypes")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--destination", type=Path, default=Path("outputs/nuscenes_study"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    records = summary_records(args.outputs)
    if not records:
        raise SystemExit("no completed 100-sample nuScenes prototype summaries found")
    write_csv(args.destination / "all_run_summaries.csv", records)
    (args.destination / "all_run_summaries.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    main_rows = []
    for run in MAIN_RUNS:
        rows = read_jsonl(args.outputs / run / "sample_metrics.jsonl")
        for row in rows:
            row["run"] = run
            row.setdefault("seed", 42 if run.endswith("100") else int(run[-2:]))
        main_rows.extend(rows)
    expected = 3 * 100 * 4
    if len(main_rows) != expected:
        raise RuntimeError(f"main three-seed matrix expected {expected} rows, found {len(main_rows)}")
    main_summary = summarize(main_rows)
    write_csv(args.destination / "three_seed_attack_summary.csv", main_summary)
    (args.destination / "three_seed_attack_summary.json").write_text(
        json.dumps(main_summary, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = args.outputs / "nuscenes_balanced_100_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_tags: dict[str, set[str]] = defaultdict(set)
    for sample in manifest["samples"]:
        scene_tags[sample["scene_name"]].update(sample["scenario_tags"])
    seed42_rows = [row for row in main_rows if int(row.get("seed", 42)) == 42]
    scenario_records = []
    for scene in sorted(scene_tags):
        for item in summarize([row for row in seed42_rows if row["scene_name"] == scene]):
            scenario_records.append(
                {"scene_name": scene, "scenario_tags": ";".join(sorted(scene_tags[scene])), **item}
            )
    write_csv(args.destination / "scenario_metrics.csv", scenario_records)

    correlation_rows = []
    for row in main_rows:
        enriched = dict(row)
        enriched["reasoning_distance"] = 1.0 - float(row["reasoning_similarity"])
        correlation_rows.append(enriched)
    correlations = {
        "n": len(correlation_rows),
        "feature_to_reasoning": pearson(
            correlation_rows, "feature_cosine_distance", "reasoning_distance"
        ),
        "reasoning_to_trajectory": pearson(
            correlation_rows, "reasoning_distance", "trajectory_deviation"
        ),
        "feature_to_trajectory": pearson(
            correlation_rows, "feature_cosine_distance", "trajectory_deviation"
        ),
    }
    (args.destination / "correlations.json").write_text(
        json.dumps(correlations, indent=2) + "\n", encoding="utf-8"
    )

    main_seed_records = [row for row in records if row["run"] in MAIN_RUNS]
    attacks = sorted({row["attack"] for row in main_seed_records})
    attack_means = []
    attack_stds = []
    for attack in attacks:
        values = [float(row["asr_sim_mean"]) for row in main_seed_records if row["attack"] == attack]
        attack_means.append(statistics.fmean(values))
        attack_stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(attacks, attack_means, yerr=attack_stds, capsize=4)
    axis.set(ylabel="Mock offline ASR", ylim=(0, 1.05))
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.destination / "attack_asr_three_seeds.png", dpi=160); plt.close(fig)

    line_plot(
        [row for row in records if row["run"] == "nuscenes_temporal_epsilon"],
        "epsilon_pixels", "asr_sim_mean", "Epsilon (pixel levels / 255)", "Mock offline ASR",
        args.destination / "asr_vs_epsilon.png",
    )
    camera_rows = [row for row in records if row["run"] in {
        "nuscenes_temporal_cam1", "nuscenes_temporal_cam3", "nuscenes_prototype_balanced_100"
    } and row["attack"] == "temporal"]
    line_plot(
        camera_rows, "attacked_camera_count", "asr_sim_mean", "Attacked cameras",
        "Mock offline ASR", args.destination / "asr_vs_cameras.png",
    )
    frame_rows = [row for row in records if row["run"] in {
        "nuscenes_temporal_t1", "nuscenes_temporal_t2", "nuscenes_prototype_balanced_100"
    } and row["attack"] == "temporal"]
    line_plot(
        frame_rows, "temporal_frames", "asr_sim_mean", "Temporal frames",
        "Mock offline ASR", args.destination / "asr_vs_frames.png",
    )

    points = [
        (float(row["feature_cosine_distance"]), float(row["trajectory_deviation"]))
        for row in main_rows
    ]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.scatter([p[0] for p in points], [p[1] for p in points], s=8, alpha=0.35)
    axis.set(xlabel="Feature cosine distance", ylabel="Trajectory deviation")
    axis.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.destination / "feature_vs_trajectory.png", dpi=160); plt.close(fig)

    report = [
        "# nuScenes 100-Sample Prototype Study",
        "",
        "These results validate attack logic on the deterministic mock wrapper; they are not Alpamayo robustness evidence.",
        "The balanced subset contains 100 samples (10 from each nuScenes mini scene), six cameras, and temporal frames.",
        "",
        "## Three-seed main comparison",
        "",
        "| Attack | Epsilon | N | ASR mean | Feature distance | Reasoning similarity | Trajectory deviation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_summary:
        report.append(
            f"| {row['attack']} | {row['epsilon_pixels']}/255 | {row['sample_count']} | "
            f"{row['asr_sim_mean']:.4f} | {row['feature_cosine_distance_mean']:.6f} | "
            f"{row['reasoning_similarity_mean']:.4f} | {row['trajectory_deviation_mean']:.6f} |"
        )

    report.extend(
        [
            "",
            "## Epsilon ablation (seed 42, temporal)",
            "",
            "| Epsilon | ASR | Feature distance | Reasoning similarity | Trajectory deviation |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        (row for row in records if row["run"] == "nuscenes_temporal_epsilon"),
        key=lambda row: row["epsilon_pixels"],
    ):
        report.append(
            f"| {row['epsilon_pixels']}/255 | {row['asr_sim_mean']:.4f} | "
            f"{row['feature_cosine_distance_mean']:.6f} | {row['reasoning_similarity_mean']:.4f} | "
            f"{row['trajectory_deviation_mean']:.6f} |"
        )

    report.extend(
        [
            "",
            "## Camera and temporal ablations (seed 42, epsilon 16/255)",
            "",
            "| Ablation | Level | ASR | Feature distance | Trajectory deviation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    ablation_specs = [
        ("cameras", "nuscenes_temporal_cam1", "attacked_camera_count"),
        ("cameras", "nuscenes_temporal_cam3", "attacked_camera_count"),
        ("cameras", "nuscenes_prototype_balanced_100", "attacked_camera_count"),
        ("input frames", "nuscenes_temporal_t1", "temporal_frames"),
        ("input frames", "nuscenes_temporal_t2", "temporal_frames"),
        ("input frames", "nuscenes_prototype_balanced_100", "temporal_frames"),
        ("attacked frames", "nuscenes_temporal_last1", "attacked_frame_count"),
        ("attacked frames", "nuscenes_temporal_last2", "attacked_frame_count"),
        ("attacked frames", "nuscenes_prototype_balanced_100", "attacked_frame_count"),
    ]
    for label, run, level_key in ablation_specs:
        row = next(
            item for item in records if item["run"] == run and item["attack"] == "temporal"
        )
        report.append(
            f"| {label} | {row[level_key]} | {row['asr_sim_mean']:.4f} | "
            f"{row['feature_cosine_distance_mean']:.6f} | {row['trajectory_deviation_mean']:.6f} |"
        )

    report.extend(
        [
            "",
            "## Objective ablation (seed 42, epsilon 16/255)",
            "",
            "| Run | feature | trajectory | temporal | ASR | Trajectory deviation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        (row for row in records if row["run"].startswith("nuscenes_objective_")),
        key=lambda row: row["run"],
    ):
        report.append(
            f"| {row['run'].removeprefix('nuscenes_objective_')} | "
            f"{row['objective_feature']} | {row['objective_trajectory']} | "
            f"{row['objective_temporal']} | {row['asr_sim_mean']:.4f} | "
            f"{row['trajectory_deviation_mean']:.6f} |"
        )
    report.extend(
        [
            "",
            "## Correlations",
            "",
            f"- Feature distance to reasoning distance: `{correlations['feature_to_reasoning']}`",
            f"- Reasoning distance to trajectory deviation: `{correlations['reasoning_to_trajectory']}`",
            f"- Feature distance to trajectory deviation: `{correlations['feature_to_trajectory']}`",
            "",
            "All sample rows are retained; no success-only filtering or outcome-dependent threshold tuning is used.",
            "",
        ]
    )
    (args.destination / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_run_summaries": len(records),
                "three_seed_rows": len(main_rows),
                "scenario_rows": len(scenario_records),
                "destination": str(args.destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
