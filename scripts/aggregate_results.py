#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _attack_label(payload: dict[str, Any]) -> str:
    attack = payload["config"]["attack"]
    return str(attack.get("label", attack["name"]))


def _summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": float(len(values)),
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "ci95_low": mean - 1.96 * std / math.sqrt(len(values)),
        "ci95_high": mean + 1.96 * std / math.sqrt(len(values)),
    }


def _epsilon(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pearson(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = [(float(row[left]), float(row[right])) for row in rows if left in row and right in row]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_scale = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_scale = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def _save_plots(rows: list[dict[str, Any]], destination: Path) -> list[str]:
    official = [row for row in rows if row.get("scope") == "official_model_physicalai"]
    written: list[str] = []

    epsilon_rows = [row for row in official if _epsilon(row.get("epsilon")) is not None]
    if epsilon_rows:
        epsilon_rows.sort(key=lambda row: _epsilon(row["epsilon"]) or 0)
        x = [(_epsilon(row["epsilon"]) or 0) * 255 for row in epsilon_rows]
        for filename, metric, ylabel in [
            ("asr_vs_epsilon.png", "asr_sim", "ASR (offline)"),
            ("trajectory_deviation_vs_epsilon.png", "trajectory_deviation", "Trajectory deviation (m)"),
        ]:
            points = [(xx, float(row[metric])) for xx, row in zip(x, epsilon_rows) if metric in row]
            if points:
                fig, axis = plt.subplots(figsize=(6, 4))
                axis.scatter([p[0] for p in points], [p[1] for p in points])
                axis.set(xlabel="Epsilon (pixel levels / 255)", ylabel=ylabel)
                axis.grid(alpha=0.3)
                fig.tight_layout(); fig.savefig(destination / filename, dpi=160); plt.close(fig)
                written.append(filename)
        ade_points = [
            (xx, float(row["adv_ade"]), float(row["adv_fde"]))
            for xx, row in zip(x, epsilon_rows)
            if "adv_ade" in row and "adv_fde" in row
        ]
        if ade_points:
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.scatter([p[0] for p in ade_points], [p[1] for p in ade_points], label="ADE")
            axis.scatter([p[0] for p in ade_points], [p[2] for p in ade_points], label="FDE")
            axis.set(xlabel="Epsilon (pixel levels / 255)", ylabel="Error (m)")
            axis.legend(); axis.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(destination / "ade_fde_vs_epsilon.png", dpi=160); plt.close(fig)
            written.append("ade_fde_vs_epsilon.png")

    camera_rows = [row for row in official if "attacked_cameras" in row and "asr_sim" in row]
    if camera_rows:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(
            [float(row["attacked_cameras"]) for row in camera_rows],
            [float(row["asr_sim"]) for row in camera_rows],
        )
        axis.set(xlabel="Attacked cameras", ylabel="ASR (offline)")
        axis.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(destination / "asr_vs_cameras.png", dpi=160); plt.close(fig)
        written.append("asr_vs_cameras.png")

    for filename, x_metric, x_label in [
        ("feature_vs_trajectory.png", "feature_cosine_distance", "Feature cosine distance"),
        ("reasoning_vs_trajectory.png", "reasoning_similarity", "Reasoning similarity"),
    ]:
        points = [
            (float(row[x_metric]), float(row["trajectory_deviation"]))
            for row in official
            if x_metric in row and "trajectory_deviation" in row
        ]
        if points:
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.scatter([p[0] for p in points], [p[1] for p in points])
            axis.set(xlabel=x_label, ylabel="Trajectory deviation (m)")
            axis.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(destination / filename, dpi=160); plt.close(fig)
            written.append(filename)
    return written


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed experiment result.json files")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="outputs/tables")
    args = parser.parse_args()
    outputs = Path(args.outputs)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(outputs.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        row = {
            "run": path.parent.name,
            "scope": payload.get("result_scope"),
            "attack": _attack_label(payload),
            "epsilon": payload["config"]["attack"].get("epsilon"),
            "seed": payload["config"].get("seed"),
            "sample": Path(payload["config"].get("data", {}).get("path", "unknown")).stem,
            **payload["metrics"],
        }
        row.setdefault(
            "unsafe_decision_proxy",
            float(
                float(row.get("success_reason", 0)) >= 1
                and float(row.get("success_trajectory", 0)) >= 1
            ),
        )
        rows.append(row)
    if not rows:
        raise SystemExit("no completed result.json files found")

    fields = sorted({key for row in rows for key in row})
    with (destination / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "all_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    official = [row for row in rows if row.get("scope") == "official_model_physicalai"]
    table1_fields = [
        "sample", "attack", "epsilon", "adv_ade", "adv_fde", "trajectory_deviation",
        "reasoning_similarity", "feature_cosine_distance", "unsafe_decision_proxy", "asr_sim",
    ]
    _write_csv(destination / "table1_attack_metrics.csv", official, table1_fields)
    _write_csv(
        destination / "table2_camera_ablation.csv",
        [row for row in official if "attacked_cameras" in row],
        ["sample", "attack", "epsilon", "attacked_cameras", "trajectory_deviation", "asr_sim"],
    )
    _write_csv(
        destination / "table3_epsilon_ablation.csv",
        [row for row in official if _epsilon(row.get("epsilon")) is not None],
        ["sample", "attack", "epsilon", "feature_cosine_distance", "adv_ade", "adv_fde", "asr_sim"],
    )
    _write_csv(
        destination / "table4_defense.csv",
        [row for row in official if "defense_trajectory_deviation" in row or "jpeg" in str(row["attack"])],
        [
            "sample", "attack", "epsilon", "trajectory_deviation", "asr_sim",
            "defense_trajectory_deviation", "defense_reasoning_similarity",
        ],
    )
    table5 = [
        {
            "sample": row["sample"],
            "scenario_type": "unlabeled_official_validation",
            "attack": row["attack"],
            "trajectory_deviation": row.get("trajectory_deviation"),
            "asr_sim": row.get("asr_sim"),
        }
        for row in official
    ]
    _write_csv(
        destination / "table5_scenario_samples.csv",
        table5,
        ["sample", "scenario_type", "attack", "trajectory_deviation", "asr_sim"],
    )
    success_cases = [row for row in official if float(row.get("asr_sim", 0)) >= 1]
    failed_cases = [row for row in official if float(row.get("asr_sim", 0)) < 1]
    (destination / "success_cases.json").write_text(
        json.dumps(success_cases, indent=2), encoding="utf-8"
    )
    (destination / "failed_cases.json").write_text(
        json.dumps(failed_cases, indent=2), encoding="utf-8"
    )
    canonical_temporal = [
        row
        for row in official
        if row["run"] == "official_physicalai_temporal_16"
        or row["run"].startswith("official_physicalai_temporal_16_sample")
    ]
    unsafe_values = [float(row["unsafe_decision_proxy"]) for row in canonical_temporal]
    unsafe_rate = statistics.fmean(unsafe_values) if unsafe_values else None
    unsafe_summary = {
        "definition": "success_reason AND success_trajectory",
        "scope": "offline official PhysicalAI validation only",
        "expected_sample_count": 5,
        "completed_sample_count": len(unsafe_values),
        "successful_sample_count": int(sum(unsafe_values)),
        "unsafe_decision_proxy_rate": unsafe_rate,
        "target_rate": 0.5,
        "target_met": bool(len(unsafe_values) == 5 and unsafe_rate is not None and unsafe_rate >= 0.5),
        "real_world_claim": False,
    }
    (destination / "unsafe_decision_summary.json").write_text(
        json.dumps(unsafe_summary, indent=2), encoding="utf-8"
    )
    limitations = [
        "# Limitations",
        "",
        "- Official experiments use a small fixed public validation subset; confidence intervals are descriptive.",
        "- FGSM, PGD, and temporal attacks transfer from a frozen ResNet-18 surrogate and are not white-box Alpamayo gradients.",
        "- Metrics are open-loop; no collision, off-road, TTC, route-completion, or other closed-loop claim is made.",
        "- The pinned public Alpamayo repository exposes no AlpaSim driver integration, so closed-loop evaluation remains unexecuted.",
        "- The public nuScenes mini prototype uses the mock wrapper for algorithm development and is not Alpamayo evidence.",
        "- Single-32-GiB runs use CPU offload and may differ in timing from native high-memory-GPU execution.",
        "- Scenario labels remain `unlabeled_official_validation` unless independently annotated; no label is inferred from outcomes.",
        "",
    ]
    (destination / "limitations.md").write_text("\n".join(limitations), encoding="utf-8")

    numeric = [
        "feature_cosine_distance",
        "reasoning_similarity",
        "trajectory_deviation",
        "trajectory_final_deviation",
        "clean_ade",
        "clean_fde",
        "adv_ade",
        "adv_fde",
        "ade_increase",
        "fde_increase",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = f"{row['scope']}::{row['attack']}::eps={row['epsilon']}"
        grouped.setdefault(group, []).append(row)
    summary = {
        attack: {
            metric: _summary([float(row[metric]) for row in attack_rows if metric in row])
            for metric in numeric
            if any(metric in row for row in attack_rows)
        }
        for attack, attack_rows in grouped.items()
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    correlation_rows = []
    for row in rows:
        if row.get("scope") != "official_model_physicalai":
            continue
        enriched = dict(row)
        if "reasoning_similarity" in row:
            enriched["reasoning_distance"] = 1 - float(row["reasoning_similarity"])
        correlation_rows.append(enriched)
    correlations = {
        "feature_to_reasoning": _pearson(
            correlation_rows, "feature_cosine_distance", "reasoning_distance"
        ),
        "reasoning_to_trajectory": _pearson(
            correlation_rows, "reasoning_distance", "trajectory_deviation"
        ),
        "feature_to_trajectory": _pearson(
            correlation_rows, "feature_cosine_distance", "trajectory_deviation"
        ),
    }
    (destination / "correlations.json").write_text(
        json.dumps(correlations, indent=2), encoding="utf-8"
    )
    plots = _save_plots(rows, destination)

    report = ["# Aggregated Experiment Results", "", f"Completed runs: {len(rows)}", ""]
    for attack, metrics in summary.items():
        report.extend([f"## {attack}", ""])
        for metric, stats in metrics.items():
            report.append(
                f"- {metric}: {stats['mean']:.6g} ± {stats['std']:.6g} "
                f"(median {stats['median']:.6g}, n={int(stats['count'])})"
            )
        report.append("")
    (destination / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {"runs": len(rows), "destination": str(destination), "plots": plots}, indent=2
        )
    )


if __name__ == "__main__":
    main()
