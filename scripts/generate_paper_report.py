#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _rows(outputs: Path) -> list[dict[str, Any]]:
    rows = []
    for result_path in sorted(outputs.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("result_scope") != "official_model_physicalai":
            continue
        config = payload["config"]
        attack = config["attack"]
        rows.append(
            {
                "run": result_path.parent.name,
                "sample": Path(config.get("data", {}).get("path", "unknown")).stem,
                "attack": attack.get("label", attack["name"]),
                "epsilon": attack.get("epsilon"),
                "metrics": payload["metrics"],
            }
        )
    return rows


def _has(rows: list[dict[str, Any]], fragment: str) -> bool:
    return any(fragment.lower() in row["attack"].lower() for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-bounded paper-style report")
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", default="outputs/PAPER_REPORT.md")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    outputs = project / "outputs"
    rows = _rows(outputs)
    manifests = sorted((project / "data/physicalai").glob("validation_sample_*.manifest.json"))
    correlations_path = outputs / "tables/correlations.json"
    correlations = (
        json.loads(correlations_path.read_text(encoding="utf-8"))
        if correlations_path.is_file()
        else {}
    )
    nuscenes_summary_path = outputs / "nuscenes_study/three_seed_attack_summary.json"
    nuscenes_summary = (
        json.loads(nuscenes_summary_path.read_text(encoding="utf-8"))
        if nuscenes_summary_path.is_file()
        else []
    )
    unsafe_summary_path = outputs / "tables/unsafe_decision_summary.json"
    unsafe_summary = (
        json.loads(unsafe_summary_path.read_text(encoding="utf-8"))
        if unsafe_summary_path.is_file()
        else {}
    )

    lines = [
        "# Alpamayo 2 Multi-View Temporal Adversarial Robustness",
        "",
        "## Evidence boundary",
        "",
        f"This report contains {len(rows)} completed real-data runs over "
        f"{len(manifests)} cached official validation samples. Mock and synthetic runs are excluded from this count.",
        "All attacks are offline pixel-domain experiments; no real vehicle or public-road system is involved.",
        "Transfer attacks use a frozen ImageNet ResNet-18 surrogate and are not white-box gradients through Alpamayo.",
        "nuScenes mock-wrapper prototypes are reported separately and are never counted as official runs.",
        "",
        "## Method",
        "",
        "Inputs use six official cameras, four temporal frames, and source-pixel perturbations projected to the configured Linf budget.",
        "Offline success is fixed before evaluation: feature cosine distance > 0.05, reasoning similarity < 0.8 (or structured entity/action mismatch), trajectory deviation > 0.5 m, ADE increase > 0.5 m, or FDE increase > 1.0 m.",
        "",
        "## Completed real-data runs",
        "",
        "| Run | Sample | Attack | epsilon | ADE | FDE | Trajectory deviation | Reasoning similarity | Unsafe proxy | ASR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row["metrics"]
        lines.append(
            "| {run} | {sample} | {attack} | {epsilon} | {ade} | {fde} | {traj} | {reason} | {unsafe} | {asr} |".format(
                run=row["run"],
                sample=row["sample"],
                attack=row["attack"],
                epsilon=_fmt(row["epsilon"]),
                ade=_fmt(metric.get("adv_ade")),
                fde=_fmt(metric.get("adv_fde")),
                traj=_fmt(metric.get("trajectory_deviation")),
                reason=_fmt(metric.get("reasoning_similarity")),
                unsafe=_fmt(
                    metric.get(
                        "unsafe_decision_proxy",
                        float(
                            float(metric.get("success_reason", 0)) >= 1
                            and float(metric.get("success_trajectory", 0)) >= 1
                        ),
                    )
                ),
                asr=_fmt(metric.get("asr_sim")),
            )
        )

    lines.extend(["", "## Offline unsafe-next-action proxy", ""])
    if unsafe_summary:
        lines.extend(
            [
                "The strict proxy is positive only when both a reasoning/action semantic failure and a pre-registered trajectory failure occur in the same run.",
                f"Canonical temporal attack: {unsafe_summary.get('successful_sample_count', 0)}/{unsafe_summary.get('completed_sample_count', 0)} official samples "
                f"(rate `{_fmt(unsafe_summary.get('unsafe_decision_proxy_rate'))}`; target `0.5`; target met: `{unsafe_summary.get('target_met')}`).",
                "This is an offline open-loop safety proxy, not evidence that a real vehicle would execute a dangerous maneuver.",
            ]
        )
    else:
        lines.append("The five-sample strict proxy aggregation is pending.")

    rq_status = {
        "RQ1 — frame-wise attacks": "reported above"
        if _has(rows, "fgsm") or _has(rows, "pgd")
        else "pending official transfer runs",
        "RQ2 — temporal versus frame-wise": "reported above"
        if _has(rows, "temporal") and (_has(rows, "fgsm") or _has(rows, "pgd"))
        else "pending matched official runs",
        "RQ3 — multi-view redundancy": "reported above"
        if _has(rows, "cam1") and _has(rows, "cam3")
        else "pending 1/3/6-camera matrix",
        "RQ4 — feature shift to CoC": "correlation available"
        if correlations.get("feature_to_reasoning") is not None
        else "pending sufficient real runs",
        "RQ5 — CoC to trajectory": "correlation available"
        if correlations.get("reasoning_to_trajectory") is not None
        else "pending sufficient real runs",
        "RQ6 — traffic scenarios": "nuScenes metadata-proxy analysis available; official clips remain unlabeled",
        "RQ7 — temporal defense": "reported above"
        if _has(rows, "jpeg")
        else "pending official defense run",
    }
    lines.extend(["", "## Research questions", ""])
    lines.extend(f"- {name}: {status}." for name, status in rq_status.items())

    lines.extend(["", "## nuScenes algorithm-development prototype", ""])
    if nuscenes_summary:
        lines.extend(
            [
                "The following balanced 100-sample, three-seed results use the deterministic mock wrapper and are not Alpamayo evidence.",
                "",
                "| Attack | Epsilon | N | ASR | Feature distance | Reasoning similarity | Trajectory deviation |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in nuscenes_summary:
            lines.append(
                f"| {item['attack']} | {item['epsilon_pixels']}/255 | {item['sample_count']} | "
                f"{_fmt(item.get('asr_sim_mean'))} | {_fmt(item.get('feature_cosine_distance_mean'))} | "
                f"{_fmt(item.get('reasoning_similarity_mean'))} | "
                f"{_fmt(item.get('trajectory_deviation_mean'))} |"
            )
    else:
        lines.append("The reproducible nuScenes three-seed aggregation is still running.")

    lines.extend(
        [
            "",
            "## Correlation analysis",
            "",
            f"- feature → reasoning: `{_fmt(correlations.get('feature_to_reasoning'))}`",
            f"- reasoning → trajectory: `{_fmt(correlations.get('reasoning_to_trajectory'))}`",
            f"- feature → trajectory: `{_fmt(correlations.get('feature_to_trajectory'))}`",
            "",
            "## Reproducibility",
            "",
            "- Official random seed: 42; nuScenes prototype main comparison uses seeds 42, 43, and 44.",
            "- Official source commit: `5e7975f4a2100ee8ac1a62b79239bbabefddcbef`.",
            "- Verify bundles and results with `python scripts/verify_artifacts.py --project .`.",
            "- Regenerate official tables with `python scripts/aggregate_results.py --outputs outputs --destination outputs/tables`.",
            "- Regenerate nuScenes tables with `python scripts/aggregate_nuscenes_prototypes.py --outputs outputs --destination outputs/nuscenes_study`.",
            "- The final authorized four-GPU batch is reproduced with `bash scripts/run_all_gpu_suites.sh`; the courtesy scheduler remains available as `scripts/wait_for_gpus.py`.",
            "",
            "## Limitations",
            "",
            "- The official validation subset is small and fixed; confidence intervals are descriptive.",
            "- AlpaSim prerequisites and an official Alpamayo driver adapter are unavailable, so no closed-loop safety claim is made.",
            "- nuScenes prototype results use the mock wrapper for algorithm development and do not measure Alpamayo robustness.",
            "- CPU offload permits execution on 32 GiB GPUs but changes runtime characteristics.",
            "- Pending rows are never replaced by mock or synthetic evidence.",
            "",
        ]
    )
    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"real_runs": len(rows), "samples": len(manifests), "output": str(output)}))


if __name__ == "__main__":
    main()
