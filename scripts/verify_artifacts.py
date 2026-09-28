#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from alpamayo_adv.attacks.base_attack import epsilon_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_attack(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = Path(manifest["output"])
    sample = Path(manifest["sample"])
    if not bundle.is_file() or not sample.is_file():
        raise FileNotFoundError(f"missing bundle or source for {manifest_path}")
    if _sha256(bundle) != manifest["sha256"]:
        raise AssertionError(f"SHA-256 mismatch: {bundle}")
    clean = torch.load(sample, map_location="cpu", weights_only=False)["image_frames"]
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    adv = payload["image_frames"] if isinstance(payload, dict) else payload
    if adv.dtype != torch.uint8 or adv.shape != clean.shape:
        raise AssertionError(f"invalid adversarial tensor in {bundle}")
    delta = (adv.to(torch.int16) - clean.to(torch.int16)).abs()
    linf = int(delta.max())
    budget = round(float(manifest["epsilon"]) * 255)
    if linf > budget or linf != int(manifest["linf_pixels"]):
        raise AssertionError(f"pixel budget mismatch in {bundle}: {linf} > {budget}")
    attacked_cameras = set(int(value) for value in manifest["attacked_cameras"])
    attacked_frames = set(int(value) for value in manifest.get("attacked_frames", range(clean.shape[1])))
    for camera in range(clean.shape[0]):
        for frame in range(clean.shape[1]):
            if camera not in attacked_cameras or frame not in attacked_frames:
                if not torch.equal(clean[camera, frame], adv[camera, frame]):
                    raise AssertionError(
                        f"unselected camera/frame changed in {bundle}: {camera}/{frame}"
                    )
    return {"bundle": str(bundle), "linf_pixels": linf, "sha256": manifest["sha256"]}


def _verify_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise AssertionError(f"result is not completed: {path}")
    for sibling in ("result.csv", "report.md", "comparison.png"):
        if not (path.parent / sibling).is_file():
            raise FileNotFoundError(f"missing {sibling} beside {path}")
    epsilon = epsilon_from_config(payload["config"]["attack"].get("epsilon", 0))
    linf = float(payload["metrics"]["linf"])
    if linf > epsilon + 1e-6:
        raise AssertionError(f"result pixel budget exceeded: {path}: {linf} > {epsilon}")
    if payload["metrics"]["pixel_min"] < 0 or payload["metrics"]["pixel_max"] > 1:
        raise AssertionError(f"result pixels outside [0,1]: {path}")
    return {
        "result": str(path),
        "scope": payload.get("result_scope"),
        "linf": linf,
        "epsilon": epsilon,
    }


def _read_key_value(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def _verify_nuscenes(project: Path) -> dict[str, Any] | None:
    root = project / "data/nuscenes"
    marker_path = root / ".v1.0-mini.complete"
    if not marker_path.is_file():
        return None
    marker = _read_key_value(marker_path)
    archive = root / "v1.0-mini.tgz"
    expected_bytes = int(marker["bytes"])
    if archive.stat().st_size != expected_bytes:
        raise AssertionError("nuScenes archive byte count does not match completion marker")
    if _sha256(archive) != marker["sha256"]:
        raise AssertionError("nuScenes archive SHA-256 does not match completion marker")
    sample_metadata = json.loads((root / "v1.0-mini/sample.json").read_text(encoding="utf-8"))
    front_images = list((root / "samples/CAM_FRONT").glob("*"))
    manifest_path = project / "outputs/nuscenes_balanced_100_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(sample_metadata) != 404 or len(front_images) != 404:
        raise AssertionError("nuScenes mini expected 404 samples and CAM_FRONT images")
    if manifest["sample_count"] != 100 or manifest["scene_count"] != 10:
        raise AssertionError("balanced nuScenes manifest must contain 100 samples over 10 scenes")
    if manifest["missing_camera_file_count"] != 0:
        raise AssertionError("balanced nuScenes manifest reports missing camera files")
    return {
        "archive": str(archive),
        "bytes": expected_bytes,
        "sha256": marker["sha256"],
        "official_sample_count": len(sample_metadata),
        "balanced_sample_count": manifest["sample_count"],
        "balanced_scene_count": manifest["scene_count"],
    }


def _verify_nuscenes_prototype(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise AssertionError(f"nuScenes prototype is not completed: {path}")
    rows_path = path.parent / "sample_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = payload["sample_count"] * len(payload["attacks"]) * len(payload["epsilon_pixels"])
    if len(rows) != expected or payload["result_row_count"] != expected:
        raise AssertionError(f"nuScenes prototype row-count mismatch: {path}")
    for row in rows:
        epsilon = float(row["epsilon_pixels"]) / 255.0
        if float(row["linf"]) > epsilon + 1e-6:
            raise AssertionError(f"nuScenes prototype pixel budget exceeded: {path}")
        if float(row["pixel_min"]) < 0 or float(row["pixel_max"]) > 1:
            raise AssertionError(f"nuScenes prototype pixels outside [0,1]: {path}")
    return {
        "summary": str(path),
        "samples": payload["sample_count"],
        "rows": len(rows),
        "scope": payload.get("result_scope"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify generated attack/results")
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", default="outputs/artifact_verification.json")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    attacks = [
        _verify_attack(path)
        for path in sorted((project / "data/physicalai/attacks").glob("*.json"))
    ]
    results = [
        _verify_result(path) for path in sorted((project / "outputs").glob("*/result.json"))
    ]
    nuscenes = _verify_nuscenes(project)
    nuscenes_prototypes = [
        _verify_nuscenes_prototype(path)
        for path in sorted((project / "outputs").glob("nuscenes_*/summary.json"))
    ]
    report = {
        "status": "verified",
        "attacks": attacks,
        "results": results,
        "nuscenes": nuscenes,
        "nuscenes_prototypes": nuscenes_prototypes,
    }
    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "attacks": len(attacks),
                "results": len(results),
                "nuscenes_prototypes": len(nuscenes_prototypes),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
