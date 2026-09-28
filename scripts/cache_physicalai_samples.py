#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from alpamayo_adv.datasets import load_official_sample


def _safe_sample(data: dict[str, Any], clip_id: str, t0_us: int) -> tuple[dict[str, Any], list[str]]:
    safe: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            safe[key] = value.detach().cpu()
        elif isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif key in {"camera_names", "input_profile"}:
            safe[key] = value
        else:
            skipped.append(key)
    safe.update(clip_id=clip_id, t0_us=t0_us, offline_bundle_version=1)
    return safe, skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache official PhysicalAI samples for offline runs")
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    entries = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))["samples"]
    entries = entries[args.start_index :]
    if args.limit is not None:
        entries = entries[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    for offset, entry in enumerate(entries, start=args.start_index):
        clip_id = str(entry["clip_id"])
        t0_us = int(entry["t0_us"])
        output = output_dir / f"validation_sample_{offset}.pt"
        manifest_path = output.with_suffix(".manifest.json")
        if args.skip_existing and output.is_file() and manifest_path.is_file():
            print(f"skip_existing index={offset} clip_id={clip_id}", flush=True)
            completed.append(str(output))
            continue
        print(f"download_start index={offset} clip_id={clip_id} t0_us={t0_us}", flush=True)
        data = load_official_sample(clip_id, t0_us)
        safe, skipped = _safe_sample(data, clip_id, t0_us)
        tmp = output.with_suffix(".pt.tmp")
        torch.save(safe, tmp)
        tmp.replace(output)
        tensors = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in safe.items()
            if isinstance(value, torch.Tensor)
        }
        manifest = {
            "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
            "clip_id": clip_id,
            "t0_us": t0_us,
            "note": entry.get("note"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle": output.name,
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "camera_names": safe.get("camera_names"),
            "camera_indices": safe["camera_indices"].tolist(),
            "tensors": tensors,
            "skipped_nonserializable_keys": skipped,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        completed.append(str(output))
        print(
            f"download_complete index={offset} bytes={output.stat().st_size} "
            f"sha256={manifest['sha256']}",
            flush=True,
        )
    print(json.dumps({"completed": completed}, indent=2))


if __name__ == "__main__":
    main()
