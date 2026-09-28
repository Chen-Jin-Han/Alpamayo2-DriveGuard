#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from alpamayo_adv.attacks.base_attack import epsilon_from_config
from alpamayo_adv.attacks.surrogate_transfer import generate_resnet18_transfer_attack
from alpamayo_adv.datasets import load_cached_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen-ResNet transfer attack")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["fgsm", "pgd", "temporal"], required=True)
    parser.add_argument("--epsilon", default="16/255")
    parser.add_argument("--step-size", default="2/255")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cameras", type=int, nargs="*")
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample = load_cached_sample(args.sample)
    clean = sample["image_frames"]
    epsilon = epsilon_from_config(args.epsilon)
    adv, info = generate_resnet18_transfer_attack(
        clean,
        kind=args.kind,
        epsilon=epsilon,
        step_size=epsilon_from_config(args.step_size),
        iterations=args.iterations,
        attacked_cameras=args.cameras,
        attacked_frames=args.frames,
        seed=args.seed,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save({"image_frames": adv, "attack": asdict(info)}, tmp)
    tmp.replace(output)
    delta = (adv.to(torch.int16) - clean.to(torch.int16)).abs()
    metadata = {
        **asdict(info),
        "sample": str(Path(args.sample).resolve()),
        "output": str(output.resolve()),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "linf_pixels": int(delta.max()),
        "changed_pixels": int((delta > 0).sum()),
        "pixel_min": int(adv.min()),
        "pixel_max": int(adv.max()),
        "surrogate": "torchvision ResNet-18 ImageNet1K_V1",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
