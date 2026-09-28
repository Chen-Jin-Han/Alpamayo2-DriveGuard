#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from alpamayo_adv.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an explicitly camera-masked attack")
    parser.add_argument("--config", default="configs/multiview.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cameras = config["attack"].get("attacked_cameras")
    if not cameras:
        raise SystemExit("multiview config must define attack.attacked_cameras")
    print(json.dumps(run_experiment(args.config), indent=2))


if __name__ == "__main__":
    main()
