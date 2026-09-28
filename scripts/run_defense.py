#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from alpamayo_adv.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an attack plus defense evaluation")
    parser.add_argument("--config", default="configs/defense.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("defense", {}).get("name", "none") == "none":
        raise SystemExit("run_defense.py requires a non-empty defense")
    print(json.dumps(run_experiment(args.config), indent=2))


if __name__ == "__main__":
    main()
