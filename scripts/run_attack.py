#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from alpamayo_adv.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an adversarial evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config["attack"]["name"] == "clean":
        raise SystemExit("run_attack.py requires a non-clean attack config")
    print(json.dumps(run_experiment(args.config), indent=2))


if __name__ == "__main__":
    main()
