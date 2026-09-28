#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alpamayo_adv.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean baseline")
    parser.add_argument("--config", default="configs/clean.yaml")
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
