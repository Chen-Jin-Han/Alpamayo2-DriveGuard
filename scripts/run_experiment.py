#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alpamayo_adv.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.config), indent=2))


if __name__ == "__main__":
    main()

