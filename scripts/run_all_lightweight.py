#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from alpamayo_adv.experiment import run_experiment


CONFIGS = (
    "configs/clean.yaml",
    "configs/fgsm.yaml",
    "configs/pgd.yaml",
    "configs/btc_uap.yaml",
    "configs/multiview.yaml",
    "configs/defense.yaml",
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    results = []
    for relative in CONFIGS:
        print(f"running={relative}", flush=True)
        results.append(run_experiment(project / relative))
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/aggregate_results.py"),
            "--outputs",
            str(project / "outputs"),
            "--destination",
            str(project / "outputs/tables"),
        ],
        check=True,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
