#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("aggregate_results.py")
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
