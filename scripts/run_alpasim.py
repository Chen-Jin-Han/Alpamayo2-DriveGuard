#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alpamayo_adv.simulation import probe_alpasim


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AlpaSim prerequisites")
    parser.add_argument("--scene-root")
    parser.add_argument("--driver-adapter")
    args = parser.parse_args()
    status = probe_alpasim(args.scene_root, args.driver_adapter)
    print(status.to_json())
    if not status.ready:
        raise SystemExit(2)
    raise SystemExit(
        "Prerequisites exist, but no pinned public driver API was found in the official repository; "
        "do not claim a rollout until an adapter-specific implementation and test are added."
    )


if __name__ == "__main__":
    main()
