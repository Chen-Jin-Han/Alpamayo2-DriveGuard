from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class AlpaSimStatus:
    package_available: bool
    scene_root_available: bool
    driver_adapter_available: bool
    ready: bool
    scene_root: str | None
    reason: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def probe_alpasim(
    scene_root: str | None = None,
    driver_adapter: str | None = None,
) -> AlpaSimStatus:
    """Probe closed-loop prerequisites without inventing simulator compatibility."""
    requested_root = scene_root or os.environ.get("ALPASIM_SCENE_ROOT")
    requested_adapter = driver_adapter or os.environ.get("ALPAMAYO_ALPASIM_DRIVER")
    package_available = find_spec("alpasim") is not None
    scene_available = bool(requested_root and Path(requested_root).is_dir())
    adapter_available = bool(requested_adapter and Path(requested_adapter).is_file())
    ready = package_available and scene_available and adapter_available
    missing = []
    if not package_available:
        missing.append("AlpaSim Python package")
    if not scene_available:
        missing.append("licensed scene root")
    if not adapter_available:
        missing.append("verified Alpamayo driver adapter")
    reason = "ready" if ready else "missing: " + ", ".join(missing)
    return AlpaSimStatus(
        package_available=package_available,
        scene_root_available=scene_available,
        driver_adapter_available=adapter_available,
        ready=ready,
        scene_root=requested_root,
        reason=reason,
    )
