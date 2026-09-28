#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    checks = {
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "nvcc": shutil.which("nvcc") is not None,
        "uv": shutil.which("uv") is not None,
        "hf_cli": shutil.which("hf") is not None,
        "official_package": importlib.util.find_spec("alpamayo2_super") is not None,
        "flash_attention": importlib.util.find_spec("flash_attn") is not None,
        "hf_token_env": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "model_id": os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", "nvidia/Alpamayo2-Super"),
        "project": str(Path.cwd()),
    }
    if checks["hf_cli"]:
        checks["hf_authenticated"] = subprocess.run(
            ["hf", "auth", "whoami"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    else:
        checks["hf_authenticated"] = False
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = os.environ.get("HF_TOKEN") or None
        api = HfApi(token=token)
        hf_hub_download("nvidia/Alpamayo2-Super", "config.json", token=token)
        checks["model_access"] = True
    except Exception:
        checks["model_access"] = False
    try:
        api.list_repo_refs("nvidia/PhysicalAI-Autonomous-Vehicles", repo_type="dataset")
        checks["dataset_access"] = True
    except Exception:
        checks["dataset_access"] = False
    print(json.dumps(checks, indent=2))
    required = (
        "nvidia_smi", "uv", "official_package", "flash_attention",
        "model_access",
    )
    if not all(checks[key] for key in required):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
