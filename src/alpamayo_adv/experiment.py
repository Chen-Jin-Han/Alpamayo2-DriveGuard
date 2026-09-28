from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from alpamayo_adv.attacks import fgsm, pgd, random_noise, temporal_pgd
from alpamayo_adv.attacks.base_attack import epsilon_from_config
from alpamayo_adv.datasets import load_cached_sample, make_mock_sample, make_official_synthetic_sample
from alpamayo_adv.defenses import gaussian_denoise, jpeg_compress, temporal_smooth
from alpamayo_adv.metrics import attack_success_metrics, evaluate_pair
from alpamayo_adv.models import MockAlpamayoWrapper, OfficialAlpamayoWrapper
from alpamayo_adv.models.alpamayo_wrapper import validate_image_frames
from alpamayo_adv.types import InferenceOutput
from alpamayo_adv.visualization import save_comparison


def _objective(wrapper, clean_output, weights: dict[str, float]):
    clean_features = clean_output.visual_features.detach()
    clean_trajectory = clean_output.trajectory.detach()

    def objective(frames: torch.Tensor) -> torch.Tensor:
        output = wrapper.differentiable_forward(frames)
        feature = (output.visual_features - clean_features).square().mean()
        trajectory = (output.trajectory - clean_trajectory).square().mean()
        temporal = (output.visual_features[:, 1:] - output.visual_features[:, :-1]).square().mean()
        return (
            weights.get("feature", 1.0) * feature
            + weights.get("trajectory", 1.0) * trajectory
            + weights.get("temporal", 0.0) * temporal
        )

    return objective


def _fgsm_objective(wrapper, clean_output):
    """A fixed first-order feature direction with non-zero gradient at the clean point."""
    clean_features = clean_output.visual_features.detach()
    centered = clean_features - clean_features.mean()
    direction = centered.sign()
    if not torch.any(direction):
        direction = torch.ones_like(clean_features)

    def objective(frames: torch.Tensor) -> torch.Tensor:
        features = wrapper.differentiable_forward(frames).visual_features
        return (features * direction).mean()

    return objective


def _write_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    clean_reasoning: str,
    adv_reasoning: str,
    clean_trajectory: torch.Tensor,
    adv_trajectory: torch.Tensor,
    defended_reasoning: str | None = None,
    defended_trajectory: torch.Tensor | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config["backend"] == "mock":
        result_scope = "mock_pipeline"
    elif config.get("data", {}).get("source") == "synthetic":
        result_scope = "official_model_synthetic_input"
    else:
        result_scope = "official_model_physicalai"
    payload = {
        "status": "completed",
        "result_scope": result_scope,
        "config": config,
        "metrics": metrics,
        "clean_reasoning": clean_reasoning,
        "adversarial_reasoning": adv_reasoning,
        "clean_trajectory": clean_trajectory.detach().cpu().tolist(),
        "adversarial_trajectory": adv_trajectory.detach().cpu().tolist(),
    }
    if defended_reasoning is not None and defended_trajectory is not None:
        payload["defended_reasoning"] = defended_reasoning
        payload["defended_trajectory"] = defended_trajectory.detach().cpu().tolist()
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader(); writer.writerow(metrics)
    report = [
        "# Experiment Result",
        "",
        f"- Backend: `{config['backend']}`",
        f"- Scope: `{payload['result_scope']}`",
        f"- Attack: `{config['attack'].get('label', config['attack']['name'])}`",
        f"- Epsilon: `{config['attack']['epsilon']}`",
        "",
        "Mock results validate code paths only and are not evidence about Alpamayo 2 Super.",
        "",
        "## Metrics",
        "",
        *[f"- {key}: {value:.8g}" for key, value in metrics.items()],
        "",
        "## Reasoning",
        "",
        f"- Clean: {clean_reasoning}",
        f"- Adversarial: {adv_reasoning}",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _load_reference_output(
    reference_path: Path,
    config: dict[str, Any],
    clean_output: InferenceOutput,
) -> tuple[dict[str, Any], InferenceOutput]:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"reference result is not completed: {reference_path}")
    reference_config = payload["config"]
    for section, key in (("data", "path"), ("attack", "path"), ("attack", "epsilon")):
        if str(reference_config.get(section, {}).get(key)) != str(config.get(section, {}).get(key)):
            raise ValueError(f"reference {section}.{key} does not match defense config")
    reference_clean_trajectory = torch.tensor(
        payload["clean_trajectory"],
        device=clean_output.trajectory.device,
        dtype=clean_output.trajectory.dtype,
    )
    if payload.get("clean_reasoning") != clean_output.reasoning:
        raise ValueError("reference clean reasoning does not match deterministic clean inference")
    if reference_clean_trajectory.shape != clean_output.trajectory.shape or not torch.allclose(
        reference_clean_trajectory, clean_output.trajectory, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("reference clean trajectory does not match deterministic clean inference")
    adversarial_trajectory = torch.tensor(
        payload["adversarial_trajectory"],
        device=clean_output.trajectory.device,
        dtype=clean_output.trajectory.dtype,
    )
    adversarial_output = InferenceOutput(
        reasoning=payload["adversarial_reasoning"],
        trajectory=adversarial_trajectory,
        visual_features=torch.empty(0, device=clean_output.trajectory.device),
        extra={"reused_reference": str(reference_path)},
    )
    return payload, adversarial_output


def _project_root_for_config(config_path: Path) -> Path:
    """Resolve the project root for configs at any nesting depth."""
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/alpamayo_adv").is_dir():
            return candidate
    raise ValueError(f"cannot locate project root for config: {config_path}")


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if config["backend"] == "mock":
        data = make_mock_sample(**config.get("mock_data", {}), seed=seed)
        wrapper = MockAlpamayoWrapper(device=device)
    elif config["backend"] == "official":
        if config["data"].get("source") == "synthetic":
            data = make_official_synthetic_sample(
                height=int(config["data"].get("height", 224)),
                width=int(config["data"].get("width", 384)),
                seed=seed,
            )
        elif config["data"].get("source") == "cached":
            data = load_cached_sample(config["data"]["path"])
        else:
            from alpamayo_adv.datasets.physicalai_dataset import load_official_sample

            data = load_official_sample(config["data"]["clip_id"], int(config["data"]["t0_us"]))
        wrapper = OfficialAlpamayoWrapper(
            model_id=config["model_id"],
            device_map=config.get("device_map", "cuda:0"),
            diffusion_steps=int(config.get("diffusion_steps", 10)),
            seed=seed,
            offload_folder=config.get("offload_folder"),
            max_memory=config.get("max_memory"),
        )
    else:
        raise ValueError("backend must be mock or official")

    data = wrapper.preprocess(data)
    clean_frames = data["image_frames"]
    clean = wrapper.forward_clean(data)
    attack = config["attack"]
    epsilon = epsilon_from_config(attack["epsilon"])
    attacked_cameras = attack.get("attacked_cameras")
    name = attack["name"]
    if name == "clean":
        adv_frames = clean_frames.clone()
    elif name == "random":
        adv_frames = random_noise(clean_frames, epsilon, seed, attacked_cameras)
    elif name == "precomputed":
        payload = torch.load(attack["path"], map_location="cpu", weights_only=False)
        stored_frames = payload["image_frames"] if isinstance(payload, dict) else payload
        adv_frames = validate_image_frames(stored_frames).to(clean_frames.device)
        if adv_frames.shape != clean_frames.shape:
            raise ValueError(
                f"precomputed attack shape {tuple(adv_frames.shape)} does not match "
                f"clean shape {tuple(clean_frames.shape)}"
            )
    elif name == "fgsm":
        objective = _fgsm_objective(wrapper, clean)
        adv_frames = fgsm(clean_frames, objective, epsilon, attacked_cameras)
    elif name == "pgd":
        objective = _objective(wrapper, clean, config.get("objective", {}))
        adv_frames = pgd(
            clean_frames, objective, epsilon, epsilon_from_config(attack["step_size"]),
            int(attack["iterations"]), bool(attack.get("random_start", True)), seed,
            attacked_cameras,
        )
    elif name in {"temporal", "btc_uap"}:
        objective = _objective(wrapper, clean, config.get("objective", {}))
        adv_frames = temporal_pgd(
            clean_frames, objective, epsilon, epsilon_from_config(attack["step_size"]),
            int(attack["iterations"]), bool(attack.get("shared_across_cameras", False)),
            seed, attacked_cameras,
        )
    else:
        raise ValueError(f"unknown attack: {name}")

    defense_name = config.get("defense", {}).get("name", "none")
    reference_payload = None
    reference_value = config.get("defense", {}).get("reference_result")
    if defense_name != "none" and reference_value:
        reference_path = Path(reference_value)
        if not reference_path.is_absolute():
            reference_path = _project_root_for_config(config_path) / reference_path
        reference_payload, adv = _load_reference_output(reference_path, config, clean)
        metrics = dict(reference_payload["metrics"])
        actual_linf = float((adv_frames - clean_frames).abs().max().item())
        if abs(actual_linf - float(metrics["linf"])) > 1e-6:
            raise ValueError("reference perturbation norm does not match configured attack bundle")
    else:
        # A clean baseline is an identity comparison and must not pay for a second
        # stochastic diffusion pass that could create a false non-zero deviation.
        adv = clean if name == "clean" else wrapper.forward_adv(data, adv_frames)
        metrics = evaluate_pair(clean_frames, adv_frames, clean, adv, data.get("ego_future_xyz"))
        metrics.update(attack_success_metrics(metrics, config.get("success_thresholds")))
    if metrics["linf"] > epsilon + 1e-6:
        raise AssertionError(f"pixel budget violated: {metrics['linf']} > {epsilon}")
    if metrics["pixel_min"] < 0 or metrics["pixel_max"] > 1:
        raise AssertionError("adversarial pixels are outside [0,1]")

    defended = None
    if defense_name != "none":
        if defense_name == "gaussian":
            defended_frames = gaussian_denoise(adv_frames)
        elif defense_name == "temporal_smoothing":
            defended_frames = temporal_smooth(adv_frames, config["defense"].get("alpha", 0.5))
        elif defense_name == "jpeg":
            defended_frames = jpeg_compress(adv_frames, int(config["defense"].get("quality", 75)))
        else:
            raise ValueError(f"unknown defense: {defense_name}")
        defended = wrapper.forward_adv(data, defended_frames)
        defended_metrics = evaluate_pair(
            clean_frames, defended_frames, clean, defended, data.get("ego_future_xyz")
        )
        defended_metrics.update(
            attack_success_metrics(defended_metrics, config.get("success_thresholds"))
        )
        metrics.update({f"defense_{key}": value for key, value in defended_metrics.items()})

    output_dir = Path(config.get("output_dir", "outputs/run"))
    if not output_dir.is_absolute():
        output_dir = _project_root_for_config(config_path) / output_dir
    _write_artifacts(
        output_dir,
        config,
        metrics,
        clean.reasoning,
        adv.reasoning,
        clean.trajectory,
        adv.trajectory,
        defended.reasoning if defended is not None else None,
        defended.trajectory if defended is not None else None,
    )
    save_comparison(clean_frames, adv_frames, clean.trajectory, adv.trajectory, output_dir / "comparison.png")
    return {"output_dir": str(output_dir), "metrics": metrics}
