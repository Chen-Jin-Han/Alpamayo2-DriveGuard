import torch
import pytest

from alpamayo_adv.datasets import load_cached_sample, make_mock_sample
from alpamayo_adv.attacks import fgsm
from alpamayo_adv.experiment import _fgsm_objective, _load_reference_output, _project_root_for_config
from alpamayo_adv.metrics import attack_success_metrics, evaluate_pair
from alpamayo_adv.metrics.attack_metrics import _trajectory_errors
from alpamayo_adv.models import MockAlpamayoWrapper


def test_mock_pipeline_shapes_and_metrics():
    data = make_mock_sample(height=24, width=32)
    wrapper = MockAlpamayoWrapper()
    output = wrapper.forward_clean(data)
    assert output.trajectory.shape == (64, 3)
    assert output.visual_features.shape[:2] == (6, 4)
    metrics = evaluate_pair(data["image_frames"], data["image_frames"], output, output)
    assert metrics["linf"] == 0
    assert metrics["trajectory_deviation"] == 0
    assert metrics["reasoning_similarity"] == 1
    assert metrics["reasoning_entity_f1"] == 1
    assert metrics["reasoning_action_f1"] == 1
    assert metrics["longitudinal_deviation"] == 0
    assert metrics["lateral_deviation"] == 0
    assert metrics["heading_deviation_radians"] == 0


def test_ground_truth_is_moved_to_prediction_device():
    data = make_mock_sample(height=24, width=32)
    wrapper = MockAlpamayoWrapper(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    output = wrapper.forward_clean(data)
    metrics = evaluate_pair(
        data["image_frames"].to(output.trajectory.device),
        data["image_frames"].to(output.trajectory.device),
        output,
        output,
        data["ego_future_xyz"],
    )
    assert "clean_ade" in metrics


def test_cached_official_sample_and_batched_ground_truth(tmp_path):
    sample_path = tmp_path / "sample.pt"
    sample = {
        "image_frames": torch.zeros(6, 4, 3, 8, 8, dtype=torch.uint8),
        "camera_indices": torch.tensor([0, 1, 2, 3, 5, 6]),
        "ego_history_xyz": torch.zeros(1, 1, 16, 3),
        "ego_history_rot": torch.eye(3).expand(1, 1, 16, 3, 3).clone(),
    }
    torch.save(sample, sample_path)
    loaded = load_cached_sample(sample_path)
    assert loaded["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]

    pred = torch.zeros(64, 3)
    gt = torch.ones(1, 1, 64, 3)
    ade, fde = _trajectory_errors(pred, gt)
    assert ade == pytest.approx(2 ** 0.5)
    assert fde == pytest.approx(2 ** 0.5)


def test_attack_success_uses_fixed_thresholds():
    metrics = {
        "feature_cosine_distance": 0.06,
        "reasoning_similarity": 0.9,
        "trajectory_deviation": 0.1,
        "ade_increase": 0.0,
        "fde_increase": 0.0,
    }
    success = attack_success_metrics(metrics)
    assert success == {
        "success_feature": 1.0,
        "success_reason": 0.0,
        "success_trajectory": 0.0,
        "unsafe_decision_proxy": 0.0,
        "asr_sim": 1.0,
    }


def test_fgsm_pipeline_objective_has_nonzero_clean_gradient():
    data = make_mock_sample(cameras=2, frames=2, height=16, width=24)
    wrapper = MockAlpamayoWrapper()
    clean = wrapper.preprocess(data)["image_frames"]
    clean_output = wrapper.forward_clean(data)
    adv = fgsm(clean, _fgsm_objective(wrapper, clean_output), epsilon=4 / 255)
    assert float((adv - clean).abs().max()) == pytest.approx(4 / 255, abs=1e-6)
    assert not torch.equal(adv, clean)


def test_reference_output_requires_matching_clean_inference(tmp_path):
    data = make_mock_sample(cameras=2, frames=2, height=16, width=24)
    clean = MockAlpamayoWrapper().forward_clean(data)
    path = tmp_path / "reference.json"
    payload = {
        "status": "completed",
        "config": {
            "data": {"path": "sample.pt"},
            "attack": {"path": "attack.pt", "epsilon": "16/255"},
        },
        "metrics": {"linf": 16 / 255},
        "clean_reasoning": clean.reasoning,
        "adversarial_reasoning": "changed",
        "clean_trajectory": clean.trajectory.tolist(),
        "adversarial_trajectory": (clean.trajectory + 1).tolist(),
    }
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    config = {
        "data": {"path": "sample.pt"},
        "attack": {"path": "attack.pt", "epsilon": "16/255"},
    }
    loaded, adv = _load_reference_output(path, config, clean)
    assert loaded["metrics"]["linf"] == pytest.approx(16 / 255)
    assert adv.reasoning == "changed"
    assert torch.allclose(adv.trajectory, clean.trajectory + 1)


def test_nested_config_resolves_project_root(tmp_path):
    project = tmp_path / "project"
    config = project / "configs" / "generated" / "run.yaml"
    config.parent.mkdir(parents=True)
    (project / "src" / "alpamayo_adv").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config.write_text("backend: mock\n", encoding="utf-8")
    assert _project_root_for_config(config) == project
