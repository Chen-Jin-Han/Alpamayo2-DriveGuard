import pytest

from alpamayo_adv.datasets.nuscenes_dataset import (
    NuScenesSequenceConfig,
    scene_matches,
    select_samples,
)


def test_scene_filter_is_case_insensitive_and_optional():
    description = "Busy intersection with pedestrians and parked vehicles"
    assert scene_matches(description, [])
    assert scene_matches(description, ["PEDESTRIAN"])
    assert not scene_matches(description, ["highway"])


def test_nuscenes_config_rejects_partial_resize():
    config = NuScenesSequenceConfig(dataroot="missing", resize_height=224)
    with pytest.raises(ValueError, match="set together"):
        # Validation happens before the optional devkit import.
        from alpamayo_adv.datasets.nuscenes_dataset import NuScenesSequenceDataset

        NuScenesSequenceDataset(config)


def test_round_robin_selection_balances_scenes():
    samples = [
        {"scene_token": "a", "id": 1},
        {"scene_token": "a", "id": 2},
        {"scene_token": "a", "id": 3},
        {"scene_token": "b", "id": 4},
        {"scene_token": "b", "id": 5},
    ]
    assert [row["id"] for row in select_samples(samples, 4, "round_robin")] == [1, 4, 2, 5]
