from scripts.prepare_nuscenes_manifest import scenario_tags
from scripts.run_nuscenes_prototype import summarize


def test_scenario_tags_are_case_insensitive_and_multilabel():
    assert scenario_tags("Night drive in heavy traffic") == ["night", "dense_traffic"]


def test_scenario_tags_cover_requested_metadata_proxies():
    tags = scenario_tags("Following a van, peds at intersection, car overtaking us")
    assert {"vehicle_following", "pedestrian", "intersection", "lane_change"} <= set(tags)


def test_scenario_tags_fall_back_to_unspecified():
    assert scenario_tags("Residential area") == ["unspecified"]


def test_prototype_summary_groups_attack_and_epsilon():
    rows = [
        {
            "sample_index": 0,
            "sample_token": "a",
            "scene_name": "scene-1",
            "attack": "fgsm",
            "epsilon_pixels": 16,
            "linf": 1.0,
        },
        {
            "sample_index": 1,
            "sample_token": "b",
            "scene_name": "scene-2",
            "attack": "fgsm",
            "epsilon_pixels": 16,
            "linf": 3.0,
        },
    ]
    result = summarize(rows)
    assert result[0]["sample_count"] == 2
    assert result[0]["scene_count"] == 2
    assert result[0]["linf_mean"] == 2.0
    assert result[0]["linf_median"] == 2.0
    assert result[0]["linf_ci95_low"] < 2.0 < result[0]["linf_ci95_high"]
