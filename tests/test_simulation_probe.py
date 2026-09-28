from alpamayo_adv.simulation import probe_alpasim


def test_alpasim_probe_does_not_claim_readiness_without_prerequisites(tmp_path):
    missing_adapter = tmp_path / "missing_driver.py"
    status = probe_alpasim(str(tmp_path), str(missing_adapter))
    assert status.scene_root_available
    assert not status.driver_adapter_available
    assert not status.ready
    assert "missing" in status.reason
