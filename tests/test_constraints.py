import torch

from alpamayo_adv.attacks import fgsm, pgd
from alpamayo_adv.attacks.base_attack import project_linf
from alpamayo_adv.attacks.surrogate_transfer import _attack_mask, _normalize_kind


def test_projection_respects_pixel_budget():
    clean = torch.rand(2, 3, 3, 8, 8)
    epsilon = 16 / 255
    adv = project_linf(clean + torch.randn_like(clean), clean, epsilon)
    assert (adv - clean).abs().max() <= epsilon + 1e-7
    assert adv.min() >= 0 and adv.max() <= 1


def test_epsilon_zero_is_identity():
    clean = torch.rand(2, 3, 3, 8, 8)
    objective = lambda value: value.square().mean()
    assert torch.equal(clean, fgsm(clean, objective, 0.0))
    assert torch.equal(clean, pgd(clean, objective, 0.0, 0.0, 2))


def test_camera_mask():
    clean = torch.rand(3, 2, 3, 8, 8)
    adv = fgsm(clean, lambda value: value.mean(), 4 / 255, attacked_cameras=[1])
    assert torch.equal(clean[0], adv[0])
    assert not torch.equal(clean[1], adv[1])
    assert torch.equal(clean[2], adv[2])


def test_frame_mask():
    clean = torch.rand(2, 4, 3, 8, 8)
    adv = fgsm(
        clean,
        lambda value: value.mean(),
        4 / 255,
        attacked_cameras=[0, 1],
        attacked_frames=[3],
    )
    assert torch.equal(clean[:, :3], adv[:, :3])
    assert not torch.equal(clean[:, 3], adv[:, 3])


def test_surrogate_attack_helpers():
    mask = _attack_mask(6, 4, (1, 3, 5), (2, 3), torch.device("cpu"))
    assert mask[:, 0, 0, 0, 0].tolist() == [0, 0, 0, 0, 0, 0]
    assert mask[:, 2, 0, 0, 0].tolist() == [0, 1, 0, 1, 0, 1]
    assert _normalize_kind("temporal") == "temporal"
