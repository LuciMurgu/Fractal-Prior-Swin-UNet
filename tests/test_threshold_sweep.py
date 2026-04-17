import torch

from fractal_swin_unet.exp.threshold import sweep_thresholds


def test_threshold_sweep_selects_best_tau() -> None:
    gts = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    probs = torch.tensor([[[[0.4, 0.05], [0.05, 0.4]]]])
    grid = [round(0.1 + i * 0.05, 2) for i in range(17)]

    result = sweep_thresholds(probs, gts, grid)

    assert result["tau_star"] in {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}
    assert result["best_dice"] > result["dice_at_0_5"]
    assert 0.0 <= result["best_dice"] <= 1.0
    assert 0.0 <= result["dice_at_0_5"] <= 1.0
