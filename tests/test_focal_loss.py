import torch

from fractal_swin_unet.losses import focal_loss


def test_focal_loss_confidence_cases() -> None:
    logits_good = torch.tensor([[[[10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]]]])
    logits_bad = torch.tensor([[[[-10.0, 10.0]]]])

    loss_good = focal_loss(logits_good, targets, alpha=0.25, gamma=2.0)
    loss_bad = focal_loss(logits_bad, targets, alpha=0.25, gamma=2.0)

    assert loss_good.item() < loss_bad.item()
    assert loss_good.item() < 1e-3
