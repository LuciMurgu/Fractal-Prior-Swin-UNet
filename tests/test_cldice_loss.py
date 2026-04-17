import torch

from fractal_swin_unet.losses import cldice_loss


def _make_line_mask(size: int = 32) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size)
    mask[0, 0, size // 2, :] = 1.0
    return mask


def test_cldice_loss_perfect_vs_broken() -> None:
    target = _make_line_mask()
    probs_perfect = target.clone()

    probs_broken = target.clone()
    probs_broken[0, 0, target.shape[-2] // 2, 10:22] = 0.0

    loss_perfect = cldice_loss(probs_perfect, target, iters=8)
    loss_broken = cldice_loss(probs_broken, target, iters=8)

    assert loss_perfect.item() < loss_broken.item()
    assert 0.0 <= loss_perfect.item() <= 1.5
    assert 0.0 <= loss_broken.item() <= 1.5
