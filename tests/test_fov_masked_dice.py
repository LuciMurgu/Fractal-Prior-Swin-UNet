import torch

from fractal_swin_unet.metrics import dice_score


def test_fov_masked_dice_improves() -> None:
    gt = torch.zeros(1, 1, 32, 32)
    gt[0, 0, 8:24, 8:24] = 1.0

    pred = gt.clone()
    pred[0, 0, 0:4, 0:4] = 1.0

    fov = torch.zeros(32, 32)
    fov[8:24, 8:24] = 1.0

    dice_full = dice_score(pred, gt)
    dice_fov = dice_score(pred, gt, mask=fov)

    assert dice_fov >= dice_full
