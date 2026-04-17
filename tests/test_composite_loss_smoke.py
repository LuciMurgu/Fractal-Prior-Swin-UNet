import torch

from fractal_swin_unet.losses import CompositeLoss


def test_composite_loss_smoke() -> None:
    logits = torch.randn(2, 1, 16, 16)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

    default_cfg = {
        "from_logits": True,
        "dice": {"enabled": True, "weight": 1.0, "smooth": 1.0},
        "bce": {"enabled": True, "weight": 1.0, "pos_weight": None},
        "focal": {"enabled": False, "weight": 0.0, "alpha": 0.25, "gamma": 2.0, "reduction": "mean"},
        "cldice": {"enabled": False, "weight": 0.0, "iter": 5, "smooth": 1.0, "eps": 1e-6},
        "soft_dbc": {"enabled": False, "weight": 0.0, "smooth": 1.0, "eps": 1e-6},
    }
    topo_cfg = {
        **default_cfg,
        "focal": {"enabled": True, "weight": 0.1, "alpha": 0.25, "gamma": 2.0, "reduction": "mean"},
        "cldice": {"enabled": True, "weight": 0.2, "iter": 5, "smooth": 1.0, "eps": 1e-6},
        "soft_dbc": {"enabled": True, "weight": 0.05, "smooth": 1.0, "eps": 1e-6},
    }

    loss_fn = CompositeLoss(default_cfg)
    loss, breakdown = loss_fn(logits, targets)

    assert loss.shape == ()
    assert set(breakdown.keys()) == {
        "loss_total",
        "loss_dice",
        "loss_bce",
        "loss_focal",
        "loss_cldice",
        "loss_soft_dbc",
        "loss_cbdice",
        "loss_skel_recall",
        "loss_fractal_bce",
        "loss_tv",
        "loss_curvature",
    }
    assert not torch.isnan(loss)

    loss_fn_topo = CompositeLoss(topo_cfg)
    loss_topo, breakdown_topo = loss_fn_topo(logits, targets)
    assert loss_topo.shape == ()
    assert not torch.isnan(loss_topo)
    assert breakdown_topo["loss_cldice"] >= 0.0
    assert breakdown_topo["loss_soft_dbc"] >= 0.0