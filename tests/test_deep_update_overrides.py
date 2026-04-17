from fractal_swin_unet.exp.matrix import deep_update


def test_deep_update_dot_overrides() -> None:
    cfg = {"loss": {"cldice": {"enabled": False}}, "model": {"name": "baseline"}}
    overrides = {"loss.cldice.enabled": True, "model.name": "fractal_prior"}
    out = deep_update(cfg, overrides)

    assert out["loss"]["cldice"]["enabled"] is True
    assert out["model"]["name"] == "fractal_prior"


def test_deep_update_creates_nested() -> None:
    cfg = {}
    overrides = {"a.b.c": 1}
    out = deep_update(cfg, overrides)
    assert out["a"]["b"]["c"] == 1
