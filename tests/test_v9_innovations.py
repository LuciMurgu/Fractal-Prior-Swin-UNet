"""Tests for V9 fractal innovations: multi-scale FFM, lacunarity,
cbDice, skeleton recall, fractal-weighted BCE, multi-decoder, FD regression."""

import torch
import pytest


# --- Phase 1: Multi-scale FFM + Lacunarity ---

def test_compute_multiscale_ffm_shape():
    """Multi-scale FFM returns correct number of channels."""
    from fractal_swin_unet.fractal import compute_multiscale_ffm
    image = torch.randn(3, 64, 64)
    ffm = compute_multiscale_ffm(image, window_sizes=(8, 16), include_lacunarity=True)
    assert ffm.ndim == 3
    assert ffm.shape[0] == 3  # 2 LFD + 1 lacunarity
    assert ffm.shape[1:] == (64, 64)


def test_compute_multiscale_ffm_no_lacunarity():
    from fractal_swin_unet.fractal import compute_multiscale_ffm
    image = torch.randn(3, 64, 64)
    ffm = compute_multiscale_ffm(image, window_sizes=(8, 16, 32), include_lacunarity=False)
    assert ffm.shape[0] == 3  # 3 LFD channels only


def test_lacunarity_map_shape_and_range():
    from fractal_swin_unet.fractal import compute_lacunarity_map
    image = torch.randn(3, 64, 64)
    lac = compute_lacunarity_map(image, window_size=16, box_sizes=(4, 8))
    assert lac.shape == (64, 64)
    assert lac.min() >= 0.0
    assert lac.max() <= 1.0


def test_gliding_box_lacunarity_scalar():
    from fractal_swin_unet.fractal import gliding_box_lacunarity
    patch = torch.randn(32, 32)
    lac = gliding_box_lacunarity(patch, box_sizes=(4, 8, 16))
    assert isinstance(lac, float)
    assert lac >= 0.0


# --- Phase 2: Multi-decoder + FD regression ---

def test_edge_decoder_forward():
    from fractal_swin_unet.models.multi_decoder import EdgeDecoder
    dec = EdgeDecoder(in_channels=32)
    x = torch.randn(2, 32, 16, 16)
    out = dec(x)
    assert out.shape == (2, 1, 16, 16)


def test_skeleton_decoder_forward():
    from fractal_swin_unet.models.multi_decoder import SkeletonDecoder
    dec = SkeletonDecoder(in_channels=32)
    x = torch.randn(2, 32, 16, 16)
    out = dec(x)
    assert out.shape == (2, 1, 16, 16)


def test_fd_regression_head():
    from fractal_swin_unet.models.multi_decoder import FDRegressionHead
    head = FDRegressionHead(in_channels=128)
    x = torch.randn(2, 128, 8, 8)
    out = head(x)
    assert out.shape == (2, 1)


def test_extract_edge_gt():
    from fractal_swin_unet.models.multi_decoder import extract_edge_gt
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 1.0  # square vessel
    edge = extract_edge_gt(mask)
    assert edge.shape == mask.shape
    # Edge should be non-zero only at boundaries
    assert edge[:, :, 15, 15].item() == 0.0  # interior = 0
    assert edge[:, :, 10, 15].item() > 0.0  # boundary > 0


def test_extract_skeleton_gt():
    from fractal_swin_unet.models.multi_decoder import extract_skeleton_gt
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 14:18, 5:27] = 1.0  # horizontal vessel
    skel = extract_skeleton_gt(mask, iters=5)
    assert skel.shape == mask.shape
    assert skel.min() >= 0.0
    assert skel.max() <= 1.0


def test_fractal_prior_multi_decoder_output():
    """Model with multi-decoder returns dict with all heads."""
    from fractal_swin_unet.models import FractalPriorSwinUNet
    model = FractalPriorSwinUNet(
        in_channels=3, embed_dim=16, depths=(1, 1, 1),
        enable_fractal_gate=False, use_multi_decoder=True,
        use_fd_regression=True,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (1, 1, 64, 64)
    assert "edge_logits" in out
    assert out["edge_logits"].shape == (1, 1, 64, 64)
    assert "skeleton_logits" in out
    assert out["skeleton_logits"].shape == (1, 1, 64, 64)
    assert "fd_pred" in out
    assert out["fd_pred"].shape == (1, 1)


def test_fractal_prior_multi_channel_prior():
    """Model accepts multi-channel prior for fusion."""
    from fractal_swin_unet.models import FractalPriorSwinUNet
    model = FractalPriorSwinUNet(
        in_channels=3, embed_dim=16, depths=(1, 1, 1),
        enable_fractal_gate=True, enable_prior_fusion=True,
        prior_channels=4,
    )
    x = torch.randn(1, 3, 64, 64)
    lfd = torch.randn(1, 4, 64, 64)  # 4-channel prior
    out = model(x, lfd_map=lfd)
    assert out.shape == (1, 1, 64, 64)


def test_fractal_prior_backward_compat():
    """Default model (no new features) still returns plain Tensor."""
    from fractal_swin_unet.models import FractalPriorSwinUNet
    model = FractalPriorSwinUNet(
        in_channels=3, embed_dim=16, depths=(1, 1, 1),
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 64, 64)


# --- Phase 3: New losses ---

def test_cbdice_loss():
    from fractal_swin_unet.losses import cbdice_loss
    probs = torch.sigmoid(torch.randn(2, 1, 32, 32))
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = cbdice_loss(probs, targets, iters=3)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_skeleton_recall_loss():
    from fractal_swin_unet.losses import skeleton_recall_loss
    probs = torch.sigmoid(torch.randn(2, 1, 32, 32))
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = skeleton_recall_loss(probs, targets, iters=3)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_fractal_weighted_bce():
    from fractal_swin_unet.losses import fractal_weighted_bce
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    fw = torch.rand(2, 1, 32, 32)
    loss = fractal_weighted_bce(logits, targets, fw, alpha=1.0)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    # With higher fractal weight, loss should be higher
    loss_high = fractal_weighted_bce(logits, targets, torch.ones_like(fw), alpha=2.0)
    loss_low = fractal_weighted_bce(logits, targets, torch.zeros_like(fw), alpha=2.0)
    assert loss_high.item() >= loss_low.item() - 1e-6


def test_composite_loss_new_losses_enabled():
    """CompositeLoss works with all new losses enabled."""
    from fractal_swin_unet.losses import CompositeLoss
    cfg = {
        "from_logits": True,
        "dice": {"enabled": True, "weight": 1.0},
        "bce": {"enabled": True, "weight": 1.0},
        "cbdice": {"enabled": True, "weight": 0.15, "iter": 3},
        "skeleton_recall": {"enabled": True, "weight": 0.1, "iter": 3},
        "fractal_bce": {"enabled": True, "weight": 0.5, "alpha": 1.0},
    }
    loss_fn = CompositeLoss(cfg)
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    fw = torch.rand(2, 1, 32, 32)
    loss, breakdown = loss_fn(logits, targets, fractal_weight=fw)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert breakdown["loss_cbdice"] >= 0.0
    assert breakdown["loss_skel_recall"] >= 0.0
    assert breakdown["loss_fractal_bce"] >= 0.0


# --- Phase 6: Frequency-Geometric Decomposition (FGOS-Net 2026) ---

def test_haar_dwt_roundtrip():
    """Haar DWT → IDWT should reconstruct the input."""
    from fractal_swin_unet.models.freq_geom import haar_dwt_2d, haar_idwt_2d
    x = torch.randn(2, 16, 32, 32)
    LL, LH, HL, HH = haar_dwt_2d(x)
    assert LL.shape == (2, 16, 16, 16)
    recon = haar_idwt_2d(LL, LH, HL, HH)
    assert recon.shape == x.shape
    assert torch.allclose(recon, x, atol=1e-5)


def test_freq_geom_decomposition_shape():
    from fractal_swin_unet.models.freq_geom import FreqGeomDecomposition
    module = FreqGeomDecomposition(channels=32)
    x = torch.randn(2, 32, 64, 64)
    out = module(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_freq_geom_odd_dimensions():
    """FreqGeomDecomposition handles odd spatial dimensions."""
    from fractal_swin_unet.models.freq_geom import FreqGeomDecomposition
    module = FreqGeomDecomposition(channels=16)
    x = torch.randn(1, 16, 33, 47)  # odd H and W
    out = module(x)
    assert out.shape == x.shape


def test_model_with_freq_geom():
    """Full model with freq_geom enabled produces correct output."""
    from fractal_swin_unet.models import FractalPriorSwinUNet
    model = FractalPriorSwinUNet(
        in_channels=3, embed_dim=16, depths=(1, 1, 1),
        enable_fractal_gate=False, use_freq_geom=True,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 64, 64)
    assert not torch.isnan(out).any()


def test_model_all_v9_features():
    """Full V9 model with all features: multi-FFM, multi-decoder, FD head, DWT."""
    from fractal_swin_unet.models import FractalPriorSwinUNet
    model = FractalPriorSwinUNet(
        in_channels=3, embed_dim=16, depths=(1, 1, 1),
        enable_fractal_gate=True, enable_prior_fusion=True,
        prior_channels=4, use_multi_decoder=True,
        use_fd_regression=True, use_freq_geom=True,
    )
    x = torch.randn(1, 3, 64, 64)
    lfd = torch.randn(1, 4, 64, 64)
    out = model(x, lfd_map=lfd)
    assert isinstance(out, dict)
    assert out["logits"].shape == (1, 1, 64, 64)
    assert "edge_logits" in out
    assert "skeleton_logits" in out
    assert "fd_pred" in out
    assert not torch.isnan(out["logits"]).any()
