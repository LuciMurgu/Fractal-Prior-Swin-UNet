import torch

from fractal_swin_unet.inference.tiling import tiled_predict_proba


class ConstantModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)


def test_tiled_inference_blending_uniformity() -> None:
    model = ConstantModel()
    image = torch.randn(1, 3, 64, 64)
    probs = tiled_predict_proba(
        model,
        image,
        patch_size=(32, 32),
        stride=(16, 16),
        blend="hann",
        pad_mode="constant",
        batch_tiles=4,
    )
    center = probs[:, :, 16:-16, 16:-16]
    assert torch.allclose(center, torch.full_like(center, 0.5), atol=1e-6)
