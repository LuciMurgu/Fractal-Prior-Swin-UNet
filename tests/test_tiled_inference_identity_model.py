import torch

from fractal_swin_unet.inference.tiling import tiled_predict_proba


class IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_tiled_inference_identity_model() -> None:
    model = IdentityModel()
    image = torch.randn(1, 1, 45, 53)
    probs_full = torch.sigmoid(model(image))
    probs_tiled = tiled_predict_proba(
        model,
        image,
        patch_size=(16, 17),
        stride=(8, 9),
        blend="uniform",
        batch_tiles=4,
    )

    assert torch.max(torch.abs(probs_full - probs_tiled)) < 1e-5
