from pathlib import Path

import torch

from fractal_swin_unet.inference.tiling import tiled_predict_proba


def test_infer_writes_outputs_smoke(tmp_path: Path) -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)

    model = ConstantModel()
    image = torch.randn(1, 1, 40, 40)
    probs = tiled_predict_proba(model, image, patch_size=(16, 16), stride=(8, 8))

    out_dir = tmp_path / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"probs": probs.cpu()}, out_dir / "preds.pt")

    assert (out_dir / "preds.pt").exists()
