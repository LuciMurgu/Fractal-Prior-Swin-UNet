from pathlib import Path

import torch

from fractal_swin_unet.fractal.cache_disk import save_cached_map, load_cached_map, save_metadata, metadata_path


def test_disk_cache_roundtrip(tmp_path: Path) -> None:
    tensor = torch.rand(32, 32)
    cache_file = tmp_path / "map.pt"
    meta_file = tmp_path / "map.json"

    save_cached_map(cache_file, tensor, dtype="float16")
    save_metadata(meta_file, {"sample_id": "s1"})
    loaded = load_cached_map(cache_file)

    assert loaded.shape == tensor.shape
    assert torch.max(torch.abs(loaded - tensor)) < 1e-2
    assert meta_file.exists()
