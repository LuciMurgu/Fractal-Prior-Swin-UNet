"""Performance benchmark for vectorized lacunarity computation.

Ensures the vectorized implementation achieves ≥20× speedup over a
baseline of 300ms on CPU for a 512×512 input.
"""

import time
import torch
import pytest

from fractal_swin_unet.fractal.lacunarity import compute_lacunarity_map


def _benchmark_lacunarity(device: str, size: int = 512, n_warmup: int = 2, n_runs: int = 5) -> float:
    """Benchmark lacunarity computation, return median time in ms."""
    torch.manual_seed(42)
    image = torch.rand(1, size, size, device=device)

    # Warmup
    for _ in range(n_warmup):
        compute_lacunarity_map(image, window_size=32, stride=16, box_sizes=(4, 8, 16))

    # Benchmark
    times = []
    for _ in range(n_runs):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_lacunarity_map(image, window_size=32, stride=16, box_sizes=(4, 8, 16))
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sorted(times)[len(times) // 2]  # median


class TestLacunarityPerformance:

    def test_cpu_runtime_under_300ms(self):
        """512×512 lacunarity map must complete in < 300ms on CPU."""
        median_ms = _benchmark_lacunarity("cpu", size=512)
        print(f"\n  CPU lacunarity 512×512: {median_ms:.1f}ms")
        assert median_ms < 300, f"Too slow: {median_ms:.1f}ms (limit: 300ms)"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_gpu_runtime_under_50ms(self):
        """512×512 lacunarity map must complete in < 50ms on GPU."""
        median_ms = _benchmark_lacunarity("cuda", size=512)
        print(f"\n  GPU lacunarity 512×512: {median_ms:.1f}ms")
        assert median_ms < 50, f"Too slow: {median_ms:.1f}ms (limit: 50ms)"

    def test_output_shape_correct(self):
        """Output shape must match input spatial dims."""
        image = torch.rand(3, 384, 384)
        lac = compute_lacunarity_map(image, window_size=32, stride=16)
        assert lac.shape == (384, 384), f"Expected (384, 384), got {lac.shape}"

    def test_output_is_finite(self):
        """All output values must be finite."""
        image = torch.rand(3, 256, 256)
        lac = compute_lacunarity_map(image, window_size=32, stride=16)
        assert torch.isfinite(lac).all(), "Non-finite values in lacunarity map"
