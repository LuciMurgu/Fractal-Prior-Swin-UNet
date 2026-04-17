from pathlib import Path
import tempfile

from fractal_swin_unet.exp.run_artifacts import (
    make_run_dir,
    write_code_hash,
    write_env,
    write_git_commit,
    write_manifest_and_hash,
    write_metrics,
    write_readme,
    write_resolved_config,
)


def test_run_artifacts_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = make_run_dir(base=tmpdir, run_id="test_run")
        config = {"seed": 123}
        samples = [{"id": "s1", "image_path": "x", "mask_path": "y", "split": "train"}]

        write_resolved_config(run_dir, config)
        write_env(run_dir)
        write_git_commit(run_dir)
        write_code_hash(run_dir)
        write_manifest_and_hash(run_dir, samples)
        write_metrics(run_dir, {"metric": 1.0})
        write_readme(run_dir, "python -m fractal_swin_unet.train --config configs/smoke.yaml")

        expected = [
            "resolved_config.yaml",
            "env.txt",
            "git_commit.txt",
            "code_hash.txt",
            "manifest_used.jsonl",
            "manifest_hash.txt",
            "metrics.json",
            "README_RUN.md",
        ]

        for name in expected:
            path = run_dir / name
            assert path.exists()
            assert path.read_text(encoding="utf-8").strip()
