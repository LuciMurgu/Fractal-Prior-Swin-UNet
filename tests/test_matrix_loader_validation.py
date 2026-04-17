from pathlib import Path

import pytest

from fractal_swin_unet.exp.matrix import load_matrix


def test_matrix_loader_validation(tmp_path: Path) -> None:
    valid = tmp_path / "matrix.yaml"
    valid.write_text(
        """
matrix:
  name: test
experiments:
  - name: exp1
    config: configs/smoke.yaml
""",
        encoding="utf-8",
    )
    assert load_matrix(valid)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("matrix: {}\nexperiments: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_matrix(invalid)
