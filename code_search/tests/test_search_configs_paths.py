"""Tests for ``search/configs/paths.py``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    classical_filename,
    group_dir,
    manifest_path,
    quantum_dir,
    weight_tag,
)


pytestmark = pytest.mark.fast


class TestWeightTag:
    def test_1x2_list(self):
        assert weight_tag([[2, 2]]) == "2.2"

    def test_1x2_ndarray(self):
        assert weight_tag(np.array([[2, 2]])) == "2.2"

    def test_2x2(self):
        assert weight_tag([[2, 3], [3, 2]]) == "2.3-3.2"

    def test_two_digit(self):
        assert weight_tag([[10, 2]]) == "10.2"


@pytest.fixture
def fake_cfg(tmp_path):
    return SimpleNamespace(
        results_dir=tmp_path / "results",
        group=SimpleNamespace(tag="S3"),
        shape=(1, 2),
    )


class TestDirHelpers:
    def test_group_dir(self, fake_cfg):
        assert group_dir(fake_cfg) == fake_cfg.results_dir / "S3" / "1x2"

    def test_classical_A_dir(self, fake_cfg):
        assert classical_A_dir(fake_cfg) == fake_cfg.results_dir / "S3" / "1x2" / "classical_A"

    def test_classical_B_dir(self, fake_cfg):
        assert classical_B_dir(fake_cfg) == fake_cfg.results_dir / "S3" / "1x2" / "classical_B"

    def test_quantum_dir(self, fake_cfg):
        assert quantum_dir(fake_cfg) == fake_cfg.results_dir / "S3" / "1x2" / "quantum"

    def test_manifest_path(self, fake_cfg):
        assert manifest_path(fake_cfg) == fake_cfg.results_dir / "S3" / "1x2" / "manifest.json"


class TestClassicalFilename:
    def test_format(self):
        assert classical_filename(8, "2.2", 1000, 12345) == "d8_w2.2_t1000_12345.json"
