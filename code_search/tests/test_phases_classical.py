"""Tests for ``search/phases/classical.py``.

End-to-end small-scale sanity checks: a non-abelian search on S3 with a
small budget produces JSON files matching the expected layout, and the
abelian path raises until Chunk 2.
"""

import json
from pathlib import Path

import pytest

from search.configs.config import (
    ClassicalDistanceConfig,
    ClassicalFiltersConfig,
    ClassicalStageConfig,
    GroupConfig,
    PoolConfig,
    SamplingConfig,
    SearchConfig,
)
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    manifest_path,
)


def _cfg(tmp_path: Path,
         gap_expr: str = "SymmetricGroup(3)",
         tag: str = "S3",
         weight_A=None, weight_B=None,
         total_samples: int = 4,
         d_target: int = 2,
         num_trials: int = 20) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr=gap_expr, tag=tag),
        run_stages=["classical"],
        classical=ClassicalStageConfig(
            weight_A=weight_A or [[2, 2]],
            weight_B=weight_B or [[2, 2]],
            sampling=SamplingConfig(
                total_samples=total_samples, seed=0,
                include_identity=True, max_tries=200,
            ),
            distance=ClassicalDistanceConfig(
                d_target=d_target, num_trials=num_trials,
                n_workers=1, osd_order=0,
            ),
            filters=ClassicalFiltersConfig(),
            pool=PoolConfig(),
        ),
        results_dir=tmp_path / "search_results",
    )


class TestAbelianMissingWeightPattern:
    """Abelian path now lives in Chunk 2; ``_cfg`` builds a non-abelian-style
    config (weight_A/weight_B set, weight_pattern absent), so dispatching to
    the abelian branch raises a config error."""

    pytestmark = pytest.mark.gap

    def test_abelian_missing_weight_pattern_raises(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path, gap_expr="CyclicGroup(4)", tag="C4")
        with pytest.raises(ValueError, match="weight_pattern"):
            run_classical(cfg)


class TestNonAbelianEndToEnd:
    """Small S3 search, low budget."""

    pytestmark = pytest.mark.bposd

    def test_writes_files_and_manifest(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path)
        out = run_classical(cfg)
        assert isinstance(out, dict)
        assert "new_A" in out and "new_B" in out

        # If anything was saved, structural checks.
        if out["new_A"]:
            sample = Path(out["new_A"][0])
            assert sample.exists()
            data = json.loads(sample.read_text())
            for required in ("gap_expr", "group_tag", "side", "shape",
                             "weight_matrix", "matrix", "dist",
                             "dist_estimator", "timestamp"):
                assert required in data, f"missing {required}"
            assert data["side"] == "A"
            assert data["group_tag"] == "S3"
            assert data["shape"] == [1, 2]

            # Manifest exists and last entry references this file.
            mp = manifest_path(cfg)
            assert mp.exists()
            manifest = json.loads(mp.read_text())
            assert isinstance(manifest, list) and len(manifest) >= 1
            assert manifest[-1]["new_A"] == out["new_A"]
            assert manifest[-1]["new_B"] == out["new_B"]


class TestNonAbelianNoSamples:
    """Total samples = 0: nothing is saved, no manifest entry."""

    pytestmark = pytest.mark.gap

    def test_zero_samples_no_manifest(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path, total_samples=0)
        out = run_classical(cfg)
        assert out == {"new_A": [], "new_B": []}
        # The dirs exist (mkdir was called), but no JSONs.
        assert not list((classical_A_dir(cfg) / "w2.2").glob("*.json"))
        # Manifest NOT created when no new files.
        assert not manifest_path(cfg).exists()


class TestNonAbelianMissingWeightB:
    """Non-abelian needs weight_B; raises if absent."""

    pytestmark = pytest.mark.gap

    def test_missing_weight_B_raises(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path)
        cfg.classical.weight_B = None
        with pytest.raises(ValueError, match="weight_B"):
            run_classical(cfg)
