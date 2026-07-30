"""Tests for the abelian path of ``run_classical`` (Chunk 2)."""

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
    WeightPatternConfig,
)
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    manifest_path,
)


def _cfg(tmp_path: Path,
         gap_expr: str = "CyclicGroup(4)",
         tag: str = "C4",
         entry_max: int = 1,
         num_weight_samples: int = 4,
         ring_samples_per_weight: int = 2,
         max_row_weight: int = 2,
         d_target: int = 2,
         num_trials: int = 20) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr=gap_expr, tag=tag),
        run_stages=["classical"],
        classical=ClassicalStageConfig(
            weight_pattern=WeightPatternConfig(
                entry_max=entry_max,
                num_weight_samples=num_weight_samples,
                ring_samples_per_weight=ring_samples_per_weight,
                max_row_weight=max_row_weight,
            ),
            sampling=SamplingConfig(
                total_samples=0,   # unused in abelian path
                seed=0,
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


class TestAbelianPath:
    pytestmark = pytest.mark.bposd

    def test_writes_only_to_classical_A(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path)
        out = run_classical(cfg)
        assert isinstance(out, dict)
        assert out["new_B"] == []   # B = A for abelian; no classical_B

        if out["new_A"]:
            sample = Path(out["new_A"][0])
            assert sample.exists()
            data = json.loads(sample.read_text())
            assert data["side"] == "A"
            assert data["group_tag"] == "C4"

            # No classical_B directory ever created for this run.
            assert not classical_B_dir(cfg).exists()

    def test_missing_weight_pattern_raises(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path)
        cfg.classical.weight_pattern = None
        with pytest.raises(ValueError, match="weight_pattern"):
            run_classical(cfg)


class TestAbelianZeroSamples:
    pytestmark = pytest.mark.gap

    def test_zero_weight_samples_no_files(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg(tmp_path, num_weight_samples=0)
        out = run_classical(cfg)
        assert out == {"new_A": [], "new_B": []}
        assert not manifest_path(cfg).exists()
