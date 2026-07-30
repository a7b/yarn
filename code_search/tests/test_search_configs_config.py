"""Tests for ``search/configs/config.py`` — dataclass shape + defaults."""

from pathlib import Path

import pytest

from search.configs.config import (
    BPOSDConfig,
    ClassicalDistanceConfig,
    ClassicalFiltersConfig,
    ClassicalStageConfig,
    GroupConfig,
    PairingStageConfig,
    PoolConfig,
    SamplingConfig,
    SearchConfig,
)


pytestmark = pytest.mark.fast


class TestGroupConfig:
    def test_required_gap_expr(self):
        g = GroupConfig(gap_expr="SymmetricGroup(3)")
        assert g.gap_expr == "SymmetricGroup(3)"
        assert g.tag is None


class TestSamplingConfig:
    def test_defaults(self):
        s = SamplingConfig(total_samples=10)
        assert s.total_samples == 10
        assert s.include_identity is True
        assert s.min_element_order == 1
        assert s.avoid_same_coset is False
        assert s.canonicalize is True


class TestClassicalFiltersConfig:
    def test_all_disabled_by_default(self):
        c = ClassicalFiltersConfig()
        assert c.min_base_girth_bound is None
        assert c.require_any_block_col_full_rank is False
        assert c.min_ring_distance_bound is None


class TestClassicalStageConfig:
    def test_required_fields(self):
        s = ClassicalStageConfig(
            weight_A=[[2, 2]],
            distance=ClassicalDistanceConfig(d_target=8, num_trials=100),
            sampling=SamplingConfig(total_samples=10),
        )
        assert s.weight_A == [[2, 2]]
        assert s.weight_B is None   # abelian default
        assert isinstance(s.filters, ClassicalFiltersConfig)
        assert isinstance(s.pool, PoolConfig)


class TestSearchConfig:
    def test_minimal(self):
        s = SearchConfig(
            shape=(1, 2),
            group=GroupConfig(gap_expr="SymmetricGroup(3)"),
            run_stages=["classical"],
            classical=ClassicalStageConfig(
                weight_A=[[2, 2]], weight_B=[[2, 2]],
                distance=ClassicalDistanceConfig(d_target=8, num_trials=100),
                sampling=SamplingConfig(total_samples=10),
            ),
        )
        assert s.shape == (1, 2)
        assert s.pairing is None
        assert s.results_dir == Path("search_results")
        assert s.verbose is False
