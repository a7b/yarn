"""Tests for ``search/sampling/_shared/weight_matrix.py``."""

from itertools import islice

import numpy as np
import pytest

from search.sampling._shared.weight_matrix import (
    all_weight_patterns,
    random_weight_patterns,
)


pytestmark = pytest.mark.fast


class TestRandomWeightPatterns:
    def test_yields_up_to_num_samples(self):
        rng = np.random.default_rng(0)
        results = list(random_weight_patterns(
            (1, 2), entry_max=3, num_samples=5, rng=rng,
        ))
        assert len(results) <= 5

    def test_zero_samples_yields_nothing(self):
        rng = np.random.default_rng(0)
        results = list(random_weight_patterns(
            (1, 2), entry_max=3, num_samples=0, rng=rng,
        ))
        assert results == []

    def test_yields_correct_shape(self):
        rng = np.random.default_rng(0)
        for W in islice(random_weight_patterns(
            (2, 3), entry_max=2, num_samples=3, rng=rng,
        ), 3):
            assert W.shape == (2, 3)
            assert W.dtype.kind in "iu"

    def test_dedupe_yields_unique(self):
        # entry_max=0 → only one pattern possible (all zeros).
        # With dedupe=True, max one yield.
        rng = np.random.default_rng(0)
        results = list(random_weight_patterns(
            (1, 2), entry_max=0, num_samples=5, rng=rng,
            max_tries=100,
        ))
        assert len(results) == 1

    def test_max_row_weight_filter(self):
        rng = np.random.default_rng(0)
        for W in islice(random_weight_patterns(
            (1, 4), entry_max=3, num_samples=5, rng=rng,
            max_row_weight=2,
        ), 5):
            assert int(W.sum(axis=1).max()) <= 2

    def test_max_col_weight_filter(self):
        rng = np.random.default_rng(0)
        for W in islice(random_weight_patterns(
            (3, 2), entry_max=3, num_samples=5, rng=rng,
            max_col_weight=3,
        ), 5):
            assert int(W.sum(axis=0).max()) <= 3

    def test_negative_entry_max_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="entry_max"):
            list(random_weight_patterns(
                (1, 2), entry_max=-1, num_samples=5, rng=rng,
            ))

    def test_negative_num_samples_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="num_samples"):
            list(random_weight_patterns(
                (1, 2), entry_max=3, num_samples=-1, rng=rng,
            ))

    def test_entry_min_forces_floor(self):
        """entry_min=1 forbids zero entries — every cell ≥ 1."""
        rng = np.random.default_rng(0)
        for W in islice(random_weight_patterns(
            (2, 3), entry_max=2, num_samples=10, rng=rng,
            entry_min=1,
        ), 10):
            assert (W >= 1).all()
            assert (W <= 2).all()

    def test_entry_min_equals_entry_max_yields_constant(self):
        """entry_min == entry_max → only one possible matrix."""
        rng = np.random.default_rng(0)
        results = list(random_weight_patterns(
            (1, 2), entry_max=3, num_samples=5, rng=rng,
            entry_min=3, max_tries=100,
        ))
        # Only one pattern possible: [[3, 3]]. dedupe gives exactly 1 yield.
        assert len(results) == 1
        np.testing.assert_array_equal(results[0], np.array([[3, 3]]))

    def test_entry_min_gt_entry_max_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="entry_max must be ≥ entry_min"):
            list(random_weight_patterns(
                (1, 2), entry_max=1, num_samples=5, rng=rng,
                entry_min=3,
            ))

    def test_negative_entry_min_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="entry_min"):
            list(random_weight_patterns(
                (1, 2), entry_max=3, num_samples=5, rng=rng,
                entry_min=-1,
            ))


class TestAllWeightPatterns:
    def test_enumerates_all_for_tiny_space(self):
        # 1×2 with entry_max=1 → 4 patterns total: 00, 01, 10, 11.
        results = list(all_weight_patterns((1, 2), entry_max=1))
        assert len(results) == 4

    def test_filter_reduces_count(self):
        # 1×2, entry_max=2; max_row_weight=1 → only patterns with row sum ≤ 1.
        results = list(all_weight_patterns(
            (1, 2), entry_max=2, max_row_weight=1,
        ))
        # Valid: 00, 01, 10 (3 patterns; 11/02/20/12/21/22 all have row sum ≥ 2).
        assert len(results) == 3
        for W in results:
            assert int(W.sum()) <= 1

    def test_negative_entry_max_raises(self):
        # negative entry_max with default entry_min=0 → entry_max < entry_min.
        with pytest.raises(ValueError, match="entry_max"):
            list(all_weight_patterns((1, 2), entry_max=-1))

    def test_entry_min_in_all_patterns(self):
        # 1×1, [1..3] → 3 patterns total.
        results = list(all_weight_patterns(
            (1, 1), entry_max=3, entry_min=1,
        ))
        assert len(results) == 3
        for W in results:
            assert 1 <= int(W[0, 0]) <= 3


class TestBaseGirthBoundFilter:
    """min_base_girth_bound is abelian-only — but the function itself just
    delegates to base_girth_bound on the int matrix. No GAP needed."""

    def test_filter_rejects_weight3_entry(self):
        # For 1×1 patterns, only weight=3 has a finite bound (6); 0/1/2 give
        # inf (no row/col pair to trigger the bound). So min=8 rejects only
        # the weight-3 single entry.
        rng = np.random.default_rng(0)
        results = list(random_weight_patterns(
            (1, 1), entry_max=3, num_samples=100, rng=rng,
            min_base_girth_bound=8,
        ))
        for W in results:
            assert int(W[0, 0]) != 3
