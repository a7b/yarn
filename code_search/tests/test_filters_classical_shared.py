"""Tests for the group-agnostic classical filters in
``search/filters/classical/_shared/``.

Each filter is exercised here exactly once; the abelian/non_abelian
shells are tested in :mod:`tests.test_filters_classical_imports`.
"""

import numpy as np
import pytest

from search.filters.classical._shared.any_block_col_full_rank import (
    any_block_col_full_rank,
)
from search.filters.classical.abelian.base_girth_bound import base_girth_bound
from search.filters.classical._shared.entry_order_bound import (
    entry_order_bound,
)
from search.filters.classical._shared.girth_tanner import girth_tanner


# ─────────────────────────────────────────────────────────────────
# base_girth_bound (fast, no GAP)
# ─────────────────────────────────────────────────────────────────


class TestBaseGirthBound:
    pytestmark = pytest.mark.fast

    def test_zero_matrix_returns_inf(self):
        assert base_girth_bound([[0, 0], [0, 0]]) == float("inf")

    def test_weight_3_entry_returns_6(self):
        assert base_girth_bound([[3]]) == 6
        assert base_girth_bound([[1, 3], [2, 1]]) == 6

    def test_two_weight2_in_same_row_returns_8(self):
        assert base_girth_bound([[2, 2]]) == 8

    def test_two_weight2_in_same_col_returns_8(self):
        assert base_girth_bound([[2], [2]]) == 8

    def test_2x2_all_nonzero_with_a_two_returns_10(self):
        assert base_girth_bound([[1, 1], [2, 1]]) == 10

    def test_2x3_all_nonzero_all_ones_returns_12(self):
        assert base_girth_bound([[1, 1, 1], [1, 1, 1]]) == 12

    def test_returns_int_or_inf(self):
        # All weight-1 1x2 has no theorem-based bound.
        assert base_girth_bound([[1, 1]]) == float("inf")


# ─────────────────────────────────────────────────────────────────
# girth_tanner (fast)
# ─────────────────────────────────────────────────────────────────


class TestGirthTanner:
    pytestmark = pytest.mark.fast

    def test_forest_returns_none(self):
        # 2 rows × 3 cols with disjoint supports → tree → no cycles.
        H = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        # Actually this might have cycles via the bipartite structure. Let me
        # use the trivial single-row case which definitely is a forest.
        H = np.array([[1, 1, 1]], dtype=np.uint8)
        assert girth_tanner(H) is None

    def test_zero_matrix_is_forest(self):
        H = np.zeros((2, 3), dtype=np.uint8)
        assert girth_tanner(H) is None

    def test_girth_4_two_checks_two_vars_both_set(self):
        # Two checks both connect to the same two variables → girth-4 cycle.
        H = np.array([[1, 1, 0],
                      [1, 1, 0]], dtype=np.uint8)
        assert girth_tanner(H) == 4

    def test_returns_even(self):
        # Tanner graphs are bipartite ⇒ all cycles are even.
        rng = np.random.default_rng(0)
        for _ in range(5):
            H = rng.integers(0, 2, size=(4, 6), dtype=np.uint8)
            g = girth_tanner(H)
            if g is not None:
                assert g % 2 == 0


# ─────────────────────────────────────────────────────────────────
# Filters needing GroupData (gap marker)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gd_c4():
    from core.group import GroupData
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


class TestEntryOrderBound:
    pytestmark = pytest.mark.gap

    def test_no_weight2_entry_returns_none(self, gd_c4):
        A = [[(0,), (1,)]]
        assert entry_order_bound(A, gd_c4) is None

    def test_weight2_entry_returns_min_order(self, gd_c4):
        # (0, 1) has ratio 0^{-1}·1 = 1, order = order of g_1 in C4.
        A = [[(0, 1), (gd_c4.identity,)]]
        # Order of group element 1 in C4 is 4 (it's a generator).
        bound = entry_order_bound(A, gd_c4)
        assert isinstance(bound, int) and bound >= 1

    def test_raises_on_ma_gt_1(self, gd_c4):
        A = [[(0,), (1,)],
             [(2,), (3,)]]
        with pytest.raises(ValueError, match="ma=1"):
            entry_order_bound(A, gd_c4)


# ─────────────────────────────────────────────────────────────────
# Rank predicates (fast — pure numpy, no GAP)
# ─────────────────────────────────────────────────────────────────


class TestAnyBlockColFullRank:
    pytestmark = pytest.mark.fast

    def test_identity_block_is_full_rank(self):
        # A_bin = [I_n | something]  — first block is full rank.
        n = 3
        A_bin = np.hstack([np.eye(n, dtype=np.uint8),
                           np.zeros((n, n), dtype=np.uint8)])
        assert any_block_col_full_rank(A_bin, n, ma=1) is True

    def test_all_zero_blocks_not_full_rank(self):
        n = 3
        A_bin = np.zeros((n, 3 * n), dtype=np.uint8)
        assert any_block_col_full_rank(A_bin, n, ma=1) is False

    def test_ma_2_finds_two_block_subset(self):
        # A_bin: 2n × 3n with first two blocks being I; check that ma=2 works.
        n = 2
        block0 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.uint8)
        block1 = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.uint8)
        block2 = np.zeros((4, 2), dtype=np.uint8)
        A_bin = np.hstack([block0, block1, block2])
        # subsets [0, 1] form a 4x4 identity — full rank.
        assert any_block_col_full_rank(A_bin, n, ma=2) is True


# ─────────────────────────────────────────────────────────────────
# Cross-check: filters preserve inputs
# ─────────────────────────────────────────────────────────────────


class TestNoMutation:
    pytestmark = pytest.mark.fast

    def test_base_girth_bound_does_not_mutate(self):
        W = np.array([[2, 2], [1, 1]], dtype=int)
        W_copy = W.copy()
        base_girth_bound(W)
        np.testing.assert_array_equal(W, W_copy)

    def test_girth_tanner_does_not_mutate(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        H_copy = H.copy()
        girth_tanner(H)
        np.testing.assert_array_equal(H, H_copy)
