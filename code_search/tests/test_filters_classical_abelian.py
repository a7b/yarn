"""Tests for ``search/filters/classical/abelian/ring_distance_bound.py``.

Abelian-only filter; uses the permanent over F₂[G].
"""

import pytest

from search.filters.classical.abelian.ring_distance_bound import (
    ring_distance_bound,
)

pytestmark = pytest.mark.gap


@pytest.fixture(scope="module")
def gd_c4():
    from core.group import GroupData
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


class TestRingDistanceBound:
    def test_1x2_returns_sum_of_supports(self, gd_c4):
        # J=1, I=2; target_size = 2. The only subset S = {0, 1} contributes
        # |A[0][0]| + |A[0][1]|.
        A = [[(0, 1), (2, 3)]]   # |.|=2 + |.|=2 = 4
        assert ring_distance_bound(A, gd_c4) == 4

    def test_1x3_returns_min_pairwise_sum(self, gd_c4):
        # J=1, I=3; for each S of size 2, total = sum of weights. min over the 3 choose 2 = 3 subsets.
        A = [[(0,), (1, 2), (3,)]]   # weights 1, 2, 1
        # subsets: {0,1}→3, {0,2}→2, {1,2}→3 → min=2
        assert ring_distance_bound(A, gd_c4) == 2

    def test_target_size_exceeds_returns_inf(self, gd_c4):
        # J=1, I=1: target_size = 2 > I = 1 → inf.
        A = [[(0, 1)]]
        assert ring_distance_bound(A, gd_c4) == float("inf")

    def test_raises_on_non_abelian(self, gd_s3):
        A = [[(0, 1), (2,)]]
        with pytest.raises(ValueError, match="abelian"):
            ring_distance_bound(A, gd_s3)
