"""Tests for non-abelian-only classical filters in
``search/filters/classical/non_abelian/``.

Currently: ``abelianization_bound``.
"""

import pytest

from search.filters.classical.non_abelian.abelianization_bound import (
    abelianization_bound,
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


class TestAbelianizationBound:
    def test_non_abelian_uses_commutator(self, gd_s3):
        # |[S3, S3]| = 3 (A3); bound = total entries × 3.
        A = [[(0, 1), (2,)]]
        assert abelianization_bound(A, gd_s3) == 3 * 3

    def test_raises_on_abelian(self, gd_c4):
        A = [[(0, 1), (2,)]]
        with pytest.raises(ValueError, match="non-abelian"):
            abelianization_bound(A, gd_c4)

    def test_raises_on_ma_gt_1(self, gd_s3):
        A = [[(0,)], [(1,)]]
        with pytest.raises(ValueError, match="ma=1"):
            abelianization_bound(A, gd_s3)
