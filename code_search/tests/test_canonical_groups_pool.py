"""T1: canonical-search foundations — group enumerator + brute full-rank pool.

Marked ``gap`` (needs gappy + GAP for ``GroupData`` and SmallGroup queries).
"""

from collections import Counter

import pytest

from core.group import GroupData, right_rep
from search.canonical.groups import enumerate_band, enumerate_orders, nonabelian_ids
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
    count_weight_elements,
    enumerate_weight_elements,
)

pytestmark = pytest.mark.gap


def test_d16_band_counts_match_gap():
    d16 = enumerate_band("d16")
    c = Counter(s.order for s in d16)
    assert c[84] == 13 and c[88] == 9 and c[90] == 8 and c[100] == 12
    assert 96 not in c                       # giant order deferred
    assert len(d16) == 50
    # ascending by (order, id)
    assert [(s.order, s.small_group_id) for s in d16] == sorted(
        (s.order, s.small_group_id) for s in d16
    )


def test_d16_band_includes_giant_when_requested():
    d16f = enumerate_band("d16", include_deferred=True)
    c = Counter(s.order for s in d16f)
    assert c[96] == 224                      # natural ascending position
    assert len(d16f) == 274


def test_nonabelian_ids_skip_abelian_orders():
    assert nonabelian_ids(85) == []          # 85 = 5*17, cyclic only
    assert nonabelian_ids(84)[:3] == [1, 2, 3]


def test_brute_pool_sg84_11_complete():
    gd = GroupData("SmallGroup(84,11)")
    pool = build_full_rank_block_pool_brute(gd, 3, force_identity=True)
    assert len(pool) == 507                  # measured ground truth
    assert all(p[0] == 0 for p in pool)      # identity forced into support
    assert all(list(p) == sorted(set(p)) for p in pool)  # canonical sorted tuples


def test_brute_pool_left_right_identical():
    gd = GroupData("SmallGroup(84,11)")
    left = set(build_full_rank_block_pool_brute(gd, 3, force_identity=True))
    right = set(build_full_rank_block_pool_brute(gd, 3, force_identity=True,
                                                 rep_func=right_rep))
    assert left == right                     # rank L[x] == rank R[x]


def test_enumerate_weight_elements_count_and_identity():
    gd = GroupData("SmallGroup(84,11)")
    got = list(enumerate_weight_elements(gd, 3, force_identity=True))
    assert len(got) == count_weight_elements(gd.n, 3, force_identity=True) == 3403
    assert all(e[0] == 0 for e in got)
    assert all(list(e) == sorted(e) for e in got)
    free = count_weight_elements(gd.n, 3, force_identity=False)
    assert free == 95284                     # C(84,3): the brute free-column space
