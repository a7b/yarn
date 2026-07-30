"""Tests for ``search/sampling/_shared/full_rank_block_pool.py``.

Covers ``build_full_rank_block_pool`` and ``sample_A_from_pool``.
"""

import numpy as np
import pytest

from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool,
    sample_A_from_pool,
)

pytestmark = pytest.mark.gap


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="module")
def gd_c4():
    from core.group import GroupData
    return GroupData("CyclicGroup(4)")


# ─────────────────────────────────────────────────────────────────
# Pool builder
# ─────────────────────────────────────────────────────────────────


class TestBuildPool:
    def test_weight_1_always_full_rank(self, gd_s3):
        # Each L[g] is a permutation matrix → full rank. Pool quickly fills.
        pool = build_full_rank_block_pool(
            gd_s3, weight=1, max_pool_size=4, seed=0
        )
        assert len(pool) == 4
        # Each entry is a 1-tuple.
        for x in pool:
            assert isinstance(x, tuple) and len(x) == 1

    def test_every_pool_entry_has_full_rank_lift(self, gd_s3):
        from core.f2 import f2_rank
        from core.group import left_rep
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=8, max_tries=2000, seed=42
        )
        for x in pool:
            M = left_rep(x, gd_s3)
            assert f2_rank(M) == gd_s3.n, (
                f"entry {x} has rank {f2_rank(M)} != n={gd_s3.n}"
            )

    def test_max_pool_size_respected(self, gd_s3):
        pool = build_full_rank_block_pool(
            gd_s3, weight=2, max_pool_size=3, max_tries=2000, seed=42
        )
        assert len(pool) <= 3

    def test_seed_stable(self, gd_s3):
        a = build_full_rank_block_pool(
            gd_s3, weight=2, max_pool_size=5, max_tries=500, seed=7
        )
        b = build_full_rank_block_pool(
            gd_s3, weight=2, max_pool_size=5, max_tries=500, seed=7
        )
        assert a == b

    def test_different_seeds_diverge(self, gd_s3):
        a = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=1
        )
        b = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=2
        )
        # Almost certainly different orderings/contents.
        assert a != b or (len(a) == 0 and len(b) == 0)

    def test_entries_are_canonical(self, gd_c4):
        pool = build_full_rank_block_pool(
            gd_c4, weight=3, max_pool_size=4, max_tries=500, seed=11
        )
        for x in pool:
            assert list(x) == sorted(set(x))

    def test_ma_gt_1_raises(self, gd_s3):
        with pytest.raises(NotImplementedError, match="ma=1"):
            build_full_rank_block_pool(
                gd_s3, weight=5, max_pool_size=4, ma=2
            )

    def test_invalid_weight_raises(self, gd_s3):
        with pytest.raises(ValueError, match="weight"):
            build_full_rank_block_pool(
                gd_s3, weight=0, max_pool_size=4
            )
        with pytest.raises(ValueError, match="weight"):
            build_full_rank_block_pool(
                gd_s3, weight=gd_s3.n + 1, max_pool_size=4
            )

    def test_right_rep_option_gives_same_pool(self, gd_s3):
        """rank(L[x]) == rank(R[x]) for all x — same pool either way."""
        from core.group import right_rep
        a = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=99
        )
        b = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=99,
            rep_func=right_rep,
        )
        # Same RNG path → same sampled set → same acceptance (since rank
        # of L[x] == rank of R[x]).
        assert a == b


# ─────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────


class TestSampleAFromPool:
    def test_shape_1x2_anchor_at_last_position(self, gd_s3):
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=5, max_tries=2000, seed=0
        )
        assert len(pool) >= 1
        rng = np.random.default_rng(123)
        A = sample_A_from_pool(
            gd_s3, pool, shape_a=(1, 2), free_col_weight=2, rng=rng
        )
        assert len(A) == 1
        assert len(A[0]) == 2
        assert A[0][1] in pool                          # last col is anchor
        assert len(A[0][0]) == 2                        # free col weight = 2

    def test_full_rank_property_holds(self, gd_s3):
        """A_bin has at least one full-rank block-col (it's at na-1)."""
        from core.f2 import f2_rank
        from core.classical_code import build_A_bin
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=0
        )
        rng = np.random.default_rng(7)
        for _ in range(5):
            A = sample_A_from_pool(
                gd_s3, pool, shape_a=(1, 2), free_col_weight=2, rng=rng
            )
            A_bin = build_A_bin(A, gd_s3)
            n = gd_s3.n
            # Block-col 1 (the last one) is from the pool → full rank.
            block_last = A_bin[:, n:]
            assert f2_rank(block_last) == n

    def test_per_column_weights_list(self, gd_c4):
        pool = build_full_rank_block_pool(
            gd_c4, weight=3, max_pool_size=4, max_tries=500, seed=0
        )
        rng = np.random.default_rng(5)
        # shape (1, 3): two free cols, one anchor.
        A = sample_A_from_pool(
            gd_c4, pool, shape_a=(1, 3), free_col_weight=[1, 3], rng=rng
        )
        assert len(A[0]) == 3
        assert len(A[0][0]) == 1
        assert len(A[0][1]) == 3
        assert A[0][2] in pool

    def test_empty_pool_raises(self, gd_s3):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="empty"):
            sample_A_from_pool(
                gd_s3, [], shape_a=(1, 2), free_col_weight=2, rng=rng
            )

    def test_wrong_length_free_weights_raises(self, gd_s3):
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=2, max_tries=2000, seed=0
        )
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="length"):
            sample_A_from_pool(
                gd_s3, pool, shape_a=(1, 3),
                free_col_weight=[1],   # need length 2 (na - ma)
                rng=rng,
            )

    def test_ma_gt_1_raises(self, gd_s3):
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=2, max_tries=2000, seed=0
        )
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError, match="ma=1"):
            sample_A_from_pool(
                gd_s3, pool, shape_a=(2, 3), free_col_weight=2, rng=rng
            )

    def test_free_col_knobs_forwarded(self, gd_s3):
        """free_col_knobs reaches random_ring_element for the free cols."""
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=5, max_tries=2000, seed=0
        )
        rng = np.random.default_rng(0)
        # include_identity=False applied to the free col.
        for _ in range(10):
            A = sample_A_from_pool(
                gd_s3, pool, shape_a=(1, 2), free_col_weight=2, rng=rng,
                free_col_knobs={"include_identity": False},
            )
            if A is None:
                continue
            # Free col is A[0][0]. Anchor is A[0][1].
            assert gd_s3.identity not in A[0][0]

    def test_free_col_knobs_unknown_key_raises(self, gd_s3):
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=2, max_tries=2000, seed=0
        )
        with pytest.raises(ValueError, match="unknown key"):
            sample_A_from_pool(
                gd_s3, pool, shape_a=(1, 2), free_col_weight=2,
                rng=np.random.default_rng(0),
                free_col_knobs={"not_a_knob": True},
            )

    def test_anchor_blocks_canonical_basis_property(self, gd_s3):
        """The sampled A satisfies ``any_block_col_full_rank``."""
        from core.classical_code import build_A_bin
        from search.filters.classical._shared.any_block_col_full_rank import (
            any_block_col_full_rank,
        )
        pool = build_full_rank_block_pool(
            gd_s3, weight=5, max_pool_size=10, max_tries=2000, seed=42
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            A = sample_A_from_pool(
                gd_s3, pool, shape_a=(1, 2), free_col_weight=2, rng=rng
            )
            A_bin = build_A_bin(A, gd_s3)
            assert any_block_col_full_rank(A_bin, gd_s3.n, ma=1)
