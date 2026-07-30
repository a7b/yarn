"""Tests for core.classical_code (Type 1: implementation-aware).

Covers only the classical-code-level pieces (no Hx / Hz yet):
- build_A_bin, build_B_bin (forward)
- A_from_A_bin, B_from_B_bin (inverse, with round-trip invariants)
- canonical_form_A / canonical_form_B (on ring matrix)
- canonical_form_A_bin / canonical_form_B_bin (chained via inverse)
"""

import numpy as np
import pytest

from core.classical_code import (
    A_from_A_bin,
    B_from_B_bin,
    build_A_bin,
    build_B_bin,
    canonical_form_A,
    canonical_form_A_bin,
    canonical_form_B,
    canonical_form_B_bin,
)
from core.f2 import f2_rank
from core.group import GroupData, canonicalize, left_rep, right_rep

pytestmark = pytest.mark.gap


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


# ─────────────────────────────────────────────────────────────────
# build_A_bin / build_B_bin
# ─────────────────────────────────────────────────────────────────


class TestBuildABin:
    def test_shape(self, gd_s3):
        A = [[(0, 1), (2,)], [(3,), (4, 5)]]
        A_bin = build_A_bin(A, gd_s3)
        assert A_bin.shape == (2 * gd_s3.n, 2 * gd_s3.n)

    def test_each_block_is_left_rep(self, gd_s3):
        A = [[(0, 1), (2,)]]
        A_bin = build_A_bin(A, gd_s3)
        n = gd_s3.n
        np.testing.assert_array_equal(A_bin[:n, :n], left_rep((0, 1), gd_s3))
        np.testing.assert_array_equal(A_bin[:n, n:], left_rep((2,), gd_s3))

    def test_empty_entry_produces_zero_block(self, gd_s3):
        A = [[(), (0,)]]
        A_bin = build_A_bin(A, gd_s3)
        n = gd_s3.n
        assert np.all(A_bin[:n, :n] == 0)

    def test_dtype_uint8(self, gd_s3):
        assert build_A_bin([[(0,)]], gd_s3).dtype == np.uint8


class TestBuildBBin:
    def test_shape(self, gd_s3):
        B = [[(0,), (1, 2)]]
        B_bin = build_B_bin(B, gd_s3)
        assert B_bin.shape == (1 * gd_s3.n, 2 * gd_s3.n)

    def test_each_block_is_right_rep(self, gd_s3):
        B = [[(0, 1), (2,)]]
        B_bin = build_B_bin(B, gd_s3)
        n = gd_s3.n
        np.testing.assert_array_equal(B_bin[:n, :n], right_rep((0, 1), gd_s3))
        np.testing.assert_array_equal(B_bin[:n, n:], right_rep((2,), gd_s3))


# ─────────────────────────────────────────────────────────────────
# A_from_A_bin / B_from_B_bin — round-trip
# ─────────────────────────────────────────────────────────────────


class TestAFromABin:
    def test_round_trip_entries_canonical(self, gd_s3):
        A = [[(0, 1), (2,)], [(3,), (4, 5)]]
        A_bin = build_A_bin(A, gd_s3)
        A_rec = A_from_A_bin(A_bin, gd_s3, (2, 2))
        for ia in range(2):
            for ja in range(2):
                assert A_rec[ia][ja] == canonicalize(A[ia][ja])

    def test_round_trip_abelian(self, gd_c4):
        A = [[(0,), (1, 2), (3,)]]
        A_bin = build_A_bin(A, gd_c4)
        A_rec = A_from_A_bin(A_bin, gd_c4, (1, 3))
        for ja in range(3):
            assert A_rec[0][ja] == canonicalize(A[0][ja])

    def test_empty_entry_recovers_empty_tuple(self, gd_s3):
        A = [[(), (0,)]]
        A_bin = build_A_bin(A, gd_s3)
        A_rec = A_from_A_bin(A_bin, gd_s3, (1, 2))
        assert A_rec[0][0] == ()
        assert A_rec[0][1] == (0,)

    def test_build_recover_build_idempotent(self, gd_s3):
        A = [[(0, 1), (2, 3), ()], [(1, 4), (), (3, 5)]]
        A_bin = build_A_bin(A, gd_s3)
        A_rec = A_from_A_bin(A_bin, gd_s3, (2, 3))
        np.testing.assert_array_equal(A_bin, build_A_bin(A_rec, gd_s3))

    def test_noncanonical_input_recovers_canonical(self, gd_s3):
        # (1, 1, 2) ≡ (2,) under canonicalize. left_rep handles this via XOR,
        # so build_A_bin produces the same matrix; inverse recovers canonical.
        A = [[(1, 1, 2)]]
        A_bin = build_A_bin(A, gd_s3)
        A_rec = A_from_A_bin(A_bin, gd_s3, (1, 1))
        assert A_rec[0][0] == (2,)

    def test_shape_mismatch_raises(self, gd_s3):
        A_bin = np.zeros((2 * gd_s3.n, 2 * gd_s3.n), dtype=np.uint8)
        with pytest.raises(ValueError):
            A_from_A_bin(A_bin, gd_s3, (1, 2))

    def test_verify_passes_on_valid_input(self, gd_s3):
        # The always-on rebuild check must not false-positive on a valid lift.
        A = [[(0, 1), (2,)], [(3,), (4, 5)]]
        A_bin = build_A_bin(A, gd_s3)
        A_from_A_bin(A_bin, gd_s3, (2, 2))

    def test_verify_raises_on_non_lp_input(self, gd_s3):
        # A random binary matrix is almost certainly not a valid L-lift —
        # the always-on rebuild check must reject it.
        rng = np.random.default_rng(42)
        A_bin_random = rng.integers(0, 2, size=(gd_s3.n, gd_s3.n),
                                    dtype=np.uint8)
        with pytest.raises(ValueError, match="not a valid left-rep lift"):
            A_from_A_bin(A_bin_random, gd_s3, (1, 1))


class TestBFromBBin:
    def test_round_trip_entries_canonical(self, gd_s3):
        B = [[(0, 1), (2,)], [(3,), (4, 5)]]
        B_bin = build_B_bin(B, gd_s3)
        B_rec = B_from_B_bin(B_bin, gd_s3, (2, 2))
        for ib in range(2):
            for jb in range(2):
                assert B_rec[ib][jb] == canonicalize(B[ib][jb])

    def test_build_recover_build_idempotent(self, gd_s3):
        B = [[(0, 2), (1, 3), ()], [(4,), (5,), (0, 1)]]
        B_bin = build_B_bin(B, gd_s3)
        B_rec = B_from_B_bin(B_bin, gd_s3, (2, 3))
        np.testing.assert_array_equal(B_bin, build_B_bin(B_rec, gd_s3))

    def test_shape_mismatch_raises(self, gd_s3):
        B_bin = np.zeros((2 * gd_s3.n, 2 * gd_s3.n), dtype=np.uint8)
        with pytest.raises(ValueError):
            B_from_B_bin(B_bin, gd_s3, (1, 2))

    def test_verify_raises_on_non_lp_input(self, gd_s3):
        rng = np.random.default_rng(7)
        B_bin_random = rng.integers(0, 2, size=(gd_s3.n, gd_s3.n),
                                    dtype=np.uint8)
        with pytest.raises(ValueError, match="not a valid right-rep lift"):
            B_from_B_bin(B_bin_random, gd_s3, (1, 1))


# ─────────────────────────────────────────────────────────────────
# canonical_form_A / canonical_form_B
# ─────────────────────────────────────────────────────────────────


class TestCanonicalFormA:
    def test_full_rank_subset_lands_at_end(self, gd_s3):
        # A[0][0] = (identity,), so L[A[0][0]] = I_n, full rank by itself
        # (since ma=1). Permutation should move that block-col to the end.
        A = [[(gd_s3.identity,), (1, 2)]]
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        n = gd_s3.n
        ma = 1
        last_blocks = A_bin_can[:, -ma * n:]
        assert f2_rank(last_blocks) == ma * n

    def test_perm_consistency_with_canonical_entries(self, gd_s3):
        A = [[(0, 1), (2,)]]
        A_can, _, perm, _ = canonical_form_A(A, gd_s3)
        for i in range(len(A)):
            for k in range(len(A[0])):
                assert A_can[i][k] == canonicalize(A[i][perm[k]])

    def test_A_bin_matches_build_of_A_canonical(self, gd_s3):
        A = [[(0, 1), (2,)], [(3,), (4, 5)]]
        A_can, A_bin_can, _, _ = canonical_form_A(A, gd_s3)
        np.testing.assert_array_equal(A_bin_can, build_A_bin(A_can, gd_s3))

    def test_no_full_rank_subset_returns_flag_false(self, gd_s3):
        # All-zero ring matrix has no invertible block-col.
        A = [[(), ()]]
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is False
        assert perm == [0, 1]
        assert A_can == [[(), ()]]

    def test_canonicalizes_entries_when_no_full_rank(self, gd_s3):
        # Non-canonical input; no invertible block-col either.
        # Both entries (1,1) and (2,2) canonicalize to ().
        A = [[(1, 1), (2, 2)]]
        A_can, _, _, found = canonical_form_A(A, gd_s3)
        assert found is False
        assert A_can == [[(), ()]]

    def test_canonicalizes_entries_when_full_rank_found(self, gd_s3):
        # Identity entry plus a non-canonical other entry.
        A = [[(gd_s3.identity,), (1, 1, 2)]]
        A_can, _, _, found = canonical_form_A(A, gd_s3)
        assert found is True
        # Every entry in A_can must be canonical (sorted, distinct).
        for entry in A_can[0]:
            assert list(entry) == sorted(set(entry))

    def test_last_blocks_are_invertible_for_1x2_simple(self, gd_s3):
        # When only one block-col is full rank, it should land at the end.
        A = [[(0,), (1, 2)]]  # (0,) is identity → L = I, full rank
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        n = gd_s3.n
        # Last block (column-wise) is full rank n.
        assert f2_rank(A_bin_can[:, -n:]) == n


class TestCanonicalFormB:
    def test_full_rank_subset_lands_at_end(self, gd_s3):
        B = [[(gd_s3.identity,), (1, 2)]]
        B_can, B_bin_can, perm, found = canonical_form_B(B, gd_s3)
        assert found is True
        n = gd_s3.n
        assert f2_rank(B_bin_can[:, -n:]) == n

    def test_B_bin_matches_build_of_B_canonical(self, gd_s3):
        B = [[(0, 1), (2,)], [(3,), (4, 5)]]
        B_can, B_bin_can, _, _ = canonical_form_B(B, gd_s3)
        np.testing.assert_array_equal(B_bin_can, build_B_bin(B_can, gd_s3))

    def test_no_full_rank_subset_flag_false(self, gd_s3):
        B = [[(), ()]]
        B_can, B_bin_can, perm, found = canonical_form_B(B, gd_s3)
        assert found is False
        assert perm == [0, 1]


# ─────────────────────────────────────────────────────────────────
# canonical_form_A_bin / canonical_form_B_bin
# ─────────────────────────────────────────────────────────────────


class TestCanonicalFormFromBin:
    def test_A_bin_chain_matches_A_path(self, gd_s3):
        A = [[(gd_s3.identity,), (1, 2)]]
        A_bin = build_A_bin(A, gd_s3)
        ring_path = canonical_form_A(A, gd_s3)
        bin_path = canonical_form_A_bin(A_bin, gd_s3, (1, 2))
        assert ring_path[0] == bin_path[0]
        np.testing.assert_array_equal(ring_path[1], bin_path[1])
        assert ring_path[2] == bin_path[2]
        assert ring_path[3] == bin_path[3]

    def test_B_bin_chain_matches_B_path(self, gd_s3):
        B = [[(gd_s3.identity,), (1, 2)]]
        B_bin = build_B_bin(B, gd_s3)
        ring_path = canonical_form_B(B, gd_s3)
        bin_path = canonical_form_B_bin(B_bin, gd_s3, (1, 2))
        assert ring_path[0] == bin_path[0]
        np.testing.assert_array_equal(ring_path[1], bin_path[1])
        assert ring_path[2] == bin_path[2]
        assert ring_path[3] == bin_path[3]

    def test_A_bin_chain_handles_already_canonical(self, gd_s3):
        # Input where A is already in canonical form: full-rank subset at end.
        A = [[(1, 2), (gd_s3.identity,)]]
        A_bin = build_A_bin(A, gd_s3)
        A_can, A_bin_can, perm, found = canonical_form_A_bin(A_bin, gd_s3, (1, 2))
        assert found is True
        # The lex-first invertible subset chooses block-col 1 (identity), which
        # is already at the end. perm should leave block-col 1 last.
        assert perm[-1] == 1
