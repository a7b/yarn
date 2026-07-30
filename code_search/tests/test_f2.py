"""Tests for core.f2 (Type 1: implementation-aware)."""

import numpy as np
import pytest

from core.f2 import f2_null_space, f2_rank, f2_reduce, f2_rref, f2_solve


pytestmark = pytest.mark.fast


class TestF2RankSmall:
    def test_zero_matrix(self):
        assert f2_rank(np.zeros((3, 5), dtype=np.uint8)) == 0

    def test_identity(self):
        assert f2_rank(np.eye(5, dtype=np.uint8)) == 5

    def test_single_nonzero_row(self):
        assert f2_rank(np.array([[1, 0, 1]], dtype=np.uint8)) == 1

    def test_single_zero_row(self):
        assert f2_rank(np.array([[0, 0, 0]], dtype=np.uint8)) == 0

    def test_single_column(self):
        assert f2_rank(np.array([[1], [0], [1]], dtype=np.uint8)) == 1

    def test_rectangular_full_row_rank(self):
        assert f2_rank(np.eye(3, 5, dtype=np.uint8)) == 3

    def test_rectangular_full_col_rank(self):
        assert f2_rank(np.eye(5, 3, dtype=np.uint8)) == 3


class TestF2RankGF2Specific:
    def test_duplicate_rows_drop_to_unique_count(self):
        H = np.vstack([np.eye(3, dtype=np.uint8)] * 2)
        assert f2_rank(H) == 3

    def test_three_rows_summing_to_zero_have_rank_two(self):
        # Over Q: rows are independent (rank 3). Over GF(2): row1+row2+row3 = 0.
        H = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0]], dtype=np.uint8)
        assert f2_rank(H) == 2

    def test_row_with_pivot_at_each_column(self):
        H = np.array([[1, 1, 1],
                      [0, 1, 1],
                      [0, 0, 1]], dtype=np.uint8)
        assert f2_rank(H) == 3


class TestF2RankInputHandling:
    def test_input_not_modified(self):
        H = np.eye(4, dtype=np.uint8)
        H_copy = H.copy()
        _ = f2_rank(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_accepts_bool_dtype(self):
        H = np.eye(3, dtype=bool)
        assert f2_rank(H) == 3

    def test_accepts_int64_dtype(self):
        H = np.eye(3, dtype=np.int64)
        assert f2_rank(H) == 3

    def test_nonzero_nonbinary_integers_treated_as_one(self):
        # Convention change vs new/: non-binary integers used to break the algorithm
        # (uint8 cast left them as e.g. 2, which then XOR'd incorrectly). Now they
        # are normalized via (H != 0) so this matrix has rank 2.
        H = np.array([[2, 0],
                      [0, 3]], dtype=np.int64)
        assert f2_rank(H) == 2

    def test_negative_integers_treated_as_one(self):
        H = np.array([[-1, 0],
                      [0, -7]], dtype=np.int64)
        assert f2_rank(H) == 2


class TestF2RankInvariants:
    def test_rank_le_min_shape(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            m = int(rng.integers(1, 20))
            n = int(rng.integers(1, 20))
            H = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
            assert f2_rank(H) <= min(m, n)

    def test_rank_invariant_under_transpose(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            H = rng.integers(0, 2, size=(7, 11), dtype=np.uint8)
            assert f2_rank(H) == f2_rank(H.T)

    def test_appending_row_cannot_decrease_rank(self):
        rng = np.random.default_rng(13)
        for _ in range(10):
            H = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
            new_row = rng.integers(0, 2, size=(1, 8), dtype=np.uint8)
            r0 = f2_rank(H)
            r1 = f2_rank(np.vstack([H, new_row]))
            assert r1 >= r0


class TestF2RankEdgeCases:
    def test_zero_rows(self):
        assert f2_rank(np.zeros((0, 5), dtype=np.uint8)) == 0

    def test_zero_cols(self):
        assert f2_rank(np.zeros((5, 0), dtype=np.uint8)) == 0

    def test_single_cell_one(self):
        assert f2_rank(np.array([[1]], dtype=np.uint8)) == 1

    def test_single_cell_zero(self):
        assert f2_rank(np.array([[0]], dtype=np.uint8)) == 0


# ─────────────────────────────────────────────────────────────────
# f2_rref
# ─────────────────────────────────────────────────────────────────


class TestF2Rref:
    def test_identity_stays_identity(self):
        R, p = f2_rref(np.eye(4, dtype=np.uint8))
        np.testing.assert_array_equal(R, np.eye(4, dtype=np.uint8))
        assert p == [0, 1, 2, 3]

    def test_zero_matrix_no_pivots(self):
        R, p = f2_rref(np.zeros((3, 5), dtype=np.uint8))
        np.testing.assert_array_equal(R, np.zeros((3, 5), dtype=np.uint8))
        assert p == []

    def test_rank_equals_pivot_count(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            H = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
            _, p = f2_rref(H)
            assert len(p) == f2_rank(H)

    def test_input_not_modified(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        H_copy = H.copy()
        f2_rref(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_rref_specific_example(self):
        H = np.array([[1, 1, 0, 1],
                      [1, 0, 1, 1],
                      [0, 1, 1, 0]], dtype=np.uint8)
        R, p = f2_rref(H)
        # Reduced form: rows 1 and 2 sum to row 0 ⇒ rank 2.
        assert len(p) == 2
        # Verify each pivot column has a unique 1 in the pivot row.
        for i, c in enumerate(p):
            assert R[i, c] == 1
            for j in range(len(p)):
                if i != j:
                    assert R[j, c] == 0


# ─────────────────────────────────────────────────────────────────
# f2_solve
# ─────────────────────────────────────────────────────────────────


class TestF2Solve:
    def test_identity_solve(self):
        A = np.eye(3, dtype=np.uint8)
        B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        X = f2_solve(A, B)
        np.testing.assert_array_equal(X, B)

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        # Generate a random invertible A by tries.
        for _ in range(20):
            A = rng.integers(0, 2, size=(4, 4), dtype=np.uint8)
            if f2_rank(A) == 4:
                break
        B = rng.integers(0, 2, size=(4, 2), dtype=np.uint8)
        X = f2_solve(A, B)
        np.testing.assert_array_equal((A @ X) % 2, B)

    def test_non_square_raises(self):
        A = np.eye(3, 4, dtype=np.uint8)
        B = np.zeros((3, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="square"):
            f2_solve(A, B)

    def test_singular_raises(self):
        A = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        B = np.array([[1], [0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="not full rank"):
            f2_solve(A, B)

    def test_b_shape_mismatch_raises(self):
        A = np.eye(3, dtype=np.uint8)
        B = np.zeros((2, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            f2_solve(A, B)


# ─────────────────────────────────────────────────────────────────
# f2_reduce
# ─────────────────────────────────────────────────────────────────


class TestF2Reduce:
    def test_reduce_against_identity(self):
        basis = np.eye(3, dtype=np.uint8)
        v = np.array([1, 1, 1], dtype=np.uint8)
        # Reducing v against the full identity basis zeros every pivot col.
        r = f2_reduce(v, basis, [0, 1, 2])
        np.testing.assert_array_equal(r, np.zeros(3, dtype=np.uint8))

    def test_reduce_keeps_non_pivot_bits(self):
        # basis pivots only column 0; bit at column 2 is untouched.
        basis = np.array([[1, 0, 0]], dtype=np.uint8)
        v = np.array([1, 0, 1], dtype=np.uint8)
        r = f2_reduce(v, basis, [0])
        np.testing.assert_array_equal(r, np.array([0, 0, 1], dtype=np.uint8))

    def test_input_not_modified(self):
        basis = np.eye(3, dtype=np.uint8)
        v = np.array([1, 0, 1], dtype=np.uint8)
        v_copy = v.copy()
        f2_reduce(v, basis, [0, 1, 2])
        np.testing.assert_array_equal(v, v_copy)


# ─────────────────────────────────────────────────────────────────
# f2_null_space
# ─────────────────────────────────────────────────────────────────


class TestF2NullSpace:
    def test_zero_matrix_full_null(self):
        # ker(0) = whole space.
        N = f2_null_space(np.zeros((2, 5), dtype=np.uint8))
        assert N.shape == (5, 5)
        # ker rows are linearly independent.
        assert f2_rank(N) == 5

    def test_identity_null_empty(self):
        # ker(I_n) = {0}.
        N = f2_null_space(np.eye(4, dtype=np.uint8))
        assert N.shape == (0, 4)

    def test_null_vectors_satisfy_M_v_eq_zero(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            M = rng.integers(0, 2, size=(4, 7), dtype=np.uint8)
            N = f2_null_space(M)
            if N.shape[0] > 0:
                np.testing.assert_array_equal(
                    (M @ N.T) % 2,
                    np.zeros((4, N.shape[0]), dtype=np.uint8),
                )

    def test_dim_ker_plus_rank_equals_ncols(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            M = rng.integers(0, 2, size=(4, 7), dtype=np.uint8)
            N = f2_null_space(M)
            assert N.shape[0] + f2_rank(M) == M.shape[1]

    def test_null_vectors_independent(self):
        # The kernel basis returned should itself be full-rank in its row dim.
        rng = np.random.default_rng(3)
        M = rng.integers(0, 2, size=(3, 8), dtype=np.uint8)
        N = f2_null_space(M)
        if N.shape[0] > 0:
            assert f2_rank(N) == N.shape[0]
