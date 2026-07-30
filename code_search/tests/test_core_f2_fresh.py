"""Fresh black-box / contract tests for core.f2 and core.f2_fast (Type 2).

Written from the documented contracts only. Every expected value is derived
independently of the implementation:

  - a brute-force rank oracle (span-set enumeration over Python ints:
    the row space of an m-row binary matrix has exactly 2^rank elements),
  - exhaustive kernel enumeration for small n,
  - mathematical invariants: rank-nullity, RREF uniqueness (characterized
    by structural properties + row-space equality), solution uniqueness of
    A X = B for invertible A, and nondegeneracy of the logical pairing
    Lx . Lz^T for a valid CSS logical basis.

core.f2_fast (bit-packed, used by the search screen) previously had no
dedicated tests; widths straddling the 64-bit word boundary get particular
attention because that is where bit-packing bugs live.
"""

import numpy as np
import pytest

from core.f2 import f2_null_space, f2_rank, f2_reduce, f2_rref, f2_solve
from core.f2_fast import f2_nullspace_fast, f2_rank_fast, screen_basis

pytestmark = pytest.mark.fast


# ─────────────────────────────────────────────────────────────────
# Independent reference helpers (implementation-free oracles)
# ─────────────────────────────────────────────────────────────────


def _row_to_int(row) -> int:
    """1-D 0/1 array -> Python int with bit j = column j."""
    out = 0
    for j, b in enumerate(np.asarray(row).ravel().tolist()):
        if b:
            out |= 1 << j
    return out


def _span_set(M) -> set:
    """All 2^m XOR combinations of the rows of M, as Python ints.

    |span| = 2^rank exactly — an elimination-free rank oracle. Only for
    small row counts (2^m sets).
    """
    span = {0}
    for row in (np.asarray(M) != 0).astype(np.uint8):
        r = _row_to_int(row)
        span |= {s ^ r for s in span}
    return span


def _bf_rank(M) -> int:
    return len(_span_set(M)).bit_length() - 1


def _bf_kernel_set(M) -> set:
    """Exhaustive right-kernel of M as a set of ints (bit j = coord j)."""
    M = (np.asarray(M) != 0).astype(np.int64)
    m, n = M.shape
    ker = set()
    for val in range(2 ** n):
        v = np.array([(val >> j) & 1 for j in range(n)], dtype=np.int64)
        if not np.any((M @ v) % 2):
            ker.add(val)
    return ker


def _mm2(A, B) -> np.ndarray:
    """Binary matrix product mod 2 with safe accumulation."""
    return ((np.asarray(A, dtype=np.int64) @ np.asarray(B, dtype=np.int64))
            % 2).astype(np.uint8)


def _random_invertible(rng, n) -> np.ndarray:
    """Random invertible n x n binary matrix (certified by the span oracle)."""
    while True:
        A = rng.integers(0, 2, size=(n, n)).astype(np.uint8)
        if _bf_rank(A) == n:
            return A


# ─────────────────────────────────────────────────────────────────
# f2_rank vs an elimination-free oracle
# ─────────────────────────────────────────────────────────────────


class TestF2RankOracle:
    def test_matches_bruteforce_span_random(self):
        rng = np.random.default_rng(101)
        for _ in range(25):
            m = int(rng.integers(1, 9))
            n = int(rng.integers(1, 13))
            M = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
            assert f2_rank(M) == _bf_rank(M)

    def test_duplicating_rows_or_cols_preserves_rank(self):
        rng = np.random.default_rng(102)
        for _ in range(10):
            M = rng.integers(0, 2, size=(4, 7)).astype(np.uint8)
            r = f2_rank(M)
            assert f2_rank(np.vstack([M, M])) == r
            assert f2_rank(np.hstack([M, M])) == r

    def test_row_and_column_permutation_invariance(self):
        rng = np.random.default_rng(103)
        for _ in range(10):
            M = rng.integers(0, 2, size=(5, 8)).astype(np.uint8)
            r = f2_rank(M)
            assert f2_rank(M[rng.permutation(5), :]) == r
            assert f2_rank(M[:, rng.permutation(8)]) == r

    def test_outer_product_of_nonzero_vectors_has_rank_one(self):
        # Over any field, rank(u v^T) = 1 when u, v are nonzero.
        rng = np.random.default_rng(104)
        for _ in range(10):
            u = rng.integers(0, 2, size=6).astype(np.uint8)
            v = rng.integers(0, 2, size=9).astype(np.uint8)
            if not u.any():
                u[0] = 1
            if not v.any():
                v[0] = 1
            assert f2_rank(np.outer(u, v) % 2) == 1

    def test_rank_subadditive_under_xor(self):
        # rank(M1 + M2) <= rank(M1) + rank(M2) (rows of the sum lie in the
        # sum of the row spaces).
        rng = np.random.default_rng(105)
        for _ in range(10):
            M1 = rng.integers(0, 2, size=(5, 9)).astype(np.uint8)
            M2 = rng.integers(0, 2, size=(5, 9)).astype(np.uint8)
            assert f2_rank(M1 ^ M2) <= f2_rank(M1) + f2_rank(M2)


# ─────────────────────────────────────────────────────────────────
# f2_rref — full structural characterization of THE unique RREF
# ─────────────────────────────────────────────────────────────────


class TestF2RrefContract:
    @staticmethod
    def _assert_rref_structure(R, p):
        r = len(p)
        # Pivot columns strictly increase with row index.
        assert all(p[i] < p[i + 1] for i in range(r - 1))
        # All rows beyond the rank are zero.
        assert not R[r:].any()
        # Every pivot column is a standard basis vector e_i.
        for i, c in enumerate(p):
            assert R[i, c] == 1
            assert int(R[:, c].sum()) == 1
        # Every pivot row is zero strictly left of its pivot.
        for i, c in enumerate(p):
            assert not R[i, :c].any()

    def test_structure_random(self):
        # Structural conditions + row-space equality below uniquely
        # characterize the RREF, so together these are a complete check.
        rng = np.random.default_rng(201)
        for _ in range(15):
            m = int(rng.integers(1, 8))
            n = int(rng.integers(1, 11))
            M = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
            R, p = f2_rref(M)
            assert R.shape == M.shape
            self._assert_rref_structure(R, p)

    def test_rowspace_preserved_exactly(self):
        rng = np.random.default_rng(202)
        for _ in range(10):
            M = rng.integers(0, 2, size=(6, 9)).astype(np.uint8)
            R, _ = f2_rref(M)
            assert _span_set(R) == _span_set(M)

    def test_idempotent(self):
        rng = np.random.default_rng(203)
        for _ in range(8):
            M = rng.integers(0, 2, size=(5, 8)).astype(np.uint8)
            R1, p1 = f2_rref(M)
            R2, p2 = f2_rref(R1)
            np.testing.assert_array_equal(R1, R2)
            assert p1 == p2

    def test_zero_dimension_inputs(self):
        R, p = f2_rref(np.zeros((0, 4), dtype=np.uint8))
        assert R.shape == (0, 4)
        assert p == []
        R, p = f2_rref(np.zeros((3, 0), dtype=np.uint8))
        assert R.shape == (3, 0)
        assert p == []

    def test_hand_computed_examples(self):
        # Two equal rows: the second is eliminated.
        R, p = f2_rref(np.array([[0, 1, 1],
                                 [0, 1, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(R, [[0, 1, 1], [0, 0, 0]])
        assert p == [1]
        # By hand: r1 <- r1 XOR r0 gives [0,1,0,1]; already reduced.
        R, p = f2_rref(np.array([[1, 0, 1, 1],
                                 [1, 1, 1, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(R, [[1, 0, 1, 1], [0, 1, 0, 1]])
        assert p == [0, 1]


# ─────────────────────────────────────────────────────────────────
# f2_solve — uniqueness, inverses, degenerate widths
# ─────────────────────────────────────────────────────────────────


class TestF2SolveContract:
    def test_recovers_the_exact_planted_solution(self):
        # For invertible A the solution of A X = B is UNIQUE, so f2_solve
        # must return the planted X exactly (stronger than A@X == B).
        rng = np.random.default_rng(301)
        for _ in range(10):
            n = int(rng.integers(1, 7))
            m = int(rng.integers(1, 5))
            A = _random_invertible(rng, n)
            X = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
            B = _mm2(A, X)
            np.testing.assert_array_equal(f2_solve(A, B), X)

    def test_solve_against_identity_gives_two_sided_inverse(self):
        rng = np.random.default_rng(302)
        for _ in range(5):
            n = int(rng.integers(2, 7))
            A = _random_invertible(rng, n)
            Ainv = f2_solve(A, np.eye(n, dtype=np.uint8))
            eye = np.eye(n, dtype=np.uint8)
            np.testing.assert_array_equal(_mm2(A, Ainv), eye)
            np.testing.assert_array_equal(_mm2(Ainv, A), eye)

    def test_upper_triangular_hand_example(self):
        # A = [[1,1],[0,1]] is self-inverse over GF(2);
        # X = A^{-1} [[1],[1]] = [[0],[1]] by hand.
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        B = np.array([[1], [1]], dtype=np.uint8)
        np.testing.assert_array_equal(f2_solve(A, B), [[0], [1]])

    def test_zero_width_rhs(self):
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        X = f2_solve(A, np.zeros((2, 0), dtype=np.uint8))
        assert X.shape == (2, 0)

    def test_one_by_one(self):
        np.testing.assert_array_equal(
            f2_solve(np.array([[1]], dtype=np.uint8),
                     np.array([[1]], dtype=np.uint8)),
            [[1]],
        )

    def test_nonbinary_entries_normalized(self):
        # Contract: any nonzero entry counts as 1. A = diag(2, -3) is the
        # identity over GF(2); B entry 3 is 1.
        A = np.array([[2, 0], [0, -3]], dtype=np.int64)
        B = np.array([[1], [3]], dtype=np.int64)
        np.testing.assert_array_equal(f2_solve(A, B), [[1], [1]])

    def test_inputs_not_mutated(self):
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        B = np.array([[1], [0]], dtype=np.uint8)
        A_c, B_c = A.copy(), B.copy()
        f2_solve(A, B)
        np.testing.assert_array_equal(A, A_c)
        np.testing.assert_array_equal(B, B_c)


# ─────────────────────────────────────────────────────────────────
# f2_reduce — coset-remainder contract
# ─────────────────────────────────────────────────────────────────


class TestF2ReduceContract:
    def test_remainder_properties_random(self):
        # r = f2_reduce(v, R, p) must satisfy:
        #   (a) r is zero on every pivot column,
        #   (b) v XOR r lies in the row space of the basis,
        #   (c) reducing again changes nothing (idempotence).
        rng = np.random.default_rng(401)
        for _ in range(12):
            M = rng.integers(0, 2, size=(4, 9)).astype(np.uint8)
            R, p = f2_rref(M)
            v = rng.integers(0, 2, size=9).astype(np.uint8)
            r = f2_reduce(v, R, p)
            assert r.shape == v.shape
            for c in p:
                assert r[c] == 0
            assert _row_to_int(v ^ r) in _span_set(M)
            np.testing.assert_array_equal(f2_reduce(r, R, p), r)

    def test_vector_in_rowspace_reduces_to_zero(self):
        rng = np.random.default_rng(402)
        for _ in range(10):
            M = rng.integers(0, 2, size=(4, 8)).astype(np.uint8)
            R, p = f2_rref(M)
            # Random combination of rows of M — guaranteed in the row space.
            coeffs = rng.integers(0, 2, size=(1, 4)).astype(np.uint8)
            v = _mm2(coeffs, M)[0]
            np.testing.assert_array_equal(
                f2_reduce(v, R, p), np.zeros(8, dtype=np.uint8))

    def test_zero_vector_stays_zero(self):
        M = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        R, p = f2_rref(M)
        np.testing.assert_array_equal(
            f2_reduce(np.zeros(3, dtype=np.uint8), R, p),
            np.zeros(3, dtype=np.uint8))


# ─────────────────────────────────────────────────────────────────
# f2_null_space — exact kernel identification
# ─────────────────────────────────────────────────────────────────


class TestF2NullSpaceContract:
    def test_span_is_exactly_the_kernel(self):
        # Strongest possible check: the span of the returned basis equals
        # the exhaustively enumerated kernel, as sets.
        rng = np.random.default_rng(501)
        for _ in range(6):
            m = int(rng.integers(1, 5))
            n = int(rng.integers(1, 9))
            M = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
            N = f2_null_space(M)
            assert _span_set(N) == _bf_kernel_set(M)

    def test_zero_row_matrix_kernel_is_everything(self):
        N = f2_null_space(np.zeros((0, 4), dtype=np.uint8))
        assert N.shape == (4, 4)
        assert _span_set(N) == set(range(16))

    def test_zero_col_matrix_kernel_is_trivial(self):
        N = f2_null_space(np.zeros((3, 0), dtype=np.uint8))
        assert N.shape == (0, 0)

    def test_dtype_and_input_not_mutated(self):
        M = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
        M_c = M.copy()
        N = f2_null_space(M)
        assert N.dtype == np.uint8
        np.testing.assert_array_equal(M, M_c)


# ─────────────────────────────────────────────────────────────────
# f2_fast: f2_rank_fast — word-boundary widths
# ─────────────────────────────────────────────────────────────────


class TestF2RankFast:
    def test_matches_bruteforce_at_word_boundaries(self):
        rng = np.random.default_rng(601)
        for n in [1, 2, 5, 63, 64, 65, 127, 128, 129]:
            for _ in range(3):
                m = int(rng.integers(1, 8))
                M = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
                assert f2_rank_fast(M) == _bf_rank(M), f"width {n}"

    def test_matches_f2_rank_on_larger_matrices(self):
        # Cross-implementation consistency on shapes too big for the
        # span oracle (f2_rank itself is oracle-verified above).
        rng = np.random.default_rng(602)
        for m, n in [(30, 100), (50, 200), (40, 64), (64, 64), (65, 63)]:
            M = rng.integers(0, 2, size=(m, n)).astype(np.uint8)
            assert f2_rank_fast(M) == f2_rank(M), f"shape {(m, n)}"

    def test_identity_across_word_boundary(self):
        for w in [63, 64, 65, 128]:
            assert f2_rank_fast(np.eye(w, dtype=np.uint8)) == w

    def test_single_one_at_boundary_columns(self):
        for j in [0, 63, 64, 127]:
            M = np.zeros((1, 130), dtype=np.uint8)
            M[0, j] = 1
            assert f2_rank_fast(M) == 1

    def test_degenerate_shapes(self):
        assert f2_rank_fast(np.zeros((3, 70), dtype=np.uint8)) == 0
        assert f2_rank_fast(np.zeros((0, 5), dtype=np.uint8)) == 0
        assert f2_rank_fast(np.ones((1, 130), dtype=np.uint8)) == 1

    def test_nonbinary_entries_treated_as_one(self):
        M = np.array([[2, 0, 0], [0, -3, 0]], dtype=np.int64)
        assert f2_rank_fast(M) == 2


# ─────────────────────────────────────────────────────────────────
# f2_fast: f2_nullspace_fast
# ─────────────────────────────────────────────────────────────────


class TestF2NullspaceFast:
    def test_contract_at_word_boundaries(self):
        rng = np.random.default_rng(701)
        for n in [3, 63, 64, 65, 129]:
            M = rng.integers(0, 2, size=(6, n)).astype(np.uint8)
            N = f2_nullspace_fast(M)
            r = f2_rank(M)
            # Rank-nullity, membership, independence.
            assert N.shape == (n - r, n)
            assert not _mm2(M, N.T).any()
            assert f2_rank(N) == N.shape[0]

    def test_span_matches_slow_null_space(self):
        # Same subspace as the oracle-verified core.f2 implementation:
        # equal dimensions and stacking does not grow the rank.
        rng = np.random.default_rng(702)
        for _ in range(6):
            M = rng.integers(0, 2, size=(8, 20)).astype(np.uint8)
            N_fast = f2_nullspace_fast(M)
            N_slow = f2_null_space(M)
            assert N_fast.shape[0] == N_slow.shape[0]
            if N_fast.shape[0]:
                assert f2_rank(np.vstack([N_fast, N_slow])) == N_fast.shape[0]

    def test_identity_kernel_empty(self):
        N = f2_nullspace_fast(np.eye(64, dtype=np.uint8))
        assert N.shape == (0, 64)

    def test_single_one_row_kernel_is_coordinate_hyperplane(self):
        # ker(e_j^T) = {v : v_j = 0}: dimension n-1, every basis vector has
        # a zero at j — this pins the exact subspace.
        n = 100
        for j in [0, 63, 64, 99]:
            M = np.zeros((1, n), dtype=np.uint8)
            M[0, j] = 1
            N = f2_nullspace_fast(M)
            assert N.shape == (n - 1, n)
            assert not N[:, j].any()
            assert f2_rank(N) == n - 1


# ─────────────────────────────────────────────────────────────────
# f2_fast: screen_basis — CSS logical-basis contract
# ─────────────────────────────────────────────────────────────────


def _assert_screen_contract(Hx, Hz):
    """Full documented contract of screen_basis, plus pairing nondegeneracy.

    Nondegeneracy is a theorem for any valid logical basis: if a nonzero
    combination v of Lx rows had v . Lz^T = 0, then v would be orthogonal
    to span(Lz) + rowspan(Hz) = ker(Hx), hence v in ker(Hx)^perp =
    rowspan(Hx) — contradicting independence of Lx from rowspan(Hx).
    """
    n = Hx.shape[1]
    Lx, Lz, k = screen_basis(Hx, Hz)
    rx, rz = f2_rank(Hx), f2_rank(Hz)
    assert k == n - rx - rz
    assert Lx.shape == (k, n) and Lz.shape == (k, n)
    assert Lx.dtype == np.uint8 and Lz.dtype == np.uint8
    # Logicals commute with the opposite-type stabilizers.
    assert not _mm2(Hz, Lx.T).any()
    assert not _mm2(Hx, Lz.T).any()
    # Independent from same-type stabilizers (jointly, not just row-wise).
    assert f2_rank(np.vstack([Hx, Lx])) == rx + k
    assert f2_rank(np.vstack([Hz, Lz])) == rz + k
    # Nondegenerate logical pairing.
    assert f2_rank(_mm2(Lx, Lz.T)) == k
    return Lx, Lz, k


def _hgp(H1, H2):
    """Hypergraph product CSS pair (independent reference construction)."""
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    Hx = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]).astype(np.uint8)
    Hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]).astype(np.uint8)
    return Hx, Hz


def _circulant_rep(L):
    """Full-rank-deficient circulant repetition check matrix (1 + x)."""
    H = np.zeros((L, L), dtype=np.uint8)
    for i in range(L):
        H[i, i] = 1
        H[i, (i + 1) % L] = 1
    return H


class TestScreenBasis:
    def test_toric_code_from_hgp(self):
        # HGP of two length-3 circulant repetition codes = 3x3 toric code:
        # n = 18, k = k1*k2 + k1^T*k2^T = 1*1 + 1*1 = 2 (textbook value).
        Hx, Hz = _hgp(_circulant_rep(3), _circulant_rep(3))
        assert not _mm2(Hx, Hz.T).any()   # CSS by construction
        _, _, k = _assert_screen_contract(Hx, Hz)
        assert k == 2

    def test_random_css_pairs(self):
        # Random CSS pairs: rows of Hx are random combinations of a basis of
        # ker(Hz), so Hx . Hz^T = 0 by construction.
        rng = np.random.default_rng(801)
        tested = 0
        for _ in range(20):
            n = int(rng.integers(6, 16))
            Hz = rng.integers(0, 2, size=(int(rng.integers(1, 6)), n)).astype(np.uint8)
            K = f2_null_space(Hz)
            if K.shape[0] == 0:
                continue
            sel = rng.integers(0, 2, size=(int(rng.integers(1, 7)), K.shape[0]))
            Hx = _mm2(sel, K)
            assert not _mm2(Hx, Hz.T).any()
            _assert_screen_contract(Hx, Hz)
            tested += 1
        assert tested >= 10

    def test_k_zero_code(self):
        # n=2, rank(Hx) = rank(Hz) = 1 and Hx . Hz^T = [2] = 0 mod 2.
        Hx = np.array([[1, 1]], dtype=np.uint8)
        Hz = np.array([[1, 1]], dtype=np.uint8)
        Lx, Lz, k = screen_basis(Hx, Hz)
        assert k == 0
        assert Lx.shape == (0, 2) and Lz.shape == (0, 2)

    def test_empty_hz_classical_limit(self):
        # Hz has no rows: ker(Hz) is everything, so k = n - rank(Hx).
        # For the length-3 repetition code, the lone Z-logical must span
        # exactly {0, 111} (= ker Hx).
        Hx = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        Hz = np.zeros((0, 3), dtype=np.uint8)
        Lx, Lz, k = _assert_screen_contract(Hx, Hz)
        assert k == 1
        assert _span_set(Lz) == {0, 0b111}

    def test_wide_css_pair_across_word_boundary(self):
        rng = np.random.default_rng(802)
        n = 70
        Hz = rng.integers(0, 2, size=(4, n)).astype(np.uint8)
        K = f2_null_space(Hz)
        sel = rng.integers(0, 2, size=(5, K.shape[0]))
        Hx = _mm2(sel, K)
        assert not _mm2(Hx, Hz.T).any()
        _assert_screen_contract(Hx, Hz)
