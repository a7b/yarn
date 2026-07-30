"""Fresh black-box / contract tests for core.classical_code and
core.quantum_code (Type 2).

Expected values are derived independently of the implementation:

  - the lift maps A -> A_bin (left rep) and B -> B_bin (right rep) are ring
    homomorphisms, so lifting a ring matrix product must equal the mod-2
    product of the lifts, and lifting matrix_dagger must equal the binary
    transpose,
  - hand-derived canonical-form permutations (even-weight ring entries give
    provably singular blocks: every row of L[x] / R[x] has |x| ones, so the
    all-ones vector is in the kernel when |x| is even),
  - a closed-form logical count for 1x1 LP codes over a cyclic group:
    A = B = [[(e, h)]] gives rank Hx = rank Hz = n - n/m (m = ord(h)),
    hence k = 2n/m — derived from the cycle structure of I + permutation,
  - a fully hand-computed Hx / Hz for the LP code over the trivial group
    (pins the documented block placement bit for bit),
  - an int-based Gaussian elimination independent of core.f2 as a rank
    oracle for k,
  - the hypergraph-product toric code (k = 2) as an external known code.
"""

import numpy as np
import pytest

from core.classical_code import (
    A_from_A_bin,
    B_from_B_bin,
    build_A_bin,
    build_B_bin,
    canonical_form_A,
    canonical_form_B,
    weight_matrix,
)
from core.f2 import f2_rank
from core.f2_fast import screen_basis
from core.group import (
    GroupData,
    canonicalize,
    dagger,
    element_order,
    left_rep,
    matrix_dagger,
    right_rep,
    ring_add,
    ring_mul,
)
from core.quantum_code import (
    AB_from_Hx_Hz,
    A_bin_B_bin_from_Hx_Hz,
    build_Hx,
    build_Hz,
    build_quantum_code,
    check_css,
    compute_k,
    quantum_check_weights,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="module")
def gd_c6():
    return GroupData("CyclicGroup(6)")


@pytest.fixture(scope="module")
def gd_triv():
    return GroupData("TrivialGroup()")


# ─────────────────────────────────────────────────────────────────
# Helpers (independent references)
# ─────────────────────────────────────────────────────────────────


def _mm2(A, B):
    return ((np.asarray(A, dtype=np.int64) @ np.asarray(B, dtype=np.int64))
            % 2).astype(np.uint8)


def _int_rank(M):
    """GF(2) rank via Python-int row elimination — independent of core.f2."""
    table = {}
    for row in (np.asarray(M) != 0).astype(np.uint8):
        r = 0
        for j, b in enumerate(row.tolist()):
            if b:
                r |= 1 << j
        while r:
            lead = r.bit_length()
            if lead in table:
                r ^= table[lead]
            else:
                table[lead] = r
                break
    return len(table)


def _ring_matmul(X, Y, gd):
    """Ring matrix product over F2[G] via ring_mul / ring_add."""
    rows, inner, cols = len(X), len(Y), len(Y[0])
    assert len(X[0]) == inner
    return [
        [
            _ring_dot(X[i], [Y[t][j] for t in range(inner)], gd)
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def _ring_dot(xs, ys, gd):
    acc = ()
    for x, y in zip(xs, ys):
        acc = ring_add(acc, ring_mul(x, y, gd))
    return acc


def _rand_ring_matrix(rng, rows, cols, n, max_w=3, min_w=0):
    return [
        [
            tuple(sorted(rng.choice(
                n, size=int(rng.integers(min_w, max_w + 1)),
                replace=False).tolist()))
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]


# ─────────────────────────────────────────────────────────────────
# Lift homomorphism laws (classical level)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
class TestLiftHomomorphisms:
    def test_A_lift_respects_ring_matrix_product(self, gd_s3):
        # L is a ring homomorphism, so the binary lift of a ring matrix
        # product must equal the mod-2 product of the lifts. This pins the
        # row-block/col-block orientation and the choice of L (not R).
        rng = np.random.default_rng(51)
        for _ in range(4):
            X = _rand_ring_matrix(rng, 2, 3, 6)
            Y = _rand_ring_matrix(rng, 3, 2, 6)
            lhs = build_A_bin(_ring_matmul(X, Y, gd_s3), gd_s3)
            rhs = _mm2(build_A_bin(X, gd_s3), build_A_bin(Y, gd_s3))
            np.testing.assert_array_equal(lhs, rhs)

    def test_B_lift_respects_ring_matrix_product(self, gd_s3):
        # R[g1 g2] = R[g1] R[g2] under the documented convention, so the
        # same law holds for the right-rep lift.
        rng = np.random.default_rng(52)
        for _ in range(4):
            X = _rand_ring_matrix(rng, 2, 3, 6)
            Y = _rand_ring_matrix(rng, 3, 2, 6)
            lhs = build_B_bin(_ring_matmul(X, Y, gd_s3), gd_s3)
            rhs = _mm2(build_B_bin(X, gd_s3), build_B_bin(Y, gd_s3))
            np.testing.assert_array_equal(lhs, rhs)

    def test_lift_of_matrix_dagger_is_transpose(self, gd_s3):
        # L[x]^T = L[x†] and R[x]^T = R[x†] extend blockwise:
        # build(M†) = build(M)^T for both lifts.
        rng = np.random.default_rng(53)
        M = _rand_ring_matrix(rng, 2, 3, 6)
        Md = matrix_dagger(M, gd_s3)
        np.testing.assert_array_equal(
            build_A_bin(Md, gd_s3), build_A_bin(M, gd_s3).T)
        np.testing.assert_array_equal(
            build_B_bin(Md, gd_s3), build_B_bin(M, gd_s3).T)

    def test_lift_of_ring_identity_matrix_is_binary_identity(self, gd_s3):
        e = gd_s3.identity
        I_ring = [[(e,), ()], [(), (e,)]]
        eye = np.eye(2 * gd_s3.n, dtype=np.uint8)
        np.testing.assert_array_equal(build_A_bin(I_ring, gd_s3), eye)
        np.testing.assert_array_equal(build_B_bin(I_ring, gd_s3), eye)

    def test_lift_linear_in_entries(self, gd_s3):
        # build(X + Y) = build(X) XOR build(Y) (entrywise ring_add).
        rng = np.random.default_rng(54)
        X = _rand_ring_matrix(rng, 2, 2, 6)
        Y = _rand_ring_matrix(rng, 2, 2, 6)
        S = [[ring_add(X[i][j], Y[i][j]) for j in range(2)] for i in range(2)]
        np.testing.assert_array_equal(
            build_A_bin(S, gd_s3),
            build_A_bin(X, gd_s3) ^ build_A_bin(Y, gd_s3))


# ─────────────────────────────────────────────────────────────────
# Inverse maps: round trips and wrong-representation rejection
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
class TestInverseLifts:
    def test_random_round_trips_many_shapes(self, gd_s3, gd_c6):
        rng = np.random.default_rng(61)
        for gd in (gd_s3, gd_c6):
            for shape in [(1, 1), (2, 3), (3, 2)]:
                A = _rand_ring_matrix(rng, shape[0], shape[1], gd.n)
                A_rec = A_from_A_bin(build_A_bin(A, gd), gd, shape)
                assert A_rec == [[canonicalize(x) for x in row] for row in A]
                B = _rand_ring_matrix(rng, shape[0], shape[1], gd.n)
                B_rec = B_from_B_bin(build_B_bin(B, gd), gd, shape)
                assert B_rec == [[canonicalize(x) for x in row] for row in B]

    def test_A_from_A_bin_rejects_right_rep_lift(self, gd_s3):
        # R[g] is an L-lift of some element only when g is central; S3 has
        # trivial center, so a right-rep block must be rejected.
        g = 1
        R_block = right_rep((g,), gd_s3)
        assert not np.array_equal(
            R_block, left_rep((gd_s3.inv[g],), gd_s3))
        with pytest.raises(ValueError, match="not a valid left-rep lift"):
            A_from_A_bin(R_block, gd_s3, (1, 1))

    def test_B_from_B_bin_rejects_left_rep_lift(self, gd_s3):
        g = 1
        L_block = left_rep((g,), gd_s3)
        assert not np.array_equal(
            L_block, right_rep((gd_s3.inv[g],), gd_s3))
        with pytest.raises(ValueError, match="not a valid right-rep lift"):
            B_from_B_bin(L_block, gd_s3, (1, 1))


# ─────────────────────────────────────────────────────────────────
# Canonical block-col form: hand-derived permutations
# ─────────────────────────────────────────────────────────────────
#
# Singularity certificate used below: every row of L[x] (and of R[x]) has
# exactly |x| ones, so L[x] . 1 = |x| mod 2 — for EVEN-weight x the all-ones
# vector is in the kernel and the block cannot be invertible. Weight-1
# entries are permutation matrices, always invertible; (0,) is the identity.


@pytest.mark.gap
class TestCanonicalFormHandDerived:
    def test_only_middle_block_col_invertible(self, gd_s3):
        # Blocks: (1,2) singular (even weight), (0,) = I invertible,
        # (3,4) singular. Trailing subset (2,) fails; lex-first invertible
        # subset is (1,). others = [0, 2] -> perm = [0, 2, 1].
        A = [[(1, 2), (0,), (3, 4)]]
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        assert perm == [0, 2, 1]
        assert A_can == [[(1, 2), (3, 4), (0,)]]
        n = gd_s3.n
        assert f2_rank(A_bin_can[:, -n:]) == n

    def test_two_of_three_subset_selected(self, gd_s3):
        # Block-col 2 is zero, so the trailing pair (1, 2) fails; the
        # lex-first invertible pair is (0, 1). perm = [2, 0, 1].
        e = gd_s3.identity
        A = [[(e,), (), ()],
             [(), (e,), ()]]
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        assert perm == [2, 0, 1]
        assert A_can == [[(), (e,), ()],
                         [(), (), (e,)]]
        n = gd_s3.n
        assert f2_rank(A_bin_can[:, -2 * n:]) == 2 * n

    def test_idempotent_on_own_output(self, gd_s3):
        A = [[(1, 2), (0,), (3, 4)]]
        A_can, A_bin_can, _, _ = canonical_form_A(A, gd_s3)
        A_can2, A_bin_can2, perm2, found2 = canonical_form_A(A_can, gd_s3)
        assert found2 is True
        assert perm2 == [0, 1, 2]
        assert A_can2 == A_can
        np.testing.assert_array_equal(A_bin_can2, A_bin_can)

    def test_B_side_hand_derived_and_idempotent(self, gd_s3):
        # Same singularity certificate applies to R-blocks.
        B = [[(3, 4), (0,), (1, 2)]]
        B_can, B_bin_can, perm, found = canonical_form_B(B, gd_s3)
        assert found is True
        assert perm == [0, 2, 1]
        assert B_can == [[(3, 4), (1, 2), (0,)]]
        n = gd_s3.n
        assert f2_rank(B_bin_can[:, -n:]) == n
        B_can2, _, perm2, _ = canonical_form_B(B_can, gd_s3)
        assert perm2 == [0, 1, 2]
        assert B_can2 == B_can

    def test_all_even_weight_entries_never_full_rank(self, gd_s3):
        # Every block-col submatrix kills the all-ones vector.
        B = [[(1, 2), (3, 4)]]
        B_can, _, perm, found = canonical_form_B(B, gd_s3)
        assert found is False
        assert perm == [0, 1]
        assert B_can == [[(1, 2), (3, 4)]]

    def test_trailing_subset_preferred_when_invertible(self, gd_s3):
        # Both singleton blocks are invertible (weight 1). The trailing
        # subset must win so the map is idempotent — perm stays identity
        # even though (0,) is the lex-first invertible subset.
        A = [[(2,), (1,)]]
        A_can, _, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        assert perm == [0, 1]
        assert A_can == [[(2,), (1,)]]

    def test_square_full_rank_keeps_identity_perm(self, gd_s3):
        # 2x2 block lower-triangular with identity diagonal: invertible, and
        # the trailing subset is the whole set of block-cols.
        e = gd_s3.identity
        A = [[(e,), ()],
             [(1,), (e,)]]
        A_can, A_bin_can, perm, found = canonical_form_A(A, gd_s3)
        assert found is True
        assert perm == [0, 1]
        assert A_can == A
        assert f2_rank(A_bin_can) == 2 * gd_s3.n


# ─────────────────────────────────────────────────────────────────
# Quantum level: known k values and structural laws
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
class TestQuantumKnownK:
    def test_k_closed_form_1x1_cyclic(self, gd_c6):
        # A = B = [[(e, h)]] over C_n, m = ord(h):
        #   Hx = [L[e+h] | R[e+h^{-1}]], and u^T Hx = 0 iff u is fixed by
        #   both left- and right-translation by h, i.e. constant on <h>
        #   cosets: rank Hx = n - n/m; same for Hz.
        #   k = 2n - 2(n - n/m) = 2n/m.
        n = gd_c6.n
        for h in range(1, n):
            A = [[(0, h)]]
            B = [[(0, h)]]
            Hx = build_Hx(A, B, gd_c6)
            Hz = build_Hz(A, B, gd_c6)
            m = element_order(h, gd_c6)
            assert check_css(Hx, Hz)
            assert compute_k(Hx, Hz) == 2 * n // m, f"h={h}, ord={m}"

    def test_k_against_independent_int_rank(self, gd_s3, gd_c6):
        # compute_k must agree with an elimination routine that shares no
        # code with core.f2.
        rng = np.random.default_rng(71)
        for gd in (gd_s3, gd_c6):
            A = _rand_ring_matrix(rng, 1, 2, gd.n, min_w=1)
            B = _rand_ring_matrix(rng, 1, 2, gd.n, min_w=1)
            Hx = build_Hx(A, B, gd)
            Hz = build_Hz(A, B, gd)
            assert check_css(Hx, Hz)
            k_ref = Hx.shape[1] - _int_rank(Hx) - _int_rank(Hz)
            assert compute_k(Hx, Hz) == k_ref

    def test_screen_basis_k_matches_compute_k_on_lp_codes(self, gd_s3, gd_c6):
        # Cross-implementation: bit-packed screen vs naive f2 rank, plus
        # logical orthogonality on a real LP code.
        cases = [
            (gd_c6, [[(0, 1)]], [[(0, 1)]]),
            (gd_s3, [[(0,), (1, 2)]], [[(0,), (3,)]]),
        ]
        for gd, A, B in cases:
            Hx = build_Hx(A, B, gd)
            Hz = build_Hz(A, B, gd)
            Lx, Lz, k = screen_basis(Hx, Hz)
            assert k == compute_k(Hx, Hz)
            assert not _mm2(Hz, Lx.T).any()
            assert not _mm2(Hx, Lz.T).any()

    def test_k_invariant_under_canonicalization(self, gd_s3):
        # canonical_form_* only permutes block-cols, i.e. relabels physical
        # qubits — n, CSS, and k must be unchanged. Entries chosen so a real
        # permutation happens: (3,4) / (1,2) are even weight (singular)
        # and sit at the trailing position.
        A = [[(0,), (3, 4)]]
        B = [[(0,), (1, 2)]]
        Hx_raw = build_Hx(A, B, gd_s3)
        Hz_raw = build_Hz(A, B, gd_s3)
        result = build_quantum_code(A, B, gd_s3)
        assert result["perm_a"] == [1, 0]
        assert result["perm_b"] == [1, 0]
        assert check_css(result["Hx"], result["Hz"])
        assert result["Hx"].shape == Hx_raw.shape
        assert compute_k(result["Hx"], result["Hz"]) == \
            compute_k(Hx_raw, Hz_raw)


@pytest.mark.gap
class TestQuantumStructure:
    def test_trivial_group_block_placement_bit_exact(self, gd_triv):
        # Over the trivial group every rep block is the 1x1 matrix [1] (or
        # [0]), so the documented block placement can be written out by hand.
        # A = B = [[(e,), (e,)]] (ma=mb=1, na=nb=2, n=1):
        #   Hx rows (ia=0, ib): left 1s at cols (ja*2 + ib), right 1 at col 4.
        #   Hz rows (ja, ib=0): left 1s at cols (ja*2 + jb), right 1 at col 4.
        A = [[(0,), (0,)]]
        B = [[(0,), (0,)]]
        Hx = build_Hx(A, B, gd_triv)
        Hz = build_Hz(A, B, gd_triv)
        np.testing.assert_array_equal(Hx, [[1, 0, 1, 0, 1],
                                           [0, 1, 0, 1, 1]])
        np.testing.assert_array_equal(Hz, [[1, 1, 0, 0, 1],
                                           [0, 0, 1, 1, 1]])
        assert check_css(Hx, Hz)
        assert compute_k(Hx, Hz) == 1     # 5 - 2 - 2
        # And the inverse map recovers (A, B) exactly.
        A_rec, B_rec = AB_from_Hx_Hz(Hx, Hz, gd_triv, (1, 2))
        assert A_rec == A
        assert B_rec == B

    def test_duplicate_support_entries_reduce_mod2(self, gd_s3):
        # (g, g) is the zero ring element; builders must treat it as ().
        A1 = [[(1, 1), (0,)]]
        A2 = [[(), (0,)]]
        B = [[(0,), (2,)]]
        np.testing.assert_array_equal(
            build_Hx(A1, B, gd_s3), build_Hx(A2, B, gd_s3))
        np.testing.assert_array_equal(
            build_Hz(A1, B, gd_s3), build_Hz(A2, B, gd_s3))

    def test_right_section_corruption_division_of_labor(self, gd_s3):
        # Documented split: A_bin_B_bin_from_Hx_Hz validates only the LEFT
        # sections, so a right-section bit flip must slip through it but be
        # caught by AB_from_Hx_Hz's rebuild cross-check.
        A = [[(0,), (1, 2)]]
        B = [[(0,), (3,)]]
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        left_cols = 2 * 2 * gd_s3.n     # na * nb * n
        Hx_bad = Hx.copy()
        Hx_bad[0, left_cols] ^= 1
        A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx_bad, Hz, gd_s3, (1, 2))
        np.testing.assert_array_equal(A_bin, build_A_bin(A, gd_s3))
        np.testing.assert_array_equal(B_bin, build_B_bin(B, gd_s3))
        with pytest.raises(ValueError, match="Hx cross-check failed"):
            AB_from_Hx_Hz(Hx_bad, Hz, gd_s3, (1, 2))

    def test_check_weights_match_built_code_nonabelian_2x2(self, gd_s3):
        # quantum_check_weights is pure arithmetic on weight matrices; it
        # must equal the true max row weights of the built Hx / Hz.
        rng = np.random.default_rng(72)
        for _ in range(3):
            A = _rand_ring_matrix(rng, 2, 2, 6)
            B = _rand_ring_matrix(rng, 2, 2, 6)
            Hx = build_Hx(A, B, gd_s3)
            Hz = build_Hz(A, B, gd_s3)
            derived = quantum_check_weights(weight_matrix(A), weight_matrix(B))
            assert derived["Hx_check_weight"] == int(Hx.sum(axis=1).max())
            assert derived["Hz_check_weight"] == int(Hz.sum(axis=1).max())


# ─────────────────────────────────────────────────────────────────
# GAP-free quantum checks (hand-built binary CSS codes)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
class TestQuantumFunctionsFast:
    def test_toric_code_from_hypergraph_product(self):
        # HGP of two length-3 circulant repetition codes is the 3x3 toric
        # code: n = 18, k = 2 (textbook). Independent of the LP machinery.
        L = 3
        H = np.zeros((L, L), dtype=np.uint8)
        for i in range(L):
            H[i, i] = 1
            H[i, (i + 1) % L] = 1
        Hx = np.hstack([np.kron(H, np.eye(L, dtype=np.uint8)),
                        np.kron(np.eye(L, dtype=np.uint8), H.T)])
        Hz = np.hstack([np.kron(np.eye(L, dtype=np.uint8), H),
                        np.kron(H.T, np.eye(L, dtype=np.uint8))])
        assert check_css(Hx, Hz)
        assert compute_k(Hx, Hz) == 2

    def test_check_css_false_on_anticommuting_pair(self):
        Hx = np.array([[1, 1, 0]], dtype=np.uint8)
        Hz = np.array([[1, 0, 0]], dtype=np.uint8)
        assert check_css(Hx, Hz) is False

    def test_compute_k_zero(self):
        H = np.array([[1, 1]], dtype=np.uint8)
        assert check_css(H, H)
        assert compute_k(H, H) == 0

    def test_check_weights_hand_values_rectangular(self):
        # W_A rowsums [3, 5] -> 5; colsums [4, 3, 1] -> 4.
        # W_B rowsums [3, 2, 2] -> 3; colsums [3, 4] -> 4.
        # Hx = max rowsum(A) + max colsum(B) = 5 + 4 = 9.
        # Hz = max colsum(A) + max rowsum(B) = 4 + 3 = 7.
        W_A = np.array([[1, 2, 0],
                        [3, 1, 1]])
        W_B = np.array([[2, 1],
                        [0, 2],
                        [1, 1]])
        out = quantum_check_weights(W_A, W_B)
        assert out == {"Hx_check_weight": 9, "Hz_check_weight": 7}

    def test_check_weights_empty_side_contributes_zero(self):
        W_B = np.array([[2, 1]])
        out = quantum_check_weights(np.zeros((0, 3), dtype=int), W_B)
        # rowsum_A_max = colsum_A_max = 0; colsum_B_max = 2; rowsum_B_max = 3.
        assert out == {"Hx_check_weight": 2, "Hz_check_weight": 3}

    def test_check_weights_1d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            quantum_check_weights(np.array([1, 2]), np.array([[1]]))
