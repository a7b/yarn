"""Fresh black-box / contract tests for core.group (Type 2).

Written from the documented contracts (module docstrings + core/README.md)
with expected values derived independently:

  - group axioms checked exhaustively on the multiplication table
    (associativity, Latin-square property, uniqueness of inverses),
  - GAP's own ``Order`` as an external oracle for element_order,
  - textbook facts about small groups (C6 order multiset, Q8 structure),
  - algebraic laws that any correct implementation must satisfy
    (dagger anti-homomorphism, ring associativity, augmentation parity,
    faithfulness of the left-regular representation),
  - a structurally independent coefficient-vector reference implementation
    for ring_permanent,
  - groups not used by the existing tests: C6, Q8 = SmallGroup(8,4),
    C2 x C3 (direct product), and the trivial group.
"""

import itertools
from math import lcm

import numpy as np
import pytest
from gappy import gap

from core.group import (
    GroupData,
    canonicalize,
    dagger,
    element_order,
    is_self_dagger,
    left_rep,
    matrix_dagger,
    right_rep,
    ring_add,
    ring_mul,
    ring_permanent,
)

pytestmark = pytest.mark.gap


# ─────────────────────────────────────────────────────────────────
# Fixtures (module-scoped: GroupData init is O(n^2) GAP calls)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="module")
def gd_c6():
    return GroupData("CyclicGroup(6)")


@pytest.fixture(scope="module")
def gd_q8():
    # Quaternion group: 1 element of order 1, 1 of order 2, 6 of order 4.
    return GroupData("SmallGroup(8,4)")


@pytest.fixture(scope="module")
def gd_c2xc3():
    return GroupData("DirectProduct(CyclicGroup(2), CyclicGroup(3))")


@pytest.fixture(scope="module")
def gd_triv():
    return GroupData("TrivialGroup()")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _pow(g, j, gd):
    """g^j via the multiplication table (j >= 0)."""
    acc = gd.identity
    for _ in range(j):
        acc = gd.mult[acc][g]
    return acc


def _rand_support(rng, n, min_w=1):
    """Random canonical ring element (sorted tuple of distinct indices)."""
    w = int(rng.integers(min_w, n + 1))
    return tuple(sorted(rng.choice(n, size=w, replace=False).tolist()))


def _mm2(A, B):
    return ((np.asarray(A, dtype=np.int64) @ np.asarray(B, dtype=np.int64))
            % 2).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
# Multiplication-table axioms (exhaustive)
# ─────────────────────────────────────────────────────────────────


class TestMultTableAxioms:
    def test_associativity_s3(self, gd_s3):
        n = gd_s3.n
        m = gd_s3.mult
        for a, b, c in itertools.product(range(n), repeat=3):
            assert m[m[a][b]][c] == m[a][m[b][c]]

    def test_associativity_q8(self, gd_q8):
        n = gd_q8.n
        m = gd_q8.mult
        for a, b, c in itertools.product(range(n), repeat=3):
            assert m[m[a][b]][c] == m[a][m[b][c]]

    def test_latin_square_property(self, gd_s3, gd_q8, gd_c6):
        # Every row and every column of the table is a permutation of the
        # element indices (left/right translation is a bijection).
        for gd in (gd_s3, gd_q8, gd_c6):
            n = gd.n
            full = set(range(n))
            for g in range(n):
                assert set(gd.mult[g]) == full
                assert {gd.mult[h][g] for h in range(n)} == full

    def test_inverse_is_the_unique_table_solution(self, gd_s3):
        e = gd_s3.identity
        for g in range(gd_s3.n):
            sols = [h for h in range(gd_s3.n) if gd_s3.mult[g][h] == e]
            assert sols == [gd_s3.inv[g]]

    def test_is_abelian_flag_matches_table(self, gd_c6, gd_q8):
        for gd, expected in ((gd_c6, True), (gd_q8, False)):
            symmetric = all(
                gd.mult[a][b] == gd.mult[b][a]
                for a, b in itertools.product(range(gd.n), repeat=2)
            )
            assert gd.is_abelian is expected
            assert symmetric is expected


# ─────────────────────────────────────────────────────────────────
# Trivial group (order 1) — extreme edge case
# ─────────────────────────────────────────────────────────────────


class TestTrivialGroup:
    def test_basic_tables(self, gd_triv):
        assert gd_triv.n == 1
        assert gd_triv.identity == 0
        assert gd_triv.mult == [[0]]
        assert gd_triv.inv == [0]
        assert gd_triv.is_abelian is True
        assert gd_triv.commutator == frozenset({0})
        assert gd_triv.commutator_order == 1
        assert gd_triv.abelianization_order == 1

    def test_ring_and_reps(self, gd_triv):
        assert ring_mul((0,), (0,), gd_triv) == (0,)
        assert dagger((0,), gd_triv) == (0,)
        assert element_order(0, gd_triv) == 1
        np.testing.assert_array_equal(left_rep((0,), gd_triv), [[1]])
        np.testing.assert_array_equal(right_rep((0,), gd_triv), [[1]])
        np.testing.assert_array_equal(left_rep((), gd_triv), [[0]])


# ─────────────────────────────────────────────────────────────────
# element_order — definition-level and external-oracle checks
# ─────────────────────────────────────────────────────────────────


class TestElementOrderContract:
    def test_is_minimal_positive_power_reaching_identity(self, gd_c6, gd_q8):
        for gd in (gd_c6, gd_q8):
            e = gd.identity
            for g in range(gd.n):
                m = element_order(g, gd)
                assert _pow(g, m, gd) == e
                for j in range(1, m):
                    assert _pow(g, j, gd) != e

    def test_c6_order_multiset(self, gd_c6):
        # C6 has phi(d) elements of order d for each d | 6: 1,1,2,2.
        assert sorted(element_order(g, gd_c6) for g in range(6)) == \
            [1, 2, 3, 3, 6, 6]

    def test_q8_order_multiset(self, gd_q8):
        # Q8: identity, the central involution -1, and six elements of
        # order 4 (textbook).
        assert sorted(element_order(g, gd_q8) for g in range(8)) == \
            [1, 2, 4, 4, 4, 4, 4, 4]

    def test_matches_gap_order_oracle(self, gd_s3):
        # GAP's Elements(...) ordering is deterministic for a fixed
        # construction, so Order() can be read back element-by-element.
        for i in range(gd_s3.n):
            gap_order = int(gap.eval(
                f"Order(Elements(SymmetricGroup(3))[{i + 1}])"))
            assert element_order(i, gd_s3) == gap_order

    def test_abelian_product_order_divides_lcm(self, gd_c6):
        # In an abelian group (gh)^L = g^L h^L, so ord(gh) | lcm(ord g, ord h).
        for g, h in itertools.product(range(6), repeat=2):
            L = lcm(element_order(g, gd_c6), element_order(h, gd_c6))
            assert L % element_order(gd_c6.mult[g][h], gd_c6) == 0


# ─────────────────────────────────────────────────────────────────
# dagger algebra
# ─────────────────────────────────────────────────────────────────


class TestDaggerAlgebra:
    def test_singleton_dagger_is_inverse(self, gd_s3, gd_q8):
        for gd in (gd_s3, gd_q8):
            for g in range(gd.n):
                assert dagger((g,), gd) == (gd.inv[g],)

    def test_antihomomorphism(self, gd_s3, gd_q8):
        # (xy)† = y† x† — the order flip matters for non-abelian groups.
        rng = np.random.default_rng(11)
        for gd in (gd_s3, gd_q8):
            for _ in range(15):
                x = _rand_support(rng, gd.n)
                y = _rand_support(rng, gd.n)
                lhs = dagger(ring_mul(x, y, gd), gd)
                rhs = ring_mul(dagger(y, gd), dagger(x, gd), gd)
                assert lhs == rhs

    def test_self_dagger_singleton_iff_order_at_most_two(self, gd_s3, gd_q8):
        for gd in (gd_s3, gd_q8):
            for g in range(gd.n):
                assert is_self_dagger((g,), gd) == \
                    (element_order(g, gd) <= 2)

    def test_matrix_dagger_entrywise_nonsquare(self, gd_s3):
        M = [[(0,), (1, 2), ()],
             [(3,), (), (4, 5)]]
        Md = matrix_dagger(M, gd_s3)
        assert len(Md) == 3 and len(Md[0]) == 2
        for i in range(2):
            for j in range(3):
                assert Md[j][i] == dagger(M[i][j], gd_s3)
        # Involution also for non-square shape.
        assert matrix_dagger(Md, gd_s3) == M

    def test_matrix_dagger_canonicalizes_entries(self, gd_s3):
        # (1, 1, 2) is (2,) mod 2, so its dagger is dagger((2,)).
        Md = matrix_dagger([[(1, 1, 2)]], gd_s3)
        assert Md == [[dagger((2,), gd_s3)]]


# ─────────────────────────────────────────────────────────────────
# Ring arithmetic laws
# ─────────────────────────────────────────────────────────────────


class TestRingAlgebraLaws:
    def test_ring_mul_associative(self, gd_s3):
        rng = np.random.default_rng(21)
        for _ in range(15):
            x = _rand_support(rng, 6)
            y = _rand_support(rng, 6)
            z = _rand_support(rng, 6)
            assert ring_mul(ring_mul(x, y, gd_s3), z, gd_s3) == \
                ring_mul(x, ring_mul(y, z, gd_s3), gd_s3)

    def test_ring_mul_reduces_duplicate_inputs_mod2(self, gd_s3):
        # Module contract: repeated input entries cancel mod 2.
        y = (0, 3)
        assert ring_mul((1, 1), y, gd_s3) == ()
        assert ring_mul(y, (2, 2), gd_s3) == ()
        assert ring_mul((1, 1, 4), y, gd_s3) == ring_mul((4,), y, gd_s3)
        assert ring_mul(y, (1, 4, 1), gd_s3) == ring_mul(y, (4,), gd_s3)

    def test_augmentation_parity(self, gd_s3, gd_c6):
        # The augmentation map eps(x) = |x| mod 2 is a ring homomorphism
        # F2[G] -> F2: eps(x+y) = eps(x)+eps(y), eps(xy) = eps(x)eps(y).
        rng = np.random.default_rng(22)
        for gd in (gd_s3, gd_c6):
            for _ in range(12):
                x = _rand_support(rng, gd.n)
                y = _rand_support(rng, gd.n)
                assert len(ring_add(x, y)) % 2 == (len(x) + len(y)) % 2
                assert len(ring_mul(x, y, gd)) % 2 == (len(x) * len(y)) % 2

    def test_ring_mul_consistent_with_faithful_left_rep(self, gd_s3):
        # L is a faithful ring homomorphism: L[x y] = L[x] L[y], and the
        # identity column of L[z] reads off the support of z. Both give an
        # independent handle on ring_mul's output.
        rng = np.random.default_rng(23)
        e = gd_s3.identity
        assert e == 0
        for _ in range(10):
            x = _rand_support(rng, 6)
            y = _rand_support(rng, 6)
            prod = ring_mul(x, y, gd_s3)
            Lprod = _mm2(left_rep(x, gd_s3), left_rep(y, gd_s3))
            np.testing.assert_array_equal(left_rep(prod, gd_s3), Lprod)
            assert tuple(int(k) for k in np.where(Lprod[:, e])[0]) == prod


class TestRingPermanent:
    def test_zero_row_gives_zero(self, gd_s3):
        # Every permutation term picks one factor from the zero row.
        M = [[(1,), (2, 3)],
             [(), ()]]
        assert ring_permanent(M, gd_s3) == ()

    def test_constant_matrix_cancels_any_group(self, gd_s3):
        # All entries equal x: every term is x . x, and there are J! = 2
        # of them, cancelling mod 2 — in any group.
        x = (1, 4)
        assert ring_permanent([[x, x], [x, x]], gd_s3) == ()

    def test_equal_rows_cancel_abelian(self, gd_c6):
        # perm = x y + y x = 0 when the ring is commutative.
        x, y = (1,), (2, 4)
        assert ring_permanent([[x, y], [x, y]], gd_c6) == ()

    def test_row_and_col_swap_invariance_abelian(self, gd_c6):
        rng = np.random.default_rng(31)
        M = [[_rand_support(rng, 6, min_w=0) for _ in range(3)]
             for _ in range(3)]
        base = ring_permanent(M, gd_c6)
        M_rowswap = [M[1], M[0], M[2]]
        M_colswap = [[row[2], row[1], row[0]] for row in M]
        assert ring_permanent(M_rowswap, gd_c6) == base
        assert ring_permanent(M_colswap, gd_c6) == base

    def test_matches_coefficient_vector_reference(self, gd_s3, gd_c6):
        # Structurally independent reference: coefficient vectors over the
        # multiplication table, products taken in row order (the documented
        # formula), accumulated by XOR.
        def ref_perm(M, gd):
            n = gd.n

            def vec(sup):
                v = np.zeros(n, dtype=np.uint8)
                for g in sup:
                    v[g] ^= 1
                return v

            def mul(u, v):
                out = np.zeros(n, dtype=np.uint8)
                for i in np.nonzero(u)[0]:
                    for j in np.nonzero(v)[0]:
                        out[gd.mult[int(i)][int(j)]] ^= 1
                return out

            J = len(M)
            total = np.zeros(n, dtype=np.uint8)
            for sigma in itertools.permutations(range(J)):
                term = vec(M[0][sigma[0]])
                for i in range(1, J):
                    term = mul(term, vec(M[i][sigma[i]]))
                total ^= term
            return tuple(int(i) for i in np.nonzero(total)[0])

        rng = np.random.default_rng(32)
        for gd in (gd_s3, gd_c6):
            for J in (2, 3):
                M = [[_rand_support(rng, gd.n, min_w=0) for _ in range(J)]
                     for _ in range(J)]
                assert ring_permanent(M, gd) == ref_perm(M, gd)


# ─────────────────────────────────────────────────────────────────
# Representations: pointwise action, weights, L/R relations
# ─────────────────────────────────────────────────────────────────


class TestRepActionsAndWeights:
    def test_left_rep_action_on_basis_vectors(self, gd_s3):
        # L[g] e_h = e_{g h} (documented: column h gets a 1 in row g.h).
        n = gd_s3.n
        for g, h in itertools.product(range(n), repeat=2):
            eh = np.zeros(n, dtype=np.uint8)
            eh[h] = 1
            out = _mm2(left_rep((g,), gd_s3), eh.reshape(-1, 1)).ravel()
            expected = np.zeros(n, dtype=np.uint8)
            expected[gd_s3.mult[g][h]] = 1
            np.testing.assert_array_equal(out, expected)

    def test_right_rep_action_on_basis_vectors(self, gd_s3):
        # Documented convention: R[g] e_h = e_{h g^{-1}}.
        n = gd_s3.n
        for g, h in itertools.product(range(n), repeat=2):
            eh = np.zeros(n, dtype=np.uint8)
            eh[h] = 1
            out = _mm2(right_rep((g,), gd_s3), eh.reshape(-1, 1)).ravel()
            expected = np.zeros(n, dtype=np.uint8)
            expected[gd_s3.mult[h][gd_s3.inv[g]]] = 1
            np.testing.assert_array_equal(out, expected)

    def test_row_and_col_weights_equal_support_size(self, gd_s3):
        # L[x] (resp. R[x]) is a XOR of |x| distinct permutation matrices
        # whose 1-positions never collide within a column (g1 h = g2 h
        # implies g1 = g2), so every row and column has exactly |x| ones.
        rng = np.random.default_rng(41)
        for _ in range(8):
            x = _rand_support(rng, 6)
            for rep in (left_rep, right_rep):
                M = rep(x, gd_s3)
                assert np.all(M.sum(axis=0) == len(x))
                assert np.all(M.sum(axis=1) == len(x))

    def test_duplicate_support_entries_cancel(self, gd_s3):
        zero = np.zeros((6, 6), dtype=np.uint8)
        np.testing.assert_array_equal(left_rep((2, 2), gd_s3), zero)
        np.testing.assert_array_equal(right_rep((2, 2), gd_s3), zero)
        np.testing.assert_array_equal(
            left_rep((1, 3, 1), gd_s3), left_rep((3,), gd_s3))
        np.testing.assert_array_equal(
            right_rep((1, 3, 1), gd_s3), right_rep((3,), gd_s3))

    def test_abelian_left_rep_equals_right_rep_of_dagger(self, gd_c6):
        # Abelian only: L[g] e_h = e_{gh} = e_{h(g^{-1})^{-1}} = R[g^{-1}] e_h,
        # extended by linearity to L[x] = R[x†].
        rng = np.random.default_rng(42)
        for g in range(6):
            np.testing.assert_array_equal(
                left_rep((g,), gd_c6),
                right_rep(dagger((g,), gd_c6), gd_c6))
        for _ in range(5):
            x = _rand_support(rng, 6)
            np.testing.assert_array_equal(
                left_rep(x, gd_c6), right_rep(dagger(x, gd_c6), gd_c6))

    def test_nonabelian_left_rep_differs_from_right_dagger(self, gd_s3):
        # L[g] = R[g^{-1}] forces g central; S3 has trivial center, so every
        # non-identity element is a counterexample.
        diffs = [
            g for g in range(1, 6)
            if not np.array_equal(left_rep((g,), gd_s3),
                                  right_rep(dagger((g,), gd_s3), gd_s3))
        ]
        assert diffs == [1, 2, 3, 4, 5]

    def test_left_right_commute_on_q8(self, gd_q8):
        # L/R commutation on a non-abelian group not covered elsewhere.
        rng = np.random.default_rng(43)
        for _ in range(8):
            x = _rand_support(rng, 8)
            y = _rand_support(rng, 8)
            L = left_rep(x, gd_q8)
            R = right_rep(y, gd_q8)
            np.testing.assert_array_equal(_mm2(L, R), _mm2(R, L))


# ─────────────────────────────────────────────────────────────────
# Direct product C2 x C3
# ─────────────────────────────────────────────────────────────────


class TestDirectProductC2xC3:
    def test_factors_detected(self, gd_c2xc3):
        assert gd_c2xc3.n == 6
        assert [f.n for f in gd_c2xc3.factors] == [2, 3]
        assert len(gd_c2xc3.decompose_table) == 6
        assert len(gd_c2xc3.compose_table) == 6
        assert sorted(gd_c2xc3.kron_perm) == list(range(6))

    def test_isomorphic_to_c6(self, gd_c2xc3):
        # C2 x C3 is cyclic of order 6: abelian with order multiset
        # [1, 2, 3, 3, 6, 6] (isomorphism invariants).
        assert gd_c2xc3.is_abelian is True
        assert sorted(element_order(g, gd_c2xc3) for g in range(6)) == \
            [1, 2, 3, 3, 6, 6]

    def test_decompose_is_group_isomorphism(self, gd_c2xc3):
        # decompose must turn G-multiplication into componentwise
        # factor-multiplication, inverses into componentwise inverses,
        # and the identity into the factor identities. Exhaustive.
        gd = gd_c2xc3
        f0, f1 = gd.factors
        for g, h in itertools.product(range(6), repeat=2):
            dg, dh = gd.decompose(g), gd.decompose(h)
            assert gd.decompose(gd.mult[g][h]) == (
                f0.mult[dg[0]][dh[0]], f1.mult[dg[1]][dh[1]])
        for g in range(6):
            dg = gd.decompose(g)
            assert gd.decompose(gd.inv[g]) == (f0.inv[dg[0]], f1.inv[dg[1]])
        assert gd.decompose(gd.identity) == (f0.identity, f1.identity)

    def test_kron_factorization_every_element(self, gd_c2xc3):
        # L_G[g] = P (L_f0[g0] kron L_f1[g1]) P^T for ALL 6 elements, with P
        # built from kron_perm per the documented definition
        # P[gap_idx, kron_perm[gap_idx]] = 1. (The constructor only samples.)
        gd = gd_c2xc3
        P = np.zeros((6, 6), dtype=np.uint8)
        for gap_idx, kron_idx in enumerate(gd.kron_perm):
            P[gap_idx, kron_idx] = 1
        for g in range(6):
            g0, g1 = gd.decompose(g)
            K = np.kron(left_rep((g0,), gd.factors[0]),
                        left_rep((g1,), gd.factors[1]))
            expected = _mm2(_mm2(P, K), P.T)
            np.testing.assert_array_equal(left_rep((g,), gd), expected)

    def test_compose_out_of_range_raises_keyerror(self, gd_c2xc3):
        with pytest.raises(KeyError):
            gd_c2xc3.compose((gd_c2xc3.factors[0].n, 0))
