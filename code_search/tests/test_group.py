"""Tests for core.group (Type 1: implementation-aware).

Requires gappy + GAP. Marked with the `gap` marker.
"""

import itertools

import numpy as np
import pytest

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
# Fixtures (session-scoped — GroupData init is O(n²) GAP calls)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="session")
def gd_c2xc2():
    return GroupData("DirectProduct(CyclicGroup(2), CyclicGroup(2))")


# ─────────────────────────────────────────────────────────────────
# GroupData
# ─────────────────────────────────────────────────────────────────


class TestGroupDataS3:
    def test_order(self, gd_s3):
        assert gd_s3.n == 6

    def test_not_abelian(self, gd_s3):
        assert gd_s3.is_abelian is False

    def test_structure_string(self, gd_s3):
        # GAP returns "S3" or "Sym(3)" — accept either form.
        assert "S3" in gd_s3.structure or "Sym(3)" in gd_s3.structure or "S_3" in gd_s3.structure

    def test_identity_is_left_unit(self, gd_s3):
        e = gd_s3.identity
        for h in range(gd_s3.n):
            assert gd_s3.mult[e][h] == h

    def test_identity_is_right_unit(self, gd_s3):
        e = gd_s3.identity
        for h in range(gd_s3.n):
            assert gd_s3.mult[h][e] == h

    def test_inverse_consistency(self, gd_s3):
        e = gd_s3.identity
        for g in range(gd_s3.n):
            assert gd_s3.mult[g][gd_s3.inv[g]] == e
            assert gd_s3.mult[gd_s3.inv[g]][g] == e

    def test_commutator_is_A3(self, gd_s3):
        # [S3, S3] = A3 has order 3.
        assert gd_s3.commutator_order == 3
        assert len(gd_s3.commutator) == 3
        assert gd_s3.identity in gd_s3.commutator

    def test_abelianization_order(self, gd_s3):
        # S3 / A3 = C2.
        assert gd_s3.abelianization_order == 2

    def test_coset_id_consistent_with_membership(self, gd_s3):
        # Two elements share a coset iff they share a coset_id.
        n = gd_s3.n
        for g1, g2 in itertools.product(range(n), repeat=2):
            same_coset = (gd_s3.mult[g1][gd_s3.inv[g2]] in gd_s3.commutator)
            assert (gd_s3.coset_id[g1] == gd_s3.coset_id[g2]) == same_coset

    def test_attribute_lengths(self, gd_s3):
        n = gd_s3.n
        assert len(gd_s3.elem_strs) == n
        assert len(gd_s3.mult) == n
        assert all(len(row) == n for row in gd_s3.mult)
        assert len(gd_s3.inv) == n
        assert len(gd_s3.coset_id) == n

    def test_gap_globals_unbound_after_init(self, gd_s3):
        # Sanity: the names should not be re-readable after construction.
        from gappy import gap
        for name in ("_G", "_elems", "_comm", "_comm_elems"):
            assert bool(gap.eval(f"IsBound({name})")) is False

    def test_identity_is_index_zero(self, gd_s3, gd_c4, gd_c2xc2):
        # Package convention: GAP's Elements(G) puts identity at position 1
        # (GAP) = index 0 (Python). Enforced in GroupData.__init__.
        assert gd_s3.identity == 0
        assert gd_c4.identity == 0
        assert gd_c2xc2.identity == 0

    def test_construction_failure_cleans_up_gap_state(self):
        # If __init__ raises mid-construction, the per-instance GAP vars must
        # be unbound — otherwise the GAP session leaks state on every failure.
        from gappy import gap
        from core.group import GroupData, _instance_counter

        # Peek the next instance id without consuming it permanently: read one,
        # then we know the failing GroupData will use this id.
        next_id_peek = next(_instance_counter)
        # Put it back: we can't, so just compute names for the id AFTER the
        # one we just consumed (= the one the failing __init__ will use).
        failing_id = next_id_peek + 1

        with pytest.raises(Exception):
            GroupData('NotARealGapFunction("nope")')

        for prefix in ("_G", "_elems", "_comm", "_comm_elems"):
            var = f"{prefix}_{failing_id}"
            assert bool(gap.eval(f'IsBoundGlobal("{var}")')) is False, (
                f"GAP variable {var} leaked after construction failure."
            )


class TestGroupDataAbelian:
    def test_c4_is_abelian(self, gd_c4):
        assert gd_c4.is_abelian is True

    def test_c4_commutator_trivial(self, gd_c4):
        assert gd_c4.commutator_order == 1
        assert gd_c4.commutator == frozenset({gd_c4.identity})

    def test_c4_abelianization_equals_order(self, gd_c4):
        assert gd_c4.abelianization_order == gd_c4.n

    def test_c4_commutes(self, gd_c4):
        n = gd_c4.n
        for i, j in itertools.product(range(n), repeat=2):
            assert gd_c4.mult[i][j] == gd_c4.mult[j][i]

    def test_c2xc2_order(self, gd_c2xc2):
        assert gd_c2xc2.n == 4
        assert gd_c2xc2.is_abelian is True


# ─────────────────────────────────────────────────────────────────
# dagger / matrix_dagger
# ─────────────────────────────────────────────────────────────────


class TestDagger:
    def test_dagger_empty(self, gd_s3):
        assert dagger((), gd_s3) == ()

    def test_dagger_identity_singleton(self, gd_s3):
        e = gd_s3.identity
        assert dagger((e,), gd_s3) == (e,)

    def test_dagger_involution(self, gd_s3):
        for x in [(0,), (0, 1), (1, 2, 3), (0, 1, 2, 3, 4, 5)]:
            assert dagger(dagger(x, gd_s3), gd_s3) == tuple(sorted(x))

    def test_dagger_returns_sorted(self, gd_s3):
        # Pass an unsorted input; result must be sorted.
        result = dagger([3, 1, 4], gd_s3)
        assert list(result) == sorted(result)

    def test_dagger_accepts_frozenset(self, gd_s3):
        result = dagger(frozenset({0, 2, 4}), gd_s3)
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)

    def test_dagger_canonicalizes_duplicates(self, gd_s3):
        # (a, a) ≡ 0 in F2[G], so its dagger must be the zero element ().
        for g in range(gd_s3.n):
            assert dagger((g, g), gd_s3) == ()

    def test_dagger_mixed_duplicates(self, gd_s3):
        # (a, a, b) ≡ (b,), so dagger((a, a, b)) must equal dagger((b,)).
        for a in range(gd_s3.n):
            for b in range(gd_s3.n):
                if a == b:
                    continue
                assert dagger((a, a, b), gd_s3) == dagger((b,), gd_s3)


class TestMatrixDagger:
    def test_shape_transposes(self, gd_s3):
        M = [[(0,), (1,), (2,)],
             [(3,), (4,), (5,)]]
        Md = matrix_dagger(M, gd_s3)
        assert len(Md) == 3
        assert len(Md[0]) == 2

    def test_involution(self, gd_s3):
        M = [[(0,), (1,)],
             [(2, 3), (4, 5)]]
        assert matrix_dagger(matrix_dagger(M, gd_s3), gd_s3) == M


# ─────────────────────────────────────────────────────────────────
# element_order / is_self_dagger
# ─────────────────────────────────────────────────────────────────


class TestElementOrder:
    def test_identity_order_is_one(self, gd_s3):
        assert element_order(gd_s3.identity, gd_s3) == 1

    def test_order_divides_group_order(self, gd_s3):
        for g in range(gd_s3.n):
            assert gd_s3.n % element_order(g, gd_s3) == 0

    def test_inverse_has_same_order(self, gd_s3):
        for g in range(gd_s3.n):
            assert element_order(g, gd_s3) == element_order(gd_s3.inv[g], gd_s3)

    def test_s3_orders_match_known(self, gd_s3):
        # S3 has 1 element of order 1, 3 of order 2, 2 of order 3.
        orders = [element_order(g, gd_s3) for g in range(gd_s3.n)]
        assert sorted(orders) == [1, 2, 2, 2, 3, 3]


class TestIsSelfDagger:
    def test_empty(self, gd_s3):
        assert is_self_dagger((), gd_s3) is True

    def test_identity_singleton(self, gd_s3):
        assert is_self_dagger((gd_s3.identity,), gd_s3) is True

    def test_consistent_with_dagger(self, gd_s3):
        for x in [(0,), (0, 1), (1, 2, 3), (0, 3, 5)]:
            expected = (dagger(x, gd_s3) == tuple(sorted(x)))
            assert is_self_dagger(x, gd_s3) == expected

    def test_canonicalizes_input(self, gd_s3):
        # (g, g, h) ≡ (h,) mod 2, so the result must match is_self_dagger((h,)).
        e = gd_s3.identity
        for h in range(gd_s3.n):
            for g in range(gd_s3.n):
                if g == h:
                    continue
                assert is_self_dagger((g, g, h), gd_s3) == is_self_dagger((h,), gd_s3)
        # Even-count duplicates alone collapse to zero, which is self-dagger.
        assert is_self_dagger((e, e, 1, 1), gd_s3) is True


# ─────────────────────────────────────────────────────────────────
# Ring arithmetic
# ─────────────────────────────────────────────────────────────────


class TestRingAdd:
    def test_add_empty(self):
        assert ring_add((), ()) == ()
        assert ring_add((), (1, 2)) == (1, 2)

    def test_self_cancels(self):
        assert ring_add((1, 2, 3), (1, 2, 3)) == ()

    def test_partial_overlap(self):
        assert ring_add((1, 2), (2, 3)) == (1, 3)

    def test_commutative(self):
        x, y = (1, 3, 5), (2, 3, 7)
        assert ring_add(x, y) == ring_add(y, x)

    def test_associative(self):
        x, y, z = (1, 2), (2, 3), (3, 4)
        assert ring_add(ring_add(x, y), z) == ring_add(x, ring_add(y, z))

    def test_sorted_output(self):
        result = ring_add((3, 1), (4,))
        assert list(result) == sorted(result)

    def test_duplicates_in_one_operand_cancel(self):
        # (1, 1) ≡ 0 in F2[G], so ring_add((1, 1), ()) must be ().
        # Pre-fix this returned (1,) because set() silently de-duped.
        assert ring_add((1, 1), ()) == ()
        assert ring_add((), (1, 1)) == ()

    def test_duplicates_across_operands_cancel(self):
        # Index 2: count 1 in lhs + count 2 in rhs = 3 → survives.
        # Index 1: count 2 in lhs + count 0 in rhs = 2 → cancels.
        assert ring_add((1, 1, 2), (2, 2)) == (2,)


class TestCanonicalize:
    def test_empty(self):
        assert canonicalize(()) == ()

    def test_already_canonical_is_identity(self):
        assert canonicalize((1, 3, 5)) == (1, 3, 5)

    def test_sorts(self):
        assert canonicalize((5, 1, 3)) == (1, 3, 5)

    def test_even_duplicates_cancel(self):
        assert canonicalize((1, 1)) == ()
        assert canonicalize((1, 2, 1, 2)) == ()

    def test_odd_duplicates_survive(self):
        assert canonicalize((1, 1, 1)) == (1,)
        assert canonicalize((1, 2, 1)) == (2,)

    def test_idempotent(self):
        x = (3, 1, 1, 2, 2, 2, 5)
        assert canonicalize(canonicalize(x)) == canonicalize(x)

    def test_accepts_generator(self):
        assert canonicalize(g for g in (1, 1, 2)) == (2,)


class TestRingMul:
    def test_zero_times_anything(self, gd_s3):
        assert ring_mul((), (1, 2, 3), gd_s3) == ()
        assert ring_mul((1, 2, 3), (), gd_s3) == ()

    def test_identity_is_unit_left(self, gd_s3):
        e = gd_s3.identity
        for x in [(0,), (1, 2), (1, 2, 3)]:
            assert ring_mul((e,), x, gd_s3) == tuple(sorted(set(x)))

    def test_identity_is_unit_right(self, gd_s3):
        e = gd_s3.identity
        for x in [(0,), (1, 2), (1, 2, 3)]:
            assert ring_mul(x, (e,), gd_s3) == tuple(sorted(set(x)))

    def test_distributive_left(self, gd_s3):
        x, y, z = (1,), (2, 3), (3, 4)
        lhs = ring_mul(x, ring_add(y, z), gd_s3)
        rhs = ring_add(ring_mul(x, y, gd_s3), ring_mul(x, z, gd_s3))
        assert lhs == rhs

    def test_distributive_right(self, gd_s3):
        x, y, z = (1, 2), (3,), (3, 4)
        lhs = ring_mul(ring_add(x, y), z, gd_s3)
        rhs = ring_add(ring_mul(x, z, gd_s3), ring_mul(y, z, gd_s3))
        assert lhs == rhs

    def test_commutes_iff_abelian(self, gd_s3, gd_c4):
        # S3: there exist x, y with ring_mul(x, y) != ring_mul(y, x).
        found_noncommute = False
        for g1, g2 in itertools.product(range(gd_s3.n), repeat=2):
            if ring_mul((g1,), (g2,), gd_s3) != ring_mul((g2,), (g1,), gd_s3):
                found_noncommute = True
                break
        assert found_noncommute

        # C4: every pair commutes.
        for g1, g2 in itertools.product(range(gd_c4.n), repeat=2):
            assert ring_mul((g1,), (g2,), gd_c4) == ring_mul((g2,), (g1,), gd_c4)


class TestRingPermanent:
    def test_1x1(self, gd_s3):
        assert ring_permanent([[(1, 2)]], gd_s3) == (1, 2)

    def test_2x2_diagonal(self, gd_s3):
        # [[x, 0], [0, y]] — only the identity permutation contributes:  x · y.
        # (transposition contributes 0 · 0 = 0)
        x, y = (1,), (2, 3)
        M = [[x, ()],
             [(), y]]
        assert ring_permanent(M, gd_s3) == ring_mul(x, y, gd_s3)

    def test_2x2_off_diagonal(self, gd_s3):
        # [[0, y], [x, 0]] — only transposition contributes:  y · x.
        x, y = (1,), (2, 3)
        M = [[(), y],
             [x, ()]]
        # ring_permanent = M[0][1] * M[1][0] = y * x
        assert ring_permanent(M, gd_s3) == ring_mul(y, x, gd_s3)


# ─────────────────────────────────────────────────────────────────
# Binary representations
# ─────────────────────────────────────────────────────────────────


def _eye_uint8(n):
    return np.eye(n, dtype=np.uint8)


def _matmul_mod2(A, B):
    return ((A.astype(np.int64) @ B.astype(np.int64)) & 1).astype(np.uint8)


class TestLeftRep:
    def test_zero(self, gd_s3):
        L0 = left_rep((), gd_s3)
        np.testing.assert_array_equal(L0, np.zeros((gd_s3.n, gd_s3.n), dtype=np.uint8))

    def test_identity(self, gd_s3):
        Le = left_rep((gd_s3.identity,), gd_s3)
        np.testing.assert_array_equal(Le, _eye_uint8(gd_s3.n))

    def test_homomorphism_singleton(self, gd_s3):
        # L[g1·g2] = L[g1] @ L[g2] mod 2.
        for g1, g2 in itertools.product(range(gd_s3.n), repeat=2):
            gg = gd_s3.mult[g1][g2]
            lhs = left_rep((gg,), gd_s3)
            rhs = _matmul_mod2(left_rep((g1,), gd_s3), left_rep((g2,), gd_s3))
            np.testing.assert_array_equal(lhs, rhs)

    def test_linear(self, gd_s3):
        x, y = (1, 2), (2, 3)
        lhs = left_rep(ring_add(x, y), gd_s3)
        rhs = (left_rep(x, gd_s3) ^ left_rep(y, gd_s3))
        np.testing.assert_array_equal(lhs, rhs)

    def test_transpose_is_dagger(self, gd_s3):
        for x in [(0,), (1,), (1, 2), (1, 2, 3)]:
            np.testing.assert_array_equal(
                left_rep(x, gd_s3).T,
                left_rep(dagger(x, gd_s3), gd_s3),
            )


class TestRightRep:
    def test_zero(self, gd_s3):
        R0 = right_rep((), gd_s3)
        np.testing.assert_array_equal(R0, np.zeros((gd_s3.n, gd_s3.n), dtype=np.uint8))

    def test_identity(self, gd_s3):
        Re = right_rep((gd_s3.identity,), gd_s3)
        np.testing.assert_array_equal(Re, _eye_uint8(gd_s3.n))

    def test_homomorphism_singleton(self, gd_s3):
        # R[g1·g2] = R[g1] @ R[g2] mod 2.
        for g1, g2 in itertools.product(range(gd_s3.n), repeat=2):
            gg = gd_s3.mult[g1][g2]
            lhs = right_rep((gg,), gd_s3)
            rhs = _matmul_mod2(right_rep((g1,), gd_s3), right_rep((g2,), gd_s3))
            np.testing.assert_array_equal(lhs, rhs)

    def test_transpose_is_dagger(self, gd_s3):
        for x in [(0,), (1,), (1, 2), (1, 2, 3)]:
            np.testing.assert_array_equal(
                right_rep(x, gd_s3).T,
                right_rep(dagger(x, gd_s3), gd_s3),
            )


class TestLeftRightCommute:
    def test_singletons_commute(self, gd_s3):
        for x, y in itertools.product(range(gd_s3.n), repeat=2):
            lhs = _matmul_mod2(left_rep((x,), gd_s3), right_rep((y,), gd_s3))
            rhs = _matmul_mod2(right_rep((y,), gd_s3), left_rep((x,), gd_s3))
            np.testing.assert_array_equal(lhs, rhs)

    def test_general_elements_commute(self, gd_s3):
        x, y = (1, 2, 3), (0, 4, 5)
        lhs = _matmul_mod2(left_rep(x, gd_s3), right_rep(y, gd_s3))
        rhs = _matmul_mod2(right_rep(y, gd_s3), left_rep(x, gd_s3))
        np.testing.assert_array_equal(lhs, rhs)


# ─────────────────────────────────────────────────────────────────
# Direct product decomposition
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_s3_x_c4():
    # Auto-detected via DirectProductInfo.
    return GroupData("DirectProduct(SymmetricGroup(3), CyclicGroup(4))")


class TestDirectProductAttributes:
    def test_order_24(self, gd_s3_x_c4):
        assert gd_s3_x_c4.n == 24

    def test_two_factors(self, gd_s3_x_c4):
        assert len(gd_s3_x_c4.factors) == 2

    def test_factor_sizes(self, gd_s3_x_c4):
        sizes = sorted(f.n for f in gd_s3_x_c4.factors)
        assert sizes == [4, 6]

    def test_factor_size_product_equals_n(self, gd_s3_x_c4):
        from math import prod
        assert prod(f.n for f in gd_s3_x_c4.factors) == gd_s3_x_c4.n

    def test_decompose_table_length(self, gd_s3_x_c4):
        assert len(gd_s3_x_c4.decompose_table) == gd_s3_x_c4.n

    def test_compose_table_is_bijection(self, gd_s3_x_c4):
        assert len(gd_s3_x_c4.compose_table) == gd_s3_x_c4.n

    def test_kron_perm_is_permutation(self, gd_s3_x_c4):
        assert sorted(gd_s3_x_c4.kron_perm) == list(range(gd_s3_x_c4.n))


class TestDirectProductDecomposeCompose:
    def test_round_trip_decompose_compose(self, gd_s3_x_c4):
        for g in range(gd_s3_x_c4.n):
            indices = gd_s3_x_c4.decompose(g)
            assert gd_s3_x_c4.compose(indices) == g

    def test_round_trip_compose_decompose(self, gd_s3_x_c4):
        gd = gd_s3_x_c4
        for i1 in range(gd.factors[0].n):
            for i2 in range(gd.factors[1].n):
                g = gd.compose((i1, i2))
                assert gd.decompose(g) == (i1, i2)

    def test_decompose_identity(self, gd_s3_x_c4):
        gd = gd_s3_x_c4
        indices = gd.decompose(gd.identity)
        for i, idx in enumerate(indices):
            assert idx == gd.factors[i].identity

    def test_compose_identity_tuple(self, gd_s3_x_c4):
        gd = gd_s3_x_c4
        identity_tuple = tuple(f.identity for f in gd.factors)
        assert gd.compose(identity_tuple) == gd.identity

    def test_decompose_indices_in_range(self, gd_s3_x_c4):
        gd = gd_s3_x_c4
        for g in range(gd.n):
            indices = gd.decompose(g)
            assert len(indices) == len(gd.factors)
            for i, idx in enumerate(indices):
                assert 0 <= idx < gd.factors[i].n


class TestNoDirectProduct:
    def test_simple_group_has_no_decomposition(self, gd_s3):
        # S3 is not a direct product (DirectProductInfo not set).
        assert gd_s3.factors is None
        assert gd_s3.decompose_table is None
        assert gd_s3.compose_table is None
        assert gd_s3.kron_perm is None

    def test_decompose_raises_when_no_structure(self, gd_s3):
        with pytest.raises(RuntimeError, match="no direct-product decomposition"):
            gd_s3.decompose(0)

    def test_compose_raises_when_no_structure(self, gd_s3):
        with pytest.raises(RuntimeError, match="no direct-product decomposition"):
            gd_s3.compose((0,))
