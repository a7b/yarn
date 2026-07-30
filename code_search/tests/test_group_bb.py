"""Tests for core.group (Type 2: black-box / contract).

Reads only interface.md, tests core.group functions against their
documented invariants without reading the implementation.
"""

import numpy as np
import pytest
from gappy import gap

from core.group import (
    GroupData,
    dagger,
    matrix_dagger,
    element_order,
    is_self_dagger,
    ring_add,
    ring_mul,
    ring_permanent,
    left_rep,
    right_rep,
)


pytestmark = pytest.mark.gap


# Session-scoped fixtures for GroupData instances
@pytest.fixture(scope="session")
def gd_s3():
    """S₃: non-abelian group, order 6, commutator subgroup is A₃ (order 3)."""
    gd = GroupData("SymmetricGroup(3)")
    yield gd


@pytest.fixture(scope="session")
def gd_c4():
    """C₄: cyclic abelian group, order 4."""
    gd = GroupData("CyclicGroup(4)")
    yield gd


@pytest.fixture(scope="session")
def gd_d4():
    """D₄: dihedral group of the square (order 8), non-abelian.

    GAP's `DihedralGroup(n)` takes the *order*, not the polygon size; the
    mathematical D₄ (symmetries of a 4-gon) has order 8, so we pass 8 here.
    """
    gd = GroupData("DihedralGroup(8)")
    yield gd


# Tests for GroupData attributes and invariants

class TestGroupDataAttributes:
    """Verify GroupData structure, types, and basic invariants."""

    def test_s3_order(self, gd_s3):
        assert gd_s3.n == 6
        assert len(gd_s3.elem_strs) == 6
        assert len(gd_s3.mult) == 6
        assert len(gd_s3.inv) == 6
        assert len(gd_s3.coset_id) == 6

    def test_c4_order(self, gd_c4):
        assert gd_c4.n == 4
        assert len(gd_c4.elem_strs) == 4
        assert len(gd_c4.mult) == 4
        assert len(gd_c4.inv) == 4
        assert len(gd_c4.coset_id) == 4

    def test_d4_order(self, gd_d4):
        assert gd_d4.n == 8
        assert len(gd_d4.elem_strs) == 8
        assert len(gd_d4.mult) == 8
        assert len(gd_d4.inv) == 8
        assert len(gd_d4.coset_id) == 8

    def test_identity_is_valid_index(self, gd_s3, gd_c4, gd_d4):
        """Identity is a valid 0-based index into the group."""
        assert 0 <= gd_s3.identity < gd_s3.n
        assert 0 <= gd_c4.identity < gd_c4.n
        assert 0 <= gd_d4.identity < gd_d4.n

    def test_gap_expr_stored(self, gd_s3, gd_c4, gd_d4):
        """Gap expression is preserved."""
        assert gd_s3.gap_expr == "SymmetricGroup(3)"
        assert gd_c4.gap_expr == "CyclicGroup(4)"
        assert gd_d4.gap_expr == "DihedralGroup(8)"

    def test_is_abelian_correct(self, gd_s3, gd_c4, gd_d4):
        """S₃ and D₄ are non-abelian; C₄ is abelian."""
        assert not gd_s3.is_abelian
        assert gd_c4.is_abelian
        assert not gd_d4.is_abelian

    def test_structure_string_present(self, gd_s3, gd_c4, gd_d4):
        """structure attribute is a non-empty string."""
        assert isinstance(gd_s3.structure, str)
        assert isinstance(gd_c4.structure, str)
        assert isinstance(gd_d4.structure, str)
        assert len(gd_s3.structure) > 0
        assert len(gd_c4.structure) > 0
        assert len(gd_d4.structure) > 0

    def test_commutator_order_abelian(self, gd_c4):
        """For abelian groups, commutator_order is 1."""
        assert gd_c4.commutator_order == 1

    def test_commutator_contains_identity(self, gd_s3, gd_c4, gd_d4):
        """Commutator subgroup always contains the identity."""
        assert gd_s3.identity in gd_s3.commutator
        assert gd_c4.identity in gd_c4.commutator
        assert gd_d4.identity in gd_d4.commutator

    def test_commutator_order_matches_frozenset_size(self, gd_s3, gd_d4):
        """Commutator order equals the size of commutator frozenset."""
        assert gd_s3.commutator_order == len(gd_s3.commutator)
        assert gd_d4.commutator_order == len(gd_d4.commutator)

    def test_abelianization_order_formula(self, gd_s3, gd_c4, gd_d4):
        """abelianization_order = n / commutator_order."""
        assert gd_s3.abelianization_order == gd_s3.n // gd_s3.commutator_order
        assert gd_c4.abelianization_order == gd_c4.n // gd_c4.commutator_order
        assert gd_d4.abelianization_order == gd_d4.n // gd_d4.commutator_order

    def test_commutator_single_for_abelian(self, gd_c4):
        """For abelian group, commutator is exactly {identity}."""
        assert gd_c4.commutator == frozenset({gd_c4.identity})

    def test_coset_id_valid_indices(self, gd_s3, gd_c4, gd_d4):
        """Each coset_id[g] is a valid group element index."""
        for g, cid in enumerate(gd_s3.coset_id):
            assert 0 <= cid < gd_s3.n
        for g, cid in enumerate(gd_c4.coset_id):
            assert 0 <= cid < gd_c4.n
        for g, cid in enumerate(gd_d4.coset_id):
            assert 0 <= cid < gd_d4.n

    def test_gap_globals_unbound(self, gd_s3):
        """After construction, GAP globals are unbound."""
        # Verify that _G, _elems, _comm, _comm_elems are not defined in GAP
        assert not gap.eval("IsBound(_G)")
        assert not gap.eval("IsBound(_elems)")
        assert not gap.eval("IsBound(_comm)")
        assert not gap.eval("IsBound(_comm_elems)")


class TestMultiplicationTable:
    """Test the mult table invariants."""

    def test_mult_shape(self, gd_s3, gd_c4):
        """mult is n×n."""
        assert len(gd_s3.mult) == gd_s3.n
        assert all(len(row) == gd_s3.n for row in gd_s3.mult)
        assert len(gd_c4.mult) == gd_c4.n
        assert all(len(row) == gd_c4.n for row in gd_c4.mult)

    def test_mult_values_in_range(self, gd_s3, gd_c4):
        """All mult entries are valid indices."""
        for i in range(gd_s3.n):
            for j in range(gd_s3.n):
                assert 0 <= gd_s3.mult[i][j] < gd_s3.n
        for i in range(gd_c4.n):
            for j in range(gd_c4.n):
                assert 0 <= gd_c4.mult[i][j] < gd_c4.n

    def test_identity_is_left_identity(self, gd_s3, gd_c4, gd_d4):
        """mult[identity][h] == h for all h."""
        for h in range(gd_s3.n):
            assert gd_s3.mult[gd_s3.identity][h] == h
        for h in range(gd_c4.n):
            assert gd_c4.mult[gd_c4.identity][h] == h
        for h in range(gd_d4.n):
            assert gd_d4.mult[gd_d4.identity][h] == h

    def test_identity_is_right_identity(self, gd_s3, gd_c4, gd_d4):
        """mult[h][identity] == h for all h."""
        for h in range(gd_s3.n):
            assert gd_s3.mult[h][gd_s3.identity] == h
        for h in range(gd_c4.n):
            assert gd_c4.mult[h][gd_c4.identity] == h
        for h in range(gd_d4.n):
            assert gd_d4.mult[h][gd_d4.identity] == h

    def test_commutativity_abelian(self, gd_c4):
        """For abelian group, mult[i][j] == mult[j][i]."""
        for i in range(gd_c4.n):
            for j in range(gd_c4.n):
                assert gd_c4.mult[i][j] == gd_c4.mult[j][i]

    def test_non_commutativity_nonabelian(self, gd_s3):
        """For non-abelian S₃, there exist i, j with mult[i][j] != mult[j][i]."""
        found_noncommuting = False
        for i in range(gd_s3.n):
            for j in range(gd_s3.n):
                if gd_s3.mult[i][j] != gd_s3.mult[j][i]:
                    found_noncommuting = True
                    break
            if found_noncommuting:
                break
        assert found_noncommuting


class TestInversionTable:
    """Test the inv table invariants."""

    def test_inv_valid_indices(self, gd_s3, gd_c4, gd_d4):
        """All inv entries are valid indices."""
        for i in range(gd_s3.n):
            assert 0 <= gd_s3.inv[i] < gd_s3.n
        for i in range(gd_c4.n):
            assert 0 <= gd_c4.inv[i] < gd_c4.n
        for i in range(gd_d4.n):
            assert 0 <= gd_d4.inv[i] < gd_d4.n

    def test_mult_with_inv_right(self, gd_s3, gd_c4, gd_d4):
        """mult[i][inv[i]] == identity for all i."""
        for i in range(gd_s3.n):
            assert gd_s3.mult[i][gd_s3.inv[i]] == gd_s3.identity
        for i in range(gd_c4.n):
            assert gd_c4.mult[i][gd_c4.inv[i]] == gd_c4.identity
        for i in range(gd_d4.n):
            assert gd_d4.mult[i][gd_d4.inv[i]] == gd_d4.identity

    def test_mult_with_inv_left(self, gd_s3, gd_c4, gd_d4):
        """mult[inv[i]][i] == identity for all i."""
        for i in range(gd_s3.n):
            assert gd_s3.mult[gd_s3.inv[i]][i] == gd_s3.identity
        for i in range(gd_c4.n):
            assert gd_c4.mult[gd_c4.inv[i]][i] == gd_c4.identity
        for i in range(gd_d4.n):
            assert gd_d4.mult[gd_d4.inv[i]][i] == gd_d4.identity

    def test_inv_involution(self, gd_s3, gd_c4):
        """inv[inv[i]] == i (involution property)."""
        for i in range(gd_s3.n):
            assert gd_s3.inv[gd_s3.inv[i]] == i
        for i in range(gd_c4.n):
            assert gd_c4.inv[gd_c4.inv[i]] == i

    def test_identity_is_self_inverse(self, gd_s3, gd_c4, gd_d4):
        """inv[identity] == identity."""
        assert gd_s3.inv[gd_s3.identity] == gd_s3.identity
        assert gd_c4.inv[gd_c4.identity] == gd_c4.identity
        assert gd_d4.inv[gd_d4.identity] == gd_d4.identity


# Tests for dagger

class TestDagger:
    """Test the dagger operation on ring elements."""

    def test_dagger_empty(self, gd_s3):
        """dagger((), gd) == ()."""
        assert dagger((), gd_s3) == ()

    def test_dagger_identity(self, gd_s3):
        """dagger((identity,), gd) == (identity,)."""
        assert dagger((gd_s3.identity,), gd_s3) == (gd_s3.identity,)

    def test_dagger_involution(self, gd_s3, gd_c4):
        """dagger(dagger(x, gd), gd) == tuple(sorted(x))."""
        test_cases = [
            (),
            (gd_s3.identity,),
            (0, 1),
            (1, 2, 3),
        ]
        for x in test_cases:
            result = dagger(dagger(x, gd_s3), gd_s3)
            expected = tuple(sorted(x))
            assert result == expected

        test_cases_c4 = [
            (),
            (gd_c4.identity,),
            (0, 1),
            (1, 2),
        ]
        for x in test_cases_c4:
            result = dagger(dagger(x, gd_c4), gd_c4)
            expected = tuple(sorted(x))
            assert result == expected

    def test_dagger_returns_sorted_tuple(self, gd_s3, gd_d4):
        """dagger returns a sorted tuple."""
        # Test with various inputs
        for x in [(2, 0, 1), (5, 3), (4, 2, 1, 0)]:
            result = dagger(x, gd_s3)
            assert isinstance(result, tuple)
            assert list(result) == sorted(result)

        for x in [(3, 1, 2), (5, 4)]:
            result = dagger(x, gd_d4)
            assert isinstance(result, tuple)
            assert list(result) == sorted(result)

    def test_dagger_list_input(self, gd_s3):
        """dagger accepts list input."""
        x = [1, 3, 2]
        result = dagger(x, gd_s3)
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)

    def test_dagger_frozenset_input(self, gd_s3):
        """dagger accepts frozenset input."""
        x = frozenset({1, 3, 2})
        result = dagger(x, gd_s3)
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)


# Tests for matrix_dagger

class TestMatrixDagger:
    """Test matrix dagger (transpose + dagger each entry)."""

    def test_matrix_dagger_involution(self, gd_c4):
        """matrix_dagger(matrix_dagger(M, gd), gd) == M."""
        # Test with a 2x3 matrix of ring elements
        M = [
            [(0, 1), (2,), ()],
            [(), (1, 2), (0,)],
        ]
        result = matrix_dagger(matrix_dagger(M, gd_c4), gd_c4)
        assert result == M

    def test_matrix_dagger_shape_transpose(self, gd_c4):
        """matrix_dagger of ma×na matrix is na×ma."""
        M = [
            [(0,), (1, 2)],
            [(2,), ()],
            [(), (3,)],
        ]  # 3x2
        result = matrix_dagger(M, gd_c4)
        assert len(result) == 2
        assert all(len(row) == 3 for row in result)

    def test_matrix_dagger_1x1(self, gd_s3):
        """matrix_dagger of 1x1 matrix."""
        M = [[(0, 1)]]
        result = matrix_dagger(M, gd_s3)
        assert len(result) == 1
        assert len(result[0]) == 1
        # Check that entry is daggered
        assert result[0][0] == dagger((0, 1), gd_s3)


# Tests for element_order

class TestElementOrder:
    """Test element_order function."""

    def test_identity_order_is_one(self, gd_s3, gd_c4, gd_d4):
        """element_order(identity, gd) == 1."""
        assert element_order(gd_s3.identity, gd_s3) == 1
        assert element_order(gd_c4.identity, gd_c4) == 1
        assert element_order(gd_d4.identity, gd_d4) == 1

    def test_order_divides_group_order(self, gd_s3, gd_c4, gd_d4):
        """element_order(g, gd) divides gd.n."""
        for g in range(gd_s3.n):
            order = element_order(g, gd_s3)
            assert gd_s3.n % order == 0

        for g in range(gd_c4.n):
            order = element_order(g, gd_c4)
            assert gd_c4.n % order == 0

        for g in range(gd_d4.n):
            order = element_order(g, gd_d4)
            assert gd_d4.n % order == 0

    def test_order_of_inverse_equals_order(self, gd_s3, gd_c4, gd_d4):
        """element_order(inv[g], gd) == element_order(g, gd)."""
        for g in range(gd_s3.n):
            assert element_order(gd_s3.inv[g], gd_s3) == element_order(g, gd_s3)

        for g in range(gd_c4.n):
            assert element_order(gd_c4.inv[g], gd_c4) == element_order(g, gd_c4)

        for g in range(gd_d4.n):
            assert element_order(gd_d4.inv[g], gd_d4) == element_order(g, gd_d4)

    def test_c4_cyclic_orders(self, gd_c4):
        """C₄ element orders bounded by group order."""
        for g in range(gd_c4.n):
            order = element_order(g, gd_c4)
            assert 1 <= order <= gd_c4.n


# Tests for is_self_dagger

class TestIsSelfDagger:
    """Test is_self_dagger function."""

    def test_empty_is_self_dagger(self, gd_s3):
        """is_self_dagger((), gd) == True."""
        assert is_self_dagger((), gd_s3) is True

    def test_identity_is_self_dagger(self, gd_s3, gd_c4):
        """is_self_dagger((identity,), gd) == True."""
        assert is_self_dagger((gd_s3.identity,), gd_s3) is True
        assert is_self_dagger((gd_c4.identity,), gd_c4) is True

    def test_consistency_with_dagger(self, gd_s3, gd_c4):
        """is_self_dagger(x, gd) iff x == dagger(x, gd)."""
        test_cases_s3 = [(), (0,), (1, 2), (0, 1, 2)]
        for x in test_cases_s3:
            result = is_self_dagger(x, gd_s3)
            expected = (tuple(sorted(x)) == dagger(x, gd_s3))
            assert result == expected

        test_cases_c4 = [(), (0,), (1, 2), (0, 1)]
        for x in test_cases_c4:
            result = is_self_dagger(x, gd_c4)
            expected = (tuple(sorted(x)) == dagger(x, gd_c4))
            assert result == expected


# Tests for ring_add

class TestRingAdd:
    """Test ring addition (symmetric difference)."""

    def test_ring_add_commutative(self):
        """ring_add(x, y) == ring_add(y, x)."""
        test_cases = [
            ((), ()),
            ((0,), (1,)),
            ((0, 1), (2, 3)),
        ]
        for x, y in test_cases:
            assert ring_add(x, y) == ring_add(y, x)

    def test_ring_add_associative(self):
        """ring_add(ring_add(x, y), z) == ring_add(x, ring_add(y, z))."""
        x, y, z = (0,), (1, 2), (3,)
        left = ring_add(ring_add(x, y), z)
        right = ring_add(x, ring_add(y, z))
        assert left == right

    def test_ring_add_self_cancels(self):
        """ring_add(x, x) == ()."""
        test_cases = [(), (0,), (1, 2), (0, 1, 2)]
        for x in test_cases:
            assert ring_add(x, x) == ()

    def test_ring_add_empty_identity(self):
        """ring_add(x, ()) == tuple(sorted(set(x)))."""
        test_cases = [(), (0,), (1, 2), (0, 1, 2)]
        for x in test_cases:
            result = ring_add(x, ())
            expected = tuple(sorted(set(x)))
            assert result == expected

    def test_ring_add_returns_sorted_tuple(self):
        """ring_add always returns sorted tuple."""
        x, y = (2, 0), (3, 1)
        result = ring_add(x, y)
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)


# Tests for ring_mul

class TestRingMul:
    """Test ring multiplication."""

    def test_ring_mul_zero_left(self, gd_s3, gd_c4):
        """ring_mul((), x, gd) == ()."""
        test_cases = [(), (0,), (1, 2)]
        for x in test_cases:
            assert ring_mul((), x, gd_s3) == ()
            assert ring_mul((), x, gd_c4) == ()

    def test_ring_mul_zero_right(self, gd_s3):
        """ring_mul(x, (), gd) == ()."""
        test_cases = [(), (0,), (1, 2)]
        for x in test_cases:
            assert ring_mul(x, (), gd_s3) == ()

    def test_ring_mul_identity_left(self, gd_s3, gd_c4):
        """ring_mul((identity,), x, gd) == tuple(sorted(set(x)))."""
        test_cases = [(), (0,), (1, 2)]
        for x in test_cases:
            result_s3 = ring_mul((gd_s3.identity,), x, gd_s3)
            expected = tuple(sorted(set(x)))
            assert result_s3 == expected

            result_c4 = ring_mul((gd_c4.identity,), x, gd_c4)
            assert result_c4 == expected

    def test_ring_mul_identity_right(self, gd_s3):
        """ring_mul(x, (identity,), gd) == tuple(sorted(set(x)))."""
        test_cases = [(), (0,), (1, 2)]
        for x in test_cases:
            result = ring_mul(x, (gd_s3.identity,), gd_s3)
            expected = tuple(sorted(set(x)))
            assert result == expected

    def test_ring_mul_distributive(self, gd_s3):
        """ring_mul(x, ring_add(y, z), gd) == ring_add(ring_mul(x, y, gd), ring_mul(x, z, gd))."""
        x, y, z = (0,), (1,), (2,)
        left = ring_mul(x, ring_add(y, z), gd_s3)
        right = ring_add(ring_mul(x, y, gd_s3), ring_mul(x, z, gd_s3))
        assert left == right

    def test_ring_mul_commutative_abelian(self, gd_c4):
        """For abelian group, ring_mul(x, y, gd) == ring_mul(y, x, gd)."""
        x, y = (0, 1), (1, 2)
        assert ring_mul(x, y, gd_c4) == ring_mul(y, x, gd_c4)

    def test_ring_mul_nonabelian_may_not_commute(self, gd_s3):
        """For non-abelian group, the result is still a well-formed sorted tuple."""
        x, y = (1,), (2,)
        result = ring_mul(x, y, gd_s3)
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)


# Tests for ring_permanent

class TestRingPermanent:
    """Test ring permanent function."""

    def test_ring_permanent_1x1(self, gd_c4):
        """For a 1x1 matrix [[x]], perm = x."""
        M = [[(1, 2)]]
        result = ring_permanent(M, gd_c4)
        assert result == (1, 2)

    def test_ring_permanent_2x2_formula(self, gd_c4):
        """2x2 permanent = a₁₁·a₂₂ + a₁₂·a₂₁ (over F₂[G])."""
        M = [
            [(0,), (1,)],
            [(2,), (3,)],
        ]
        # permanent = (0,)·(3,) + (1,)·(2,)
        result = ring_permanent(M, gd_c4)
        expected_term1 = ring_mul((0,), (3,), gd_c4)
        expected_term2 = ring_mul((1,), (2,), gd_c4)
        expected = ring_add(expected_term1, expected_term2)
        assert result == expected

    def test_ring_permanent_3x3_sanity(self, gd_c4):
        """3x3 permanent returns a valid sorted-tuple ring element."""
        M = [
            [(0,), (1,), (2,)],
            [(1,), (0,), (3,)],
            [(2,), (3,), (0,)],
        ]
        result = ring_permanent(M, gd_c4)
        # Just verify it's a valid ring element
        assert isinstance(result, tuple)
        assert list(result) == sorted(result)


# Tests for left_rep and right_rep

class TestLeftRep:
    """Test left regular representation."""

    def test_left_rep_zero(self, gd_s3):
        """left_rep((), gd) is the zero matrix."""
        result = left_rep((), gd_s3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (gd_s3.n, gd_s3.n)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros((gd_s3.n, gd_s3.n), dtype=np.uint8))

    def test_left_rep_identity(self, gd_s3, gd_c4):
        """left_rep((identity,), gd) is the identity matrix."""
        result_s3 = left_rep((gd_s3.identity,), gd_s3)
        np.testing.assert_array_equal(result_s3, np.eye(gd_s3.n, dtype=np.uint8))

        result_c4 = left_rep((gd_c4.identity,), gd_c4)
        np.testing.assert_array_equal(result_c4, np.eye(gd_c4.n, dtype=np.uint8))

    def test_left_rep_homomorphism(self, gd_c4):
        """left_rep((g_i · g_j,), gd) == left_rep((g_i,), gd) @ left_rep((g_j,), gd) mod 2."""
        # Pick two elements
        g_i, g_j = 1, 2
        g_product = gd_c4.mult[g_i][g_j]

        L_i = left_rep((g_i,), gd_c4)
        L_j = left_rep((g_j,), gd_c4)
        L_product = left_rep((g_product,), gd_c4)

        result = (L_i @ L_j) % 2
        np.testing.assert_array_equal(result, L_product)

    def test_left_rep_linear(self, gd_s3):
        """left_rep(ring_add(x, y), gd) == left_rep(x, gd) XOR left_rep(y, gd)."""
        x, y = (0, 1), (1, 2)
        L_x = left_rep(x, gd_s3)
        L_y = left_rep(y, gd_s3)
        L_sum = left_rep(ring_add(x, y), gd_s3)

        result = (L_x ^ L_y)  # XOR in numpy
        np.testing.assert_array_equal(result, L_sum)

    def test_left_rep_transpose_dagger(self, gd_s3):
        """left_rep(x, gd).T == left_rep(dagger(x, gd), gd)."""
        x = (0, 1, 2)
        L_x = left_rep(x, gd_s3)
        L_dag_x = left_rep(dagger(x, gd_s3), gd_s3)

        np.testing.assert_array_equal(L_x.T, L_dag_x)

    def test_left_rep_shape_and_dtype(self, gd_s3, gd_c4, gd_d4):
        """left_rep always returns n×n uint8 matrix."""
        for x in [(), (0,), (1, 2)]:
            for gd in [gd_s3, gd_c4, gd_d4]:
                result = left_rep(x, gd)
                assert result.shape == (gd.n, gd.n)
                assert result.dtype == np.uint8


class TestRightRep:
    """Test right regular representation."""

    def test_right_rep_zero(self, gd_s3):
        """right_rep((), gd) is the zero matrix."""
        result = right_rep((), gd_s3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (gd_s3.n, gd_s3.n)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros((gd_s3.n, gd_s3.n), dtype=np.uint8))

    def test_right_rep_identity(self, gd_s3, gd_c4):
        """right_rep((identity,), gd) is the identity matrix."""
        result_s3 = right_rep((gd_s3.identity,), gd_s3)
        np.testing.assert_array_equal(result_s3, np.eye(gd_s3.n, dtype=np.uint8))

        result_c4 = right_rep((gd_c4.identity,), gd_c4)
        np.testing.assert_array_equal(result_c4, np.eye(gd_c4.n, dtype=np.uint8))

    def test_right_rep_homomorphism(self, gd_c4):
        """right_rep((g_i · g_j,), gd) == right_rep((g_i,), gd) @ right_rep((g_j,), gd) mod 2."""
        g_i, g_j = 1, 2
        g_product = gd_c4.mult[g_i][g_j]

        R_i = right_rep((g_i,), gd_c4)
        R_j = right_rep((g_j,), gd_c4)
        R_product = right_rep((g_product,), gd_c4)

        result = (R_i @ R_j) % 2
        np.testing.assert_array_equal(result, R_product)

    def test_right_rep_transpose_dagger(self, gd_s3):
        """right_rep(x, gd).T == right_rep(dagger(x, gd), gd)."""
        x = (0, 1, 2)
        R_x = right_rep(x, gd_s3)
        R_dag_x = right_rep(dagger(x, gd_s3), gd_s3)

        np.testing.assert_array_equal(R_x.T, R_dag_x)

    def test_right_rep_shape_and_dtype(self, gd_s3, gd_c4, gd_d4):
        """right_rep always returns n×n uint8 matrix."""
        for x in [(), (0,), (1, 2)]:
            for gd in [gd_s3, gd_c4, gd_d4]:
                result = right_rep(x, gd)
                assert result.shape == (gd.n, gd.n)
                assert result.dtype == np.uint8


class TestLRCommutativity:
    """Test that L and R commute."""

    def test_left_right_commute(self, gd_s3, gd_c4):
        """(L[x] @ R[y]) mod 2 == (R[y] @ L[x]) mod 2."""
        for x in [(), (0,), (0, 1)]:
            for y in [(), (1,), (1, 2)]:
                L_x = left_rep(x, gd_s3)
                R_y = right_rep(y, gd_s3)

                left_result = (L_x @ R_y) % 2
                right_result = (R_y @ L_x) % 2

                np.testing.assert_array_equal(left_result, right_result)

    def test_left_right_commute_abelian(self, gd_c4):
        """Commutativity also holds for abelian groups."""
        for x in [(), (0,), (1, 2)]:
            for y in [(), (2,), (0, 3)]:
                L_x = left_rep(x, gd_c4)
                R_y = right_rep(y, gd_c4)

                left_result = (L_x @ R_y) % 2
                right_result = (R_y @ L_x) % 2

                np.testing.assert_array_equal(left_result, right_result)
