"""Tests for ``find_logical_basis_pivot_aligned``.

The contract: ``Lx · Lz.T == I_k`` by construction — no permutation /
no P⁻¹ row-mix. Tests verify that, plus the kernel constraints
``Hz · Lx.T == 0`` and ``Hx · Lz.T == 0``.
"""

import numpy as np
import pytest

from logical_basis.logical_basis import find_logical_basis_pivot_aligned


pytestmark = pytest.mark.fast


def _steane_inputs():
    """[[7, 1, 3]] Steane code: same Hamming matrix for Hx and Hz."""
    H = np.array(
        [[1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8,
    )
    return H, H


class TestNativePairing:
    def test_Lx_Lz_T_is_identity(self):
        Hx, Hz = _steane_inputs()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        P = (Lx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
        np.testing.assert_array_equal(P, np.eye(Lz.shape[0], dtype=np.uint8))

    def test_Lz_in_ker_Hx(self):
        Hx, Hz = _steane_inputs()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        prod = (Hx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
        np.testing.assert_array_equal(prod, np.zeros_like(prod))

    def test_Lx_in_ker_Hz(self):
        Hx, Hz = _steane_inputs()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        prod = (Hz.astype(np.int32) @ Lx.T.astype(np.int32)) % 2
        np.testing.assert_array_equal(prod, np.zeros_like(prod))


class TestKValue:
    def test_steane_k_1(self):
        Hx, Hz = _steane_inputs()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        assert Lx.shape == (1, 7)
        assert Lz.shape == (1, 7)

    def test_no_logicals_returns_empty(self):
        # Full-rank Hx + Hz with no logicals.
        Hx = np.eye(4, dtype=np.uint8)
        Hz = np.zeros((0, 4), dtype=np.uint8)
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        # ker(Hx) is trivial → k = 0.
        assert Lx.shape == (0, 4)
        assert Lz.shape == (0, 4)


class TestShapeMismatch:
    def test_column_mismatch_raises(self):
        Hx = np.eye(3, dtype=np.uint8)
        Hz = np.eye(4, dtype=np.uint8)
        with pytest.raises(ValueError, match="same number of columns"):
            find_logical_basis_pivot_aligned(Hx, Hz)


class TestRankIncrementInvariant:
    """The defining CSS-logical-basis invariant:

        rank([Hx; Lx]) − rank(Hx) = k
        rank([Hz; Lz]) − rank(Hz) = k
    """

    def _check(self, Hx, Hz):
        from core.f2 import f2_rank
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        k = Lx.shape[0]
        dx_rank = f2_rank(np.vstack([Hx, Lx])) - f2_rank(Hx)
        dz_rank = f2_rank(np.vstack([Hz, Lz])) - f2_rank(Hz)
        assert dx_rank == k, f"rank([Hx; Lx]) − rank(Hx) = {dx_rank}, k = {k}"
        assert dz_rank == k, f"rank([Hz; Lz]) − rank(Hz) = {dz_rank}, k = {k}"
        assert dx_rank == dz_rank

    def test_steane(self):
        Hx, Hz = _steane_inputs()
        self._check(Hx, Hz)

    def test_two_steane_direct_sum(self):
        H = np.array(
            [[1, 0, 1, 0, 1, 0, 1],
             [0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8,
        )
        Hx = np.block([[H, np.zeros_like(H)],
                       [np.zeros_like(H), H]])
        Hz = Hx.copy()
        self._check(Hx, Hz)


class TestComparisonWithRREF:
    """The native-pair function should yield the same k as the older
    RREF-based extractor; only the row ordering / mixing differs."""

    def test_same_k(self):
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = _steane_inputs()
        Lx_a, Lz_a = find_logical_basis_pivot_aligned(Hx, Hz)
        Lx_b, Lz_b = find_logical_noncanonical_RREF(Hx, Hz)
        assert Lx_a.shape == Lx_b.shape
        assert Lz_a.shape == Lz_b.shape


class TestReorderByPivot:
    """The pairing P = Lx · Lz.T must be preserved by reorder_by_pivot."""

    def test_identity_preserved(self):
        from logical_basis.logical_basis import (
            find_logical_basis_pivot_aligned,
            reorder_by_pivot,
        )
        Hx, Hz = _steane_inputs()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        Lx2, Lz2 = reorder_by_pivot(Lx, Lz, ref="Lz")
        P = (Lx2.astype(np.int32) @ Lz2.T.astype(np.int32)) % 2
        np.testing.assert_array_equal(P, np.eye(Lz.shape[0], dtype=np.uint8))

    def test_lz_pivots_ascending(self):
        from logical_basis.logical_basis import (
            find_logical_basis_pivot_aligned,
            reorder_by_pivot,
        )
        # Build a non-trivial Lz/Lx pair (use a 2-logical fixture):
        # Direct sum of two Steane codes → k = 2.
        H = np.array(
            [[1, 0, 1, 0, 1, 0, 1],
             [0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8,
        )
        Hx = np.block([[H, np.zeros_like(H)],
                       [np.zeros_like(H), H]])
        Hz = Hx.copy()
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        Lx2, Lz2 = reorder_by_pivot(Lx, Lz, ref="Lz")
        # Lz2 pivots should be ascending.
        pivots = []
        for i in range(Lz2.shape[0]):
            nz = np.where(Lz2[i] != 0)[0]
            pivots.append(int(nz[0]) if nz.size else Lz2.shape[1])
        assert pivots == sorted(pivots)

    def test_row_count_mismatch_raises(self):
        from logical_basis.logical_basis import reorder_by_pivot
        Lx = np.zeros((2, 7), dtype=np.uint8)
        Lz = np.zeros((3, 7), dtype=np.uint8)
        with pytest.raises(ValueError, match="row count"):
            reorder_by_pivot(Lx, Lz)

    def test_invalid_ref_raises(self):
        from logical_basis.logical_basis import reorder_by_pivot
        Lx = np.zeros((2, 7), dtype=np.uint8)
        Lz = np.zeros((2, 7), dtype=np.uint8)
        with pytest.raises(ValueError, match="ref"):
            reorder_by_pivot(Lx, Lz, ref="foo")
