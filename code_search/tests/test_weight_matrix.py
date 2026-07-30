"""Tests for the weight-matrix helpers across core/ and search/.

Covers:
- ``core.classical_code.weight_matrix``
- ``core.quantum_code.quantum_check_weights``
- ``search.filters.classical.abelian.weight_distance_bound``
"""

import numpy as np
import pytest

from core.classical_code import weight_matrix
from core.quantum_code import quantum_check_weights


pytestmark = pytest.mark.fast


class TestWeightMatrix:
    def test_per_entry_weights(self):
        A = [[(0, 1), (2,)],
             [(), (0, 1, 2)]]
        W = weight_matrix(A)
        assert W.dtype.kind == "i"
        np.testing.assert_array_equal(W, np.array([[2, 1], [0, 3]]))

    def test_returns_ndarray(self):
        A = [[(0,)]]
        W = weight_matrix(A)
        assert isinstance(W, np.ndarray)
        assert W.shape == (1, 1)


class TestQuantumCheckWeights:
    def test_values_simple(self):
        # W_A = [[2, 1]]: rowsum_A_max = 3, colsum_A_max = max(2, 1) = 2.
        # W_B = [[3, 2]]: rowsum_B_max = 5, colsum_B_max = max(3, 2) = 3.
        # Hx_check_weight = rowsum_A_max + colsum_B_max = 3 + 3 = 6.
        # Hz_check_weight = colsum_A_max + rowsum_B_max = 2 + 5 = 7.
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        out = quantum_check_weights(W_A, W_B)
        assert out == {"Hx_check_weight": 6, "Hz_check_weight": 7}

    def test_consistency_with_actual_Hx_Hz(self):
        """End-to-end: derived check weights equal actual Hx/Hz max row weight."""
        from core.group import GroupData
        from core.quantum_code import build_Hx, build_Hz

        gd = GroupData("CyclicGroup(4)")
        A = [[(0, 1), (2,)]]
        B = [[(0, 1, 2), (1, 3)]]
        Hx = build_Hx(A, B, gd)
        Hz = build_Hz(A, B, gd)
        W_A = weight_matrix(A)
        W_B = weight_matrix(B)
        derived = quantum_check_weights(W_A, W_B)

        assert int(Hx.sum(axis=1).max()) == derived["Hx_check_weight"]
        assert int(Hz.sum(axis=1).max()) == derived["Hz_check_weight"]


class TestWeightDistanceBound:
    pytestmark = pytest.mark.gap

    @pytest.fixture(scope="class")
    def gd_c4(self):
        from core.group import GroupData
        return GroupData("CyclicGroup(4)")

    @pytest.fixture(scope="class")
    def gd_s3(self):
        from core.group import GroupData
        return GroupData("SymmetricGroup(3)")

    def test_J1_matches_ring_bound(self, gd_c4):
        """For J=1 the weight bound equals the ring bound (no ring mult)."""
        from search.filters.classical.abelian.weight_distance_bound import (
            weight_distance_bound,
        )
        from search.filters.classical.abelian.ring_distance_bound import (
            ring_distance_bound,
        )
        A = [[(0,), (1, 2), (3,)]]   # weights 1, 2, 1
        W_A = weight_matrix(A)
        ring_bound = ring_distance_bound(A, gd_c4)
        weight_bound = weight_distance_bound(W_A, gd_c4)
        assert ring_bound == weight_bound == 2   # min pairwise = 1+1

    def test_target_exceeds_returns_inf(self, gd_c4):
        from search.filters.classical.abelian.weight_distance_bound import (
            weight_distance_bound,
        )
        W = np.array([[2]])
        assert weight_distance_bound(W, gd_c4) == float("inf")

    def test_raises_on_non_abelian(self, gd_s3):
        from search.filters.classical.abelian.weight_distance_bound import (
            weight_distance_bound,
        )
        W = np.array([[1, 1]])
        with pytest.raises(ValueError, match="abelian"):
            weight_distance_bound(W, gd_s3)

    def test_upper_bounds_ring_bound(self, gd_c4):
        """weight_bound ≥ ring_bound (looser or equal)."""
        from search.filters.classical.abelian.weight_distance_bound import (
            weight_distance_bound,
        )
        from search.filters.classical.abelian.ring_distance_bound import (
            ring_distance_bound,
        )
        # J=2 case — weight bound should be ≥ ring bound.
        A = [[(0,), (1,), (2,)],
             [(1,), (2,), (3,)]]
        W_A = weight_matrix(A)
        assert weight_distance_bound(W_A, gd_c4) >= ring_distance_bound(A, gd_c4)
