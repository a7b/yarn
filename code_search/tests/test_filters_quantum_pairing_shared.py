"""Unit tests for ``search/filters/quantum_pairing/_shared/`` predicates.

Each predicate is tested in isolation; the dispatcher (composing them
cheapest-first) is tested in ``test_filters_dispatcher.py``.
"""

import numpy as np
import pytest

from search.filters.quantum_pairing._shared.max_check_weight import (
    max_check_weight,
)
from search.filters.quantum_pairing._shared.min_classical_distance import (
    min_classical_distance,
)
from search.filters.quantum_pairing._shared.min_classical_girth import (
    min_classical_girth,
)
from search.filters.quantum_pairing._shared.same_group import same_group
from search.filters.quantum_pairing._shared.same_shape import same_shape

pytestmark = pytest.mark.fast


class TestSameGroup:
    def test_match(self):
        assert same_group("S3", "S3") is True

    def test_mismatch(self):
        assert same_group("S3", "C5") is False


class TestSameShape:
    def test_match_tuple(self):
        assert same_shape((1, 2), (1, 2)) is True

    def test_match_list_vs_tuple(self):
        assert same_shape([1, 2], (1, 2)) is True

    def test_mismatch(self):
        assert same_shape((1, 2), (1, 3)) is False
        assert same_shape((1, 2), (2, 2)) is False


class TestMinClassicalDistance:
    def test_both_above(self):
        assert min_classical_distance(8, 10, 8) is True

    def test_one_below(self):
        assert min_classical_distance(7, 10, 8) is False

    def test_none_rejects(self):
        assert min_classical_distance(None, 10, 8) is False


class TestMinClassicalGirth:
    def test_both_above(self):
        assert min_classical_girth(8, 10, 8) is True

    def test_one_below(self):
        assert min_classical_girth(6, 10, 8) is False

    def test_none_is_forest_passes(self):
        """girth=None means forest (no cycles) → infinite → passes."""
        assert min_classical_girth(None, 10, 8) is True
        assert min_classical_girth(10, None, 8) is True


class TestMaxCheckWeight:
    def test_returns_true_when_no_cap_set(self):
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        assert max_check_weight(W_A, W_B) is True

    def test_within_caps_passes(self):
        # W_A = [[2, 1]], W_B = [[3, 2]] → Hx_cw = 6, Hz_cw = 7.
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        assert max_check_weight(W_A, W_B, max_Hx=6, max_Hz=7) is True

    def test_Hx_cap_violated(self):
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        assert max_check_weight(W_A, W_B, max_Hx=5) is False

    def test_Hz_cap_violated(self):
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        assert max_check_weight(W_A, W_B, max_Hz=6) is False
