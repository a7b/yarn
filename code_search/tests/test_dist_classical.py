"""Tests for ``core/dist/classical.py``.

Verifies the BP+OSD and SQetch classical distance wrappers with the
empty-Hz trick against small hand-checked codes.
"""

import numpy as np
import pytest

from core.dist.classical import (
    estimate_classical_distance,
    estimate_classical_distance_bposd,
    estimate_classical_distance_sqetch,
)


pytestmark = pytest.mark.bposd


class TestAliasing:
    """The unsuffixed alias resolves to the BP+OSD variant."""

    def test_alias_is_bposd(self):
        assert estimate_classical_distance is estimate_classical_distance_bposd


class TestEstimateClassicalDistance:
    def test_repetition_code_distance_n(self):
        """[n, 1, n] repetition code: H = [[1,1,0,...], [0,1,1,0,...], ...]
        The min weight nonzero codeword is the all-ones vector (weight n).
        With small n the decoder should find the all-ones quickly."""
        n = 4
        H = np.array(
            [[1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 1]], dtype=np.uint8,
        )
        d = estimate_classical_distance(H, num_trials=200, n_workers=2, osd_order=0)
        assert d == 4

    def test_hamming_7_4_3(self):
        """[7, 4, 3] Hamming code — min distance is 3."""
        H = np.array(
            [[1, 0, 1, 0, 1, 0, 1],
             [0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8,
        )
        d = estimate_classical_distance(H, num_trials=200, n_workers=2, osd_order=0)
        assert d == 3

    def test_d_target_strict_early_stop(self):
        """A codeword of weight == d_target does NOT trigger early stop;
        weight < d_target does."""
        n = 4
        H = np.array(
            [[1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 1]], dtype=np.uint8,
        )
        # Actual distance is 4. d_target=5 should still find weight 4.
        d = estimate_classical_distance(
            H, num_trials=200, n_workers=2, d_target=5, osd_order=0,
        )
        assert d == 4

    def test_full_rank_H_no_nonzero_codeword(self):
        """If H is full row rank with ker(H)={0}, returns None."""
        H = np.eye(4, dtype=np.uint8)
        d = estimate_classical_distance(H, num_trials=200, n_workers=2, osd_order=0)
        # ker(H) = {0}: no nonzero codeword. Returns None.
        assert d is None or d == 0


class TestEstimateClassicalDistanceSqetch:
    """Mirror tests against the sqetch backend. Marked `gpu` (CUDA required)."""

    pytestmark = pytest.mark.gpu

    def test_repetition_code_distance_n_sqetch(self):
        n = 4
        H = np.array(
            [[1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 1]], dtype=np.uint8,
        )
        d = estimate_classical_distance_sqetch(H, num_trials=200)
        assert d == 4

    def test_hamming_7_4_3_sqetch(self):
        H = np.array(
            [[1, 0, 1, 0, 1, 0, 1],
             [0, 1, 1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8,
        )
        d = estimate_classical_distance_sqetch(H, num_trials=200)
        assert d == 3

    def test_return_codeword(self):
        """return_codeword=True returns (dist, codeword) where codeword has
        the expected weight and lies in ker(H)."""
        H = np.array(
            [[1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 1]], dtype=np.uint8,
        )
        d, cw = estimate_classical_distance_sqetch(
            H, num_trials=200, return_codeword=True,
        )
        assert d == 4
        assert cw is not None
        assert int(cw.sum()) == 4
        # Codeword must be in ker(H).
        assert int(((H @ cw) % 2).sum()) == 0
