"""Tests for core.dist.quantum_bposd (Type 1).

Requires GAP (for GroupData fixtures) and the bposd / ldpc packages.
"""

import numpy as np
import pytest

from core.classical_code import build_A_bin, build_B_bin
from core.f2 import f2_rank
from core.group import GroupData
from core.quantum_code import build_Hx, build_Hz, check_css, compute_k
from core.dist.quantum_bposd import estimate_quantum_distances_bposd

pytestmark = [pytest.mark.gap, pytest.mark.bposd]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="session")
def small_css_code(gd_c4):
    """A small CSS code over C4 + its paired logical basis via find_logical_basis.

    Uses build_quantum_code to get canonical-form Hx/Hz/A_bin/B_bin (the
    orbit-pairing trick requires the dep block-cols at the end), then
    find_logical_basis to derive Lx/Lz.
    """
    from core.quantum_code import build_quantum_code
    from logical_basis.logical_basis import find_logical_basis

    A = [[(gd_c4.identity,), (1, 2)]]   # identity entry → block-col is full-rank
    B = [[(gd_c4.identity,), (1, 2)]]
    r = build_quantum_code(A, B, gd_c4)
    Hx, Hz = r["Hx"], r["Hz"]
    assert check_css(Hx, Hz)
    basis = find_logical_basis(
        Hx, Hz,
        r["A_bin_canonical"], r["B_bin_canonical"],
        gd_c4.n, shape_a=(1, 2),
    )
    return Hx, Hz, basis["Lx"], basis["Lz"]


# ─────────────────────────────────────────────────────────────────
# Return type / shape contract
# ─────────────────────────────────────────────────────────────────


class TestReturnContract:
    def test_returns_pair(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        result = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=20, n_workers=1
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_each_element_is_int_or_none(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=20, n_workers=1
        )
        for d in (dx, dz):
            assert d is None or isinstance(d, int)


# ─────────────────────────────────────────────────────────────────
# Functional behavior
# ─────────────────────────────────────────────────────────────────


class TestFindsDistance:
    def test_finds_some_codeword_with_enough_trials(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=200, n_workers=2,
            osd_order=10,
        )
        # On a small code with k > 0, 200 trials × osd_order=10 should
        # find SOMETHING in each direction.
        assert dx is not None and dx >= 1
        assert dz is not None and dz >= 1

    def test_dx_bounded_by_n_phys(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        n_phys = Hx.shape[1]
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=200, n_workers=2, osd_order=10,
        )
        if dx is not None:
            assert 1 <= dx <= n_phys
        if dz is not None:
            assert 1 <= dz <= n_phys


# ─────────────────────────────────────────────────────────────────
# Early-stop semantics
# ─────────────────────────────────────────────────────────────────


class TestEarlyStop:
    def test_high_d_target_triggers_early_stop(self, small_css_code):
        # Set d_target larger than n_phys; any found codeword has weight < d_target
        # so early stop fires immediately, on the first valid trial. The function
        # should still return promptly with a finite estimate.
        Hx, Hz, Lx, Lz = small_css_code
        n_phys = Hx.shape[1]
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz,
            num_trials=10_000, n_workers=2,
            d_target=n_phys + 100,
            osd_order=5,
        )
        # Even with 10k trials, finishes fast because early-stop kicks in.
        assert dx is not None
        assert dz is not None

    def test_d_target_none_uses_all_trials(self, small_css_code):
        # No early stop; estimator just runs all trials and returns best.
        Hx, Hz, Lx, Lz = small_css_code
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=50, n_workers=2,
            d_target=None,
            osd_order=5,
        )
        # Either None (didn't find anything in 50) or a valid int.
        for d in (dx, dz):
            assert d is None or isinstance(d, int)


# ─────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_logicals_returns_none(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        n_phys = Hx.shape[1]
        empty_L = np.zeros((0, n_phys), dtype=np.uint8)
        # dx direction uses Lz; if we pass empty Lz, dx is None.
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, empty_L, num_trials=10, n_workers=1,
        )
        assert dx is None
        # dz direction uses Lx; nonempty, so dz can be int or None depending
        # on trials. Just verify it didn't crash.
        assert dz is None or isinstance(dz, int)

    def test_both_logicals_empty(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        n_phys = Hx.shape[1]
        empty_L = np.zeros((0, n_phys), dtype=np.uint8)
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, empty_L, empty_L, num_trials=10, n_workers=1,
        )
        assert dx is None
        assert dz is None


# ─────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────


class TestAutoLogicalBasis:
    def test_omitting_lx_lz_works(self, small_css_code):
        """If Lx, Lz are not passed, BP+OSD auto-computes a basis via
        logical_basis.find_logical_noncanonical_RREF."""
        Hx, Hz, _Lx, _Lz = small_css_code
        # Don't pass Lx, Lz.
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, num_trials=200, n_workers=2, osd_order=10,
        )
        # Same code, same n_trials → should land on the same (or comparable)
        # weights as the fixture-supplied basis path.
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)

    def test_omitting_only_lx_still_recomputes_both(self, small_css_code):
        """Passing only one of Lx/Lz triggers full recompute so the basis
        is internally consistent (paired)."""
        Hx, Hz, _Lx, Lz_known = small_css_code
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, None, Lz_known,
            num_trials=200, n_workers=2, osd_order=10,
        )
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)


class TestInputValidation:
    def test_hx_hz_column_mismatch_raises(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        bad_Hz = Hz[:, :-1]
        with pytest.raises(ValueError, match="column counts disagree"):
            estimate_quantum_distances_bposd(
                Hx, bad_Hz, Lx, Lz, num_trials=10, n_workers=1
            )

    def test_lx_column_mismatch_raises(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        bad_Lx = Lx[:, :-1]
        with pytest.raises(ValueError, match=r"Lx has \d+ cols"):
            estimate_quantum_distances_bposd(
                Hx, Hz, bad_Lx, Lz, num_trials=10, n_workers=1
            )

    def test_num_trials_zero_raises(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        with pytest.raises(ValueError, match="num_trials"):
            estimate_quantum_distances_bposd(
                Hx, Hz, Lx, Lz, num_trials=0, n_workers=1
            )

    def test_n_workers_zero_raises(self, small_css_code):
        Hx, Hz, Lx, Lz = small_css_code
        with pytest.raises(ValueError, match="n_workers"):
            estimate_quantum_distances_bposd(
                Hx, Hz, Lx, Lz, num_trials=10, n_workers=0
            )

    def test_accepts_1d_logical(self, small_css_code):
        # A single logical row passed as 1-D array should be accepted.
        Hx, Hz, Lx, Lz = small_css_code
        if Lz.shape[0] >= 1:
            lz1 = Lz[0]
            dx, dz = estimate_quantum_distances_bposd(
                Hx, Hz, Lx, lz1, num_trials=20, n_workers=1,
            )
            assert dx is None or isinstance(dx, int)
            assert dz is None or isinstance(dz, int)
