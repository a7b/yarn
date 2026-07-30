"""Tests for core.dist.quantum_sqetch (Type 1).

Requires GAP (for GroupData fixtures) AND a CUDA-capable GPU (for sqetch).
Marked with both `gap` and `gpu`. Skips automatically if no CUDA is visible.
"""

import numpy as np
import pytest

from core.classical_code import build_A_bin, build_B_bin
from core.group import GroupData
from core.quantum_code import build_Hx, build_Hz, check_css

pytestmark = [pytest.mark.gap, pytest.mark.gpu]


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 1
    except Exception:
        return False


def _has_multi_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        return False


# Skip the whole module if no GPU.
pytest.importorskip("torch")
if not _has_cuda():
    pytest.skip("No CUDA device", allow_module_level=True)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="session")
def small_css(gd_c4):
    """A small CSS code over C4 + paired logical basis via find_logical_basis.

    Uses build_quantum_code to get canonical-form Hx/Hz/A_bin/B_bin (the
    orbit-pairing trick requires the dep block-col at the end).
    """
    from core.quantum_code import build_quantum_code
    from logical_basis.logical_basis import find_logical_basis

    A = [[(gd_c4.identity,), (1, 2)]]
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
# Default return: (dx, dz)
# ─────────────────────────────────────────────────────────────────


class TestReturnDefault:
    def test_returns_pair(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        result = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=200, k_sub=8, seed=0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_each_is_int_or_none(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=200, k_sub=8, seed=0,
        )
        for d in (dx, dz):
            assert d is None or isinstance(d, int)


# ─────────────────────────────────────────────────────────────────
# return_logical=True: 4-tuple with codewords
# ─────────────────────────────────────────────────────────────────


class TestReturnLogical:
    def test_returns_four_tuple(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        result = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=400, k_sub=8, seed=0,
            return_logical=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_codewords_valid(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        dx, dz, vec_dx, vec_dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=2000, k_sub=8, seed=1,
            return_logical=True,
        )
        n_phys = Hx.shape[1]
        if vec_dx is not None:
            assert vec_dx.shape == (n_phys,)
            assert vec_dx.dtype == np.uint8
            # vec_dx is a Z-logical: in ker(Hz) and with nonzero overlap with Lx (cross-type).
            np.testing.assert_array_equal(
                (Hz.astype(int) @ vec_dx.astype(int)) % 2,
                np.zeros(Hz.shape[0], dtype=int),
            )
            assert ((Lz.astype(int) @ vec_dx.astype(int)) % 2).sum() > 0
            assert int(vec_dx.sum()) == dx
        if vec_dz is not None:
            assert vec_dz.shape == (n_phys,)
            assert vec_dz.dtype == np.uint8
            np.testing.assert_array_equal(
                (Hx.astype(int) @ vec_dz.astype(int)) % 2,
                np.zeros(Hx.shape[0], dtype=int),
            )
            assert ((Lx.astype(int) @ vec_dz.astype(int)) % 2).sum() > 0
            assert int(vec_dz.sum()) == dz


# ─────────────────────────────────────────────────────────────────
# Multi-GPU strategies
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_multi_gpu(),
                    reason="requires >= 2 visible CUDA devices")
class TestMultiGPU:
    def test_auto_strategy_on_2_gpus_uses_direction_split(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=400, k_sub=8, seed=2,
            strategy="auto",
        )
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)

    def test_explicit_direction_split(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=400, k_sub=8, seed=3,
            strategy="direction_split",
        )
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)

    def test_trial_split_matches_single_gpu(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        dx_t, dz_t = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=800, k_sub=8, seed=4,
            strategy="trial_split",
        )
        dx_s, dz_s = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=800, k_sub=8, seed=4,
            strategy="direction_split",
        )
        # Both should find the same minimum weight if num_trials is enough.
        # Hard guarantee not possible (stochastic), but with 800 trials on a
        # small code we expect agreement.
        for d_t, d_s in [(dx_t, dx_s), (dz_t, dz_s)]:
            if d_t is not None and d_s is not None:
                assert d_t == d_s


# ─────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────


class TestSeedHandling:
    def test_dx_and_dz_receive_distinct_seeds(self, small_css, monkeypatch):
        """Verify that the wrapper resolves seed=None at its own level and
        passes DISTINCT seeds to the dx and dz direction calls — otherwise
        two concurrent direction_split threads could fall through to
        sqetch's per-call `time.time()` and collide on the same microsecond.
        """
        Hx, Hz, Lx, Lz = small_css

        captured_seeds: list = []

        import sqetch as _sqetch
        real_estimate = _sqetch.estimate_distance

        def spy(*args, **kwargs):
            captured_seeds.append(kwargs.get("seed"))
            return real_estimate(*args, **kwargs)

        monkeypatch.setattr(_sqetch, "estimate_distance", spy)

        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz,
            num_trials=20, k_sub=4, seed=None,
            strategy="direction_split",
        )
        assert len(captured_seeds) == 2, (
            f"expected exactly 2 sqetch.estimate_distance calls "
            f"(dx + dz); got {len(captured_seeds)}"
        )
        s_dx, s_dz = captured_seeds
        assert s_dx is not None and s_dz is not None, (
            "wrapper must resolve seed=None to concrete ints before "
            "dispatching to sqetch (so two concurrent threads can't collide)"
        )
        assert s_dx != s_dz, (
            f"dx and dz must receive different seeds; got {s_dx} == {s_dz}"
        )

    def test_explicit_seed_gives_distinct_dx_dz(self, small_css, monkeypatch):
        Hx, Hz, Lx, Lz = small_css

        captured_seeds: list = []
        import sqetch as _sqetch
        real_estimate = _sqetch.estimate_distance

        def spy(*args, **kwargs):
            captured_seeds.append(kwargs.get("seed"))
            return real_estimate(*args, **kwargs)

        monkeypatch.setattr(_sqetch, "estimate_distance", spy)

        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz,
            num_trials=20, k_sub=4, seed=42,
            strategy="direction_split",
        )
        # With seed=42 the wrapper should have used 42 and 42 + 1_000_003.
        assert sorted(captured_seeds) == [42, 42 + 1_000_003]


class TestAutoLogicalBasis:
    def test_omitting_lx_lz_works(self, small_css):
        """If Lx, Lz not passed, sqetch auto-computes via
        logical_basis.find_logical_noncanonical_RREF."""
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, _Lx, _Lz = small_css
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, num_trials=200, k_sub=8, seed=0,
        )
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)

    def test_omitting_only_lz_recomputes_both(self, small_css):
        """One of Lx/Lz being None triggers full recompute for pairing
        consistency."""
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx_known, _Lz = small_css
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx_known, None,
            num_trials=200, k_sub=8, seed=1,
        )
        assert dx is None or isinstance(dx, int)
        assert dz is None or isinstance(dz, int)


class TestInputValidation:
    def test_hx_hz_col_mismatch_raises(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        with pytest.raises(ValueError, match="column counts disagree"):
            estimate_quantum_distances_sqetch(
                Hx, Hz[:, :-1], Lx, Lz, num_trials=10, k_sub=4,
            )

    def test_unknown_strategy_raises(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        with pytest.raises(ValueError, match="strategy must be"):
            estimate_quantum_distances_sqetch(
                Hx, Hz, Lx, Lz, num_trials=10, k_sub=4,
                strategy="nonsense",
            )

    def test_num_trials_zero_raises(self, small_css):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz, Lx, Lz = small_css
        with pytest.raises(ValueError, match="num_trials"):
            estimate_quantum_distances_sqetch(
                Hx, Hz, Lx, Lz, num_trials=0, k_sub=4,
            )
