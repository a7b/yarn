"""Tests for core.quantum_code (Type 1: implementation-aware).

Covers:
- build_Hx / build_Hz (forward), with and without canonicalize.
- build_quantum_code (high-level wrapper).
- compute_k / check_css.
- A_bin_B_bin_from_Hx_Hz / AB_from_Hx_Hz (inverse) with cross-check.
- Non-default shape_b (mb ≠ ma, nb ≠ na).
- Non-rectangular shapes (ma > 1).
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
)
from core.f2 import f2_rank
from core.group import GroupData, dagger
from core.quantum_code import (
    AB_from_Hx_Hz,
    A_bin_B_bin_from_Hx_Hz,
    _build_Hx_raw,
    _build_Hz_raw,
    build_Hx,
    build_Hz,
    build_quantum_code,
    check_css,
    compute_k,
)

pytestmark = pytest.mark.gap


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


# Standard (1, 2) shape with mb=ma, nb=na.
@pytest.fixture
def AB_1x2_s3(gd_s3):
    A = [[(gd_s3.identity,), (1, 2)]]
    B = [[(gd_s3.identity,), (3,)]]
    return A, B


# Non-rectangular (2, 3) shape.
@pytest.fixture
def AB_2x3_s3(gd_s3):
    A = [[(gd_s3.identity,), (1, 2), (3,)],
         [(4,), (gd_s3.identity,), (5,)]]
    B = [[(gd_s3.identity,), (1, 2), (3,)],
         [(4,), (gd_s3.identity,), (5,)]]
    return A, B


# Non-default shape_b: ma=1, na=2, mb=1, nb=3.
@pytest.fixture
def AB_1x2_1x3_s3(gd_s3):
    A = [[(gd_s3.identity,), (1, 2)]]
    B = [[(gd_s3.identity,), (3,), (1, 4)]]
    return A, B


# ─────────────────────────────────────────────────────────────────
# Forward builders
# ─────────────────────────────────────────────────────────────────


class TestBuildHx:
    def test_shape_default(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        # ma=1, na=2, mb=1, nb=2, n=6
        # rows = ma·nb·n = 12, cols = (na·nb + ma·mb)·n = (4 + 1)·6 = 30
        assert Hx.shape == (12, 30)

    def test_dtype_uint8(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        assert build_Hx(A, B, gd_s3).dtype == np.uint8

    def test_non_rectangular_shape(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        # ma=2, na=3, mb=2, nb=3, n=6
        # rows = ma·nb·n = 36
        # cols = (na·nb + ma·mb)·n = (9 + 4)·6 = 78
        Hx = build_Hx(A, B, gd_s3)
        assert Hx.shape == (36, 78)

    def test_non_default_shape_b(self, gd_s3, AB_1x2_1x3_s3):
        A, B = AB_1x2_1x3_s3
        # ma=1, na=2, mb=1, nb=3, n=6
        # rows = ma·nb·n = 18
        # cols = (na·nb + ma·mb)·n = (6 + 1)·6 = 42
        Hx = build_Hx(A, B, gd_s3)
        assert Hx.shape == (18, 42)


class TestBuildHz:
    def test_shape_default(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hz = build_Hz(A, B, gd_s3)
        # ma=1, na=2, mb=1, nb=2, n=6
        # rows = na·mb·n = 2·1·6 = 12
        assert Hz.shape == (12, 30)

    def test_dtype_uint8(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        assert build_Hz(A, B, gd_s3).dtype == np.uint8

    def test_non_rectangular(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        # ma=2, na=3, mb=2, nb=3, n=6
        # rows = na·mb·n = 36
        Hz = build_Hz(A, B, gd_s3)
        assert Hz.shape == (36, 78)


# ─────────────────────────────────────────────────────────────────
# CSS orthogonality and k
# ─────────────────────────────────────────────────────────────────


class TestCheckCSS:
    def test_orthogonality_for_built_code_1x2(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        assert check_css(Hx, Hz)

    def test_orthogonality_for_built_code_2x3(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        assert check_css(Hx, Hz)

    def test_orthogonality_non_default_shape_b(self, gd_s3, AB_1x2_1x3_s3):
        A, B = AB_1x2_1x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        assert check_css(Hx, Hz)

    def test_abelian_orthogonality(self, gd_c4):
        A = [[(0,), (1, 2)]]
        B = [[(0,), (1, 2)]]
        Hx = build_Hx(A, B, gd_c4)
        Hz = build_Hz(A, B, gd_c4)
        assert check_css(Hx, Hz)


class TestComputeK:
    def test_returns_int(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        k = compute_k(Hx, Hz)
        assert isinstance(k, int)
        assert k >= 0
        assert k <= Hx.shape[1]

    def test_k_for_known_1x2_s3(self, gd_s3):
        # Two identity entries → trivial code with maximal k.
        A = [[(gd_s3.identity,), (gd_s3.identity,)]]
        B = [[(gd_s3.identity,), (gd_s3.identity,)]]
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        # The code is well-formed (CSS); k = n_phys - rank(Hx) - rank(Hz).
        # We don't predict the exact value here — just check sanity bounds.
        k = compute_k(Hx, Hz)
        assert 0 <= k <= Hx.shape[1]


# ─────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────


class TestBuildQuantumCode:
    def test_returns_expected_keys(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        result = build_quantum_code(A, B, gd_s3)
        expected_keys = {
            "Hx", "Hz",
            "A_canonical", "A_bin_canonical",
            "B_canonical", "B_bin_canonical",
            "perm_a", "perm_b",
            "has_full_rank_a", "has_full_rank_b",
        }
        assert set(result.keys()) == expected_keys

    def test_hx_hz_match_canonical_rebuild(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        result = build_quantum_code(A, B, gd_s3)
        # Hx must equal _build_Hx_raw of the canonical pair.
        Hx_expected = _build_Hx_raw(result["A_canonical"], result["B_canonical"], gd_s3)
        Hz_expected = _build_Hz_raw(result["A_canonical"], result["B_canonical"], gd_s3)
        np.testing.assert_array_equal(result["Hx"], Hx_expected)
        np.testing.assert_array_equal(result["Hz"], Hz_expected)

    def test_built_code_passes_css(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        result = build_quantum_code(A, B, gd_s3)
        assert check_css(result["Hx"], result["Hz"])


# ─────────────────────────────────────────────────────────────────
# Inverse — A_bin / B_bin extraction
# ─────────────────────────────────────────────────────────────────


class TestABinBBinFromHxHz:
    def test_matches_classical_build(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))
        np.testing.assert_array_equal(A_bin, build_A_bin(A, gd_s3))
        np.testing.assert_array_equal(B_bin, build_B_bin(B, gd_s3))

    def test_round_trip_through_classical_inverse(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))
        # Use the classical inverses to recover ring matrices.
        A_rec = A_from_A_bin(A_bin, gd_s3, (1, 2))
        B_rec = B_from_B_bin(B_bin, gd_s3, (1, 2))
        # Ring matrices should equal canonicalized entries.
        from core.group import canonicalize
        for ia in range(1):
            for ja in range(2):
                assert A_rec[ia][ja] == canonicalize(A[ia][ja])
                assert B_rec[ia][ja] == canonicalize(B[ia][ja])

    def test_non_rectangular(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (2, 3))
        np.testing.assert_array_equal(A_bin, build_A_bin(A, gd_s3))
        np.testing.assert_array_equal(B_bin, build_B_bin(B, gd_s3))

    def test_non_default_shape_b(self, gd_s3, AB_1x2_1x3_s3):
        A, B = AB_1x2_1x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2), (1, 3))
        np.testing.assert_array_equal(A_bin, build_A_bin(A, gd_s3))
        np.testing.assert_array_equal(B_bin, build_B_bin(B, gd_s3))

    def test_hx_shape_mismatch_raises(self, gd_s3):
        # Pass wrong shape_a.
        bad_Hx = np.zeros((12, 30), dtype=np.uint8)   # valid for (1, 2) (1, 2)
        bad_Hz = np.zeros((12, 30), dtype=np.uint8)
        with pytest.raises(ValueError):
            A_bin_B_bin_from_Hx_Hz(bad_Hx, bad_Hz, gd_s3, (2, 3))

    def test_hx_offdiagonal_corruption_raises(self, gd_s3, AB_1x2_s3):
        # Inject a 1 into an off-diagonal block of Hx's left section: the
        # block at row (ia=0, ib=0) × col (ja=0, ib'=1) should be zero in any
        # LP code, so a 1 there must be caught by the new LP-structure check.
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3).copy()
        Hz = build_Hz(A, B, gd_s3)
        n = gd_s3.n
        # block row 0 (ia=0, ib=0) starts at row 0; block col (ja=0, ib'=1)
        # starts at column n (since nb=2 here).
        Hx[0, n] ^= 1
        with pytest.raises(ValueError, match="Hx left section is not LP-consistent"):
            A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))

    def test_hx_diagonal_disagreement_raises(self, gd_s3, AB_1x2_s3):
        # Make the ib=1 diagonal slice of Hx's left section disagree with the
        # ib=0 fiber. The validator should catch this rather than silently
        # returning the ib=0 fiber as A_bin.
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3).copy()
        Hz = build_Hz(A, B, gd_s3)
        n = gd_s3.n
        # Flip a bit at row (ia=0, ib=1, k=0), col (ja=0, ib'=1, k'=0).
        Hx[n, n] ^= 1
        with pytest.raises(ValueError, match="Hx left section is not LP-consistent"):
            A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))

    def test_hz_offdiagonal_corruption_raises(self, gd_s3, AB_1x2_s3):
        # Same idea for Hz: inject a 1 in an off-diagonal block of its left
        # section. With ma=na=1, the ja=0, ja'=... block grid is trivial, so
        # use a 2×2 fixture so there is a real off-diagonal to corrupt.
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3).copy()
        n = gd_s3.n
        # ja=0, ja'=1 means row-block 0 (ja=0, ib=0) × col-block ((ja=1, jb=0)
        # at column 1*nb*n = 2n. Should be zero in any LP code.
        Hz[0, 2 * n] ^= 1
        with pytest.raises(ValueError, match="Hz left section is not LP-consistent"):
            A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))


class TestBuildHxRectangularityCheck:
    def test_jagged_A_raises(self, gd_s3):
        A = [[(0,), (1,)], [(2,)]]  # 2nd row has only 1 col
        B = [[(0,), (1,)]]
        with pytest.raises(ValueError, match="A is not rectangular"):
            build_Hx(A, B, gd_s3)

    def test_jagged_B_raises(self, gd_s3):
        A = [[(0,), (1,)]]
        B = [[(0,), (1,)], [(2,)]]
        with pytest.raises(ValueError, match="B is not rectangular"):
            build_Hx(A, B, gd_s3)

    def test_jagged_A_raises_in_build_Hz(self, gd_s3):
        A = [[(0,), (1,)], [(2,)]]
        B = [[(0,), (1,)]]
        with pytest.raises(ValueError, match="A is not rectangular"):
            build_Hz(A, B, gd_s3)


# ─────────────────────────────────────────────────────────────────
# Inverse — AB_from_Hx_Hz (with cross-check)
# ─────────────────────────────────────────────────────────────────


class TestABFromHxHz:
    def test_round_trip_1x2(self, gd_s3, AB_1x2_s3):
        A, B = AB_1x2_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_rec, B_rec = AB_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2))
        from core.group import canonicalize
        for ia in range(1):
            for ja in range(2):
                assert A_rec[ia][ja] == canonicalize(A[ia][ja])
                assert B_rec[ia][ja] == canonicalize(B[ia][ja])

    def test_round_trip_2x3(self, gd_s3, AB_2x3_s3):
        A, B = AB_2x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_rec, B_rec = AB_from_Hx_Hz(Hx, Hz, gd_s3, (2, 3))
        from core.group import canonicalize
        for ia in range(2):
            for ja in range(3):
                assert A_rec[ia][ja] == canonicalize(A[ia][ja])
                assert B_rec[ia][ja] == canonicalize(B[ia][ja])

    def test_non_default_shape_b(self, gd_s3, AB_1x2_1x3_s3):
        A, B = AB_1x2_1x3_s3
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_rec, B_rec = AB_from_Hx_Hz(Hx, Hz, gd_s3, (1, 2), (1, 3))
        from core.group import canonicalize
        for ia in range(1):
            for ja in range(2):
                assert A_rec[ia][ja] == canonicalize(A[ia][ja])
        for ib in range(1):
            for jb in range(3):
                assert B_rec[ib][jb] == canonicalize(B[ib][jb])

    def test_cross_check_raises_on_inconsistent_hx_hz(self, gd_s3):
        # Build Hx from one (A1, B1) and Hz from a different (A2, B2). The
        # individual lifts may extract fine, but the cross-check rebuild of
        # Hx and Hz won't both match the inputs.
        A1 = [[(gd_s3.identity,), (1, 2)]]
        B1 = [[(gd_s3.identity,), (3,)]]
        A2 = [[(1,), (2,)]]
        B2 = [[(gd_s3.identity,), (4,)]]
        Hx_mismatched = build_Hx(A1, B1, gd_s3)
        Hz_mismatched = build_Hz(A2, B2, gd_s3)
        with pytest.raises(ValueError, match="cross-check failed"):
            AB_from_Hx_Hz(Hx_mismatched, Hz_mismatched, gd_s3, (1, 2))
