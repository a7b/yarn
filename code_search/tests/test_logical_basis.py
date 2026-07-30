"""Tests for logical_basis.find_logical_basis (Type 1)."""

import numpy as np
import pytest

from core.classical_code import build_A_bin, build_B_bin
from core.f2 import f2_rank
from core.group import GroupData
from core.quantum_code import build_Hx, build_Hz, build_quantum_code, check_css
from logical_basis.logical_basis import find_logical_basis

pytestmark = pytest.mark.gap


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gd_c4():
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="session")
def gd_s3():
    return GroupData("SymmetricGroup(3)")


def _build_canonical_code(A, B, gd):
    """Use build_quantum_code so the dep block-col lands at the last position
    (the canonical orbit-pairing requirement)."""
    r = build_quantum_code(A, B, gd)
    return (
        r["Hx"], r["Hz"],
        r["A_bin_canonical"], r["B_bin_canonical"],
        r["has_full_rank_a"], r["has_full_rank_b"],
    )


@pytest.fixture
def code_eligible_c4(gd_c4):
    A = [[(gd_c4.identity,), (1, 2)]]   # canonical_form_A moves identity to LAST
    B = [[(gd_c4.identity,), (1, 2)]]
    Hx, Hz, A_bin, B_bin, ha, hb = _build_canonical_code(A, B, gd_c4)
    assert ha and hb, "fixture must have a structured canonical basis on both sides"
    return Hx, Hz, A_bin, B_bin, 4   # n = |C4|


@pytest.fixture
def code_eligible_s3(gd_s3):
    A = [[(gd_s3.identity,), (1, 2)]]
    B = [[(gd_s3.identity,), (1, 2)]]
    Hx, Hz, A_bin, B_bin, ha, hb = _build_canonical_code(A, B, gd_s3)
    assert ha and hb, "fixture must have a structured canonical basis on both sides"
    return Hx, Hz, A_bin, B_bin, 6


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_expected_keys(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        expected = {
            "Lx", "Lz", "Lz_groups",
            "k", "n", "n_groups",
            "Z_group_tags", "X_group_tags",
            "full_rank_block_cols_A", "full_rank_block_cols_B",
            "Hx_canonical", "Hz_canonical",
            "perm_a", "perm_b",
        }
        assert set(r.keys()) == expected

    def test_Lz_groups_shape_and_reshape_invariant(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        assert r["Lz_groups"].shape == (r["n_groups"], r["n"], Hx.shape[1])
        np.testing.assert_array_equal(
            r["Lz_groups"].reshape(r["k"], -1), r["Lz"]
        )

    def test_k_matches_n_phys_minus_ranks(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        expected_k = Hx.shape[1] - f2_rank(Hx) - f2_rank(Hz)
        assert r["k"] == expected_k

    def test_n_groups_equals_k_over_n(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        assert r["n_groups"] == r["k"] // r["n"]

    def test_shapes_flat(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        n_phys = Hx.shape[1]
        assert r["Lx"].shape == (r["k"], n_phys)
        assert r["Lz"].shape == (r["k"], n_phys)
        assert r["Lx"].dtype == np.uint8
        assert r["Lz"].dtype == np.uint8


# ─────────────────────────────────────────────────────────────────
# Mathematical invariants
# ─────────────────────────────────────────────────────────────────


class TestInvariants:
    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_Lz_in_ker_Hx(self, request, fixture_name):
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        assert check_css(Hx, Hz)
        prod = (Hx.astype(int) @ r["Lz"].T.astype(int)) % 2
        assert np.all(prod == 0)

    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_Lx_in_ker_Hz(self, request, fixture_name):
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        prod = (Hz.astype(int) @ r["Lx"].T.astype(int)) % 2
        assert np.all(prod == 0)

    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_pairing_invariant(self, request, fixture_name):
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        I_k = np.eye(r["k"], dtype=np.uint8)
        pairing = (r["Lx"].astype(int) @ r["Lz"].T.astype(int)) % 2
        np.testing.assert_array_equal(pairing.astype(np.uint8), I_k)

    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_Lx_logicals_not_stabilizers(self, request, fixture_name):
        # Each Lx row should be a genuine logical, i.e. independent of Hx
        # rowspace. Equivalently: stacking Lx on top of Hx should add k to
        # the rank.
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        rank_with = f2_rank(np.vstack([Hx, r["Lx"]]))
        assert rank_with == f2_rank(Hx) + r["k"]

    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_Lz_logicals_not_stabilizers(self, request, fixture_name):
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        rank_with = f2_rank(np.vstack([Hz, r["Lz"]]))
        assert rank_with == f2_rank(Hz) + r["k"]

    @pytest.mark.parametrize("fixture_name",
                             ["code_eligible_c4", "code_eligible_s3"])
    def test_logical_rank_increment_invariant(self, request, fixture_name):
        """The defining CSS invariant for a paired logical basis:

            rank([Hx; Lx]) − rank(Hx) = k
            rank([Hz; Lz]) − rank(Hz) = k

        i.e. ``Lx`` extends the X-stabilizer rowspan by EXACTLY k
        independent rows (none of them are stabilizers, and they're
        mutually independent), and same on the Z side. Combined with
        ``Lx · Lz.T = I_k`` this is the textbook definition of a paired
        logical basis.
        """
        Hx, Hz, A_bin, B_bin, n = request.getfixturevalue(fixture_name)
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        k = r["k"]
        dx_rank = f2_rank(np.vstack([Hx, r["Lx"]])) - f2_rank(Hx)
        dz_rank = f2_rank(np.vstack([Hz, r["Lz"]])) - f2_rank(Hz)
        assert dx_rank == k, (
            f"rank([Hx; Lx]) − rank(Hx) = {dx_rank}, expected k = {k}"
        )
        assert dz_rank == k, (
            f"rank([Hz; Lz]) − rank(Hz) = {dz_rank}, expected k = {k}"
        )
        # Symmetric — these are equal by the previous two asserts, but
        # call it out explicitly as the user-requested invariant.
        assert dx_rank == dz_rank


# ─────────────────────────────────────────────────────────────────
# Eligibility / error paths (no fallback)
# ─────────────────────────────────────────────────────────────────


class TestRaisesOnIneligible:
    def test_raises_when_no_full_rank_block_col(self, gd_s3):
        # Use entries that are pure weight-2 with no identity → typically
        # no single block-col is full rank for shape (1, 2) over S3.
        # If we get lucky and one IS full rank, swap entries until we find
        # an ineligible case. For S3 with weight-2 entries (g1, g2) where
        # g1, g2 generate a subgroup of order < 6, the binary lift block
        # is rank-deficient.
        A = [[(1, 2), (3, 4)]]   # weight-2 entries, no identity
        B = [[(1, 2), (3, 4)]]
        Hx = build_Hx(A, B, gd_s3)
        Hz = build_Hz(A, B, gd_s3)
        A_bin = build_A_bin(A, gd_s3)
        B_bin = build_B_bin(B, gd_s3)
        # Both block-cols of A_bin are weight-2 lifts; check if at least one
        # is rank 6 (full rank). If not, the call must raise.
        n = gd_s3.n
        from core.classical_code import canonical_form_A, canonical_form_B
        _, _, _, ha = canonical_form_A(A, gd_s3)
        _, _, _, hb = canonical_form_B(B, gd_s3)
        if not (ha and hb):
            with pytest.raises(ValueError,
                               match="no structured canonical basis|not a multiple"):
                find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        else:
            pytest.skip("happened to get an eligible code; flip entries")

    def test_raises_on_k_not_multiple_of_n(self):
        # Build a tiny non-LP code by hand where k is not a multiple of n.
        # Trivial Hx/Hz on n_phys=3 with k=1 and we'll claim n=2.
        # We can't really do this through the LP construction, so this test
        # just confirms the explicit check fires when the prerequisite fails.
        # Use 2 disjoint copies of a small classical (3,1,3) repetition code.
        Hx = np.array([[1, 1, 0],
                       [0, 1, 1]], dtype=np.uint8)
        Hz = np.zeros((0, 3), dtype=np.uint8)
        # Bogus A_bin/B_bin/shape_a — the k % n check happens before
        # _find_full_rank_block_cols, so they don't matter.
        A_bin = np.zeros((2, 6), dtype=np.uint8)
        B_bin = np.zeros((2, 6), dtype=np.uint8)
        with pytest.raises(ValueError, match="not a multiple"):
            find_logical_basis(Hx, Hz, A_bin, B_bin, 2, (1, 3))


# ─────────────────────────────────────────────────────────────────
# Save-to-disk
# ─────────────────────────────────────────────────────────────────


class TestCanonicalizeOption:
    def test_canonicalize_returns_Hx_Hz_canonical(self, gd_c4):
        # Pass NON-canonical inputs (identity entry at the first position
        # → canonical_form_* will move it to the last). canonicalize=True
        # should accept that and run on the rebuilt canonical pair.
        from core.classical_code import build_A_bin, build_B_bin
        from core.quantum_code import build_Hx, build_Hz

        A = [[(gd_c4.identity,), (1, 2)]]   # identity at block-col 0 (first)
        B = [[(gd_c4.identity,), (1, 2)]]
        # NON-canonical Hx, Hz, A_bin, B_bin.
        Hx_raw = build_Hx(A, B, gd_c4)
        Hz_raw = build_Hz(A, B, gd_c4)
        A_bin_raw = build_A_bin(A, gd_c4)
        B_bin_raw = build_B_bin(B, gd_c4)

        r = find_logical_basis(
            Hx_raw, Hz_raw, A_bin_raw, B_bin_raw, gd_c4.n, (1, 2),
            canonicalize=True, gd=gd_c4,
        )
        # Canonical Hx, Hz returned.
        assert r["Hx_canonical"] is not None
        assert r["Hz_canonical"] is not None
        # Lz, Lx must satisfy the canonical-side parity checks.
        np.testing.assert_array_equal(
            (r["Hx_canonical"].astype(int) @ r["Lz"].T.astype(int)) % 2,
            np.zeros(r["Hx_canonical"].shape[0], dtype=int)[:, None].repeat(
                r["k"], axis=1
            ).T.T,
        )

    def test_canonicalize_requires_gd(self, code_eligible_c4):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        with pytest.raises(ValueError, match="requires `gd`"):
            find_logical_basis(
                Hx, Hz, A_bin, B_bin, n, (1, 2),
                canonicalize=True,   # but no gd
            )

    def test_canonicalize_false_leaves_Hx_Hz_canonical_None(
        self, code_eligible_c4,
    ):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(Hx, Hz, A_bin, B_bin, n, (1, 2))
        assert r["Hx_canonical"] is None
        assert r["Hz_canonical"] is None
        assert r["perm_a"] is None
        assert r["perm_b"] is None


class TestSaveDir:
    def test_writes_npy_and_meta(self, code_eligible_c4, tmp_path):
        Hx, Hz, A_bin, B_bin, n = code_eligible_c4
        r = find_logical_basis(
            Hx, Hz, A_bin, B_bin, n, (1, 2),
            save_dir=tmp_path,
        )
        assert (tmp_path / "Lx.npy").exists()
        assert (tmp_path / "Lz.npy").exists()
        assert (tmp_path / "logical_basis_meta.json").exists()
        Lx_loaded = np.load(tmp_path / "Lx.npy")
        Lz_loaded = np.load(tmp_path / "Lz.npy")
        np.testing.assert_array_equal(Lx_loaded, r["Lx"])
        np.testing.assert_array_equal(Lz_loaded, r["Lz"])

        import json
        with open(tmp_path / "logical_basis_meta.json") as f:
            meta = json.load(f)
        assert meta["k"] == r["k"]
        assert meta["n"] == r["n"]
        assert meta["full_rank_block_cols_A"] == r["full_rank_block_cols_A"]
