"""Fresh-eye black-box contract tests for ``logical_basis/logical_basis.py``.

Every returned basis is checked against the full paired-logical-basis
contract with an INDEPENDENT GF(2) rank oracle written in this file (no
``core.f2`` reuse):

    Lx rows in ker(Hz),  Lz rows in ker(Hx)          (kernel constraints)
    Lx @ Lz.T == I_k exactly (mod 2)                  (symplectic pairing)
    rank([Hx; Lx]) - rank(Hx) == k                    (X rows are k genuine,
    rank([Hz; Lz]) - rank(Hz) == k                     independent logicals)
    k == n_phys - rank(Hx) - rank(Hz)
    shapes (k, n_phys), dtype uint8

on a zoo of hand-written CSS codes NOT used by the existing suites
(Shor, cube [[8,3,2]], asymmetric hypergraph products, direct sums,
redundant-row presentations) plus LP codes over hand-rolled group shims
(C3, C6, dihedral D4 of order 8 — non-abelian), which also exercise the
G-orbit path GAP-free, including its documented uniform-row-weight
property and the multi-orbit ``P^-1`` fallback that no existing test hits.

Markers: ``fast`` everywhere except the two ``gap``-marked GroupData
cross-checks (GAP-built groups must reproduce the shim-built invariants).
"""

from __future__ import annotations

import numpy as np
import pytest

from logical_basis.logical_basis import (
    find_logical_basis,
    find_logical_basis_pivot_aligned,
    find_logical_noncanonical_RREF,
    reorder_by_pivot,
)


# ─────────────────────────────────────────────────────────────────
# Independent GF(2) rank oracle (deliberately NOT core.f2)
# ─────────────────────────────────────────────────────────────────


def _oracle_rank(M) -> int:
    M = (np.atleast_2d(np.asarray(M)) % 2).astype(np.uint8).copy()
    if M.size == 0:
        return 0
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        hit = np.flatnonzero(M[r:, c])
        if hit.size == 0:
            continue
        p = r + int(hit[0])
        M[[r, p]] = M[[p, r]]
        for o in np.flatnonzero(M[:, c]):
            if o != r:
                M[o] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def assert_paired_logical_basis(Hx, Hz, Lx, Lz):
    """The full contract for a paired CSS logical basis (see module doc)."""
    n_phys = Hx.shape[1]
    k = n_phys - _oracle_rank(Hx) - _oracle_rank(Hz)
    assert k >= 0
    assert Lx.shape == (k, n_phys), f"Lx shape {Lx.shape} != ({k}, {n_phys})"
    assert Lz.shape == (k, n_phys), f"Lz shape {Lz.shape} != ({k}, {n_phys})"
    assert Lx.dtype == np.uint8 and Lz.dtype == np.uint8
    if k == 0:
        return
    # kernel constraints
    assert not ((Hz.astype(np.int64) @ Lx.T.astype(np.int64)) % 2).any(), \
        "some Lx row is not in ker(Hz)"
    assert not ((Hx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2).any(), \
        "some Lz row is not in ker(Hx)"
    # exact symplectic pairing
    P = (Lx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2
    np.testing.assert_array_equal(P, np.eye(k, dtype=np.int64),
                                  err_msg="Lx @ Lz.T != I_k")
    # rank increments: k genuine, mutually independent non-stabilizers
    assert _oracle_rank(np.vstack([Hx, Lx])) == _oracle_rank(Hx) + k
    assert _oracle_rank(np.vstack([Hz, Lz])) == _oracle_rank(Hz) + k


# ─────────────────────────────────────────────────────────────────
# Code zoo (hand-written)
# ─────────────────────────────────────────────────────────────────


HAMMING = np.array([[1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)


def rep_code(n):
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = H[i, i + 1] = 1
    return H


def steane():
    return HAMMING.copy(), HAMMING.copy()


def shor():
    Hz = np.zeros((6, 9), dtype=np.uint8)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]):
        Hz[i, a] = Hz[i, b] = 1
    Hx = np.zeros((2, 9), dtype=np.uint8)
    Hx[0, 0:6] = 1
    Hx[1, 3:9] = 1
    return Hx, Hz


def css422():
    return np.ones((1, 4), dtype=np.uint8), np.ones((1, 4), dtype=np.uint8)


def cube832():
    Hx = np.ones((1, 8), dtype=np.uint8)
    faces = [[v for v in range(8) if ((v >> axis) & 1) == 0] for axis in range(3)]
    faces.append([v for v in range(8) if ((v >> 2) & 1) == 1])
    Hz = np.zeros((4, 8), dtype=np.uint8)
    for i, f in enumerate(faces):
        Hz[i, f] = 1
    return Hx, Hz


def hgp32():
    """HGP(rep3, rep2): [[8, 1]] with Hx != Hz (asymmetric)."""
    H1, H2 = rep_code(3), rep_code(2)
    Hx = np.hstack([np.kron(H1, np.eye(2, dtype=np.uint8)),
                    np.kron(np.eye(2, dtype=np.uint8), H2.T)]).astype(np.uint8)
    Hz = np.hstack([np.kron(np.eye(3, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(1, dtype=np.uint8))]).astype(np.uint8)
    return Hx, Hz


def direct_sum(codes):
    """Block-diagonal direct sum of CSS pairs (k adds up)."""
    def blk(mats):
        rows = sum(m.shape[0] for m in mats)
        cols = sum(m.shape[1] for m in mats)
        out = np.zeros((rows, cols), dtype=np.uint8)
        r = c = 0
        for m in mats:
            out[r:r + m.shape[0], c:c + m.shape[1]] = m
            r += m.shape[0]
            c += m.shape[1]
        return out
    return blk([Hx for Hx, _ in codes]), blk([Hz for _, Hz in codes])


def steane_redundant():
    """Steane with duplicated + summed rows on both sides (rank unchanged)."""
    Hx, Hz = steane()
    Hx_r = np.vstack([Hx, Hx[0:1], (Hx[0] ^ Hx[1])[None, :]])
    Hz_r = np.vstack([Hz, Hz[2:3], (Hz[1] ^ Hz[2])[None, :]])
    return Hx_r, Hz_r


class CyclicShim:
    """Minimal GroupData-like C_n (the builders read only n/mult/inv/identity)."""

    def __init__(self, n: int):
        self.n = n
        self.mult = [[(i + j) % n for j in range(n)] for i in range(n)]
        self.inv = [(-i) % n for i in range(n)]
        self.identity = 0


class DihedralShim:
    """Minimal GroupData-like D_m (order 2m, NON-abelian for m >= 3).
    Element i = r^(i mod m) * f^(i // m); verified group axioms in __init__."""

    def __init__(self, m: int):
        n = 2 * m

        def mul(a, b):
            ra, fa = a % m, a // m
            rb, fb = b % m, b // m
            r = (ra + rb) % m if fa == 0 else (ra - rb) % m
            return r + m * (fa ^ fb)

        self.n = n
        self.mult = [[mul(i, j) for j in range(n)] for i in range(n)]
        self.inv = [next(j for j in range(n) if self.mult[i][j] == 0)
                    for i in range(n)]
        self.identity = 0
        # sanity: identity behavior + associativity spot checks
        assert all(self.mult[0][j] == j and self.mult[j][0] == j
                   for j in range(n))
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b, c = (int(x) for x in rng.integers(0, n, size=3))
            assert self.mult[self.mult[a][b]][c] == self.mult[a][self.mult[b][c]]


def lp_code(gd, A, B):
    """Canonical-form LP code dict (the orbit path needs the canonical lifts)."""
    from core.quantum_code import build_quantum_code, check_css
    r = build_quantum_code(A, B, gd)
    assert check_css(r["Hx"], r["Hz"])
    return r


HAND_CODES = [
    ("steane", steane),
    ("shor", shor),
    ("css422", css422),
    ("cube832", cube832),
    ("hgp32", hgp32),
    ("steane_plus_422", lambda: direct_sum([steane(), css422()])),
    ("steane_redundant_rows", steane_redundant),
]


# ─────────────────────────────────────────────────────────────────
# find_logical_noncanonical_RREF / find_logical_basis_pivot_aligned
# ─────────────────────────────────────────────────────────────────


BASIS_FNS = [
    ("rref", find_logical_noncanonical_RREF),
    ("pivot_aligned", find_logical_basis_pivot_aligned),
]


@pytest.mark.fast
class TestGenericBasisContracts:
    @pytest.mark.parametrize("fn_name,fn", BASIS_FNS,
                             ids=[f[0] for f in BASIS_FNS])
    @pytest.mark.parametrize("code_name,builder", HAND_CODES,
                             ids=[c[0] for c in HAND_CODES])
    def test_full_contract(self, fn_name, fn, code_name, builder):
        Hx, Hz = builder()
        # zoo sanity: really CSS
        assert not ((Hx.astype(int) @ Hz.T.astype(int)) % 2).any()
        Lx, Lz = fn(Hx, Hz)
        assert_paired_logical_basis(Hx, Hz, Lx, Lz)

    @pytest.mark.parametrize("fn_name,fn", BASIS_FNS,
                             ids=[f[0] for f in BASIS_FNS])
    def test_expected_k_values(self, fn_name, fn):
        """Independently known k values: Shor k=1, cube k=3, Steane+[[4,2,2]]
        direct sum k=3."""
        for builder, k_expect in [(shor, 1), (cube832, 3),
                                  (lambda: direct_sum([steane(), css422()]), 3)]:
            Hx, Hz = builder()
            Lx, Lz = fn(Hx, Hz)
            assert Lx.shape[0] == k_expect
            assert Lz.shape[0] == k_expect

    @pytest.mark.parametrize("fn_name,fn", BASIS_FNS,
                             ids=[f[0] for f in BASIS_FNS])
    def test_k0_returns_empty_pair(self, fn_name, fn):
        Hx = np.eye(5, dtype=np.uint8)
        Hz = np.zeros((0, 5), dtype=np.uint8)
        Lx, Lz = fn(Hx, Hz)
        assert Lx.shape == (0, 5) and Lz.shape == (0, 5)
        assert Lx.dtype == np.uint8 and Lz.dtype == np.uint8

    def test_rref_column_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of columns"):
            find_logical_noncanonical_RREF(np.eye(3, dtype=np.uint8),
                                           np.eye(4, dtype=np.uint8))

    @pytest.mark.parametrize("fn_name,fn", BASIS_FNS,
                             ids=[f[0] for f in BASIS_FNS])
    def test_inputs_not_mutated(self, fn_name, fn):
        Hx, Hz = cube832()
        Hx_copy, Hz_copy = Hx.copy(), Hz.copy()
        fn(Hx, Hz)
        np.testing.assert_array_equal(Hx, Hx_copy)
        np.testing.assert_array_equal(Hz, Hz_copy)


# ─────────────────────────────────────────────────────────────────
# reorder_by_pivot
# ─────────────────────────────────────────────────────────────────


def _pivot(row):
    nz = np.flatnonzero(row)
    return int(nz[0]) if nz.size else row.shape[0]


@pytest.mark.fast
class TestReorderByPivotFresh:
    def _random_pair(self, k=6, n=14, seed=3):
        rng = np.random.default_rng(seed)
        Lx = (rng.random((k, n)) < 0.4).astype(np.uint8)
        Lz = (rng.random((k, n)) < 0.4).astype(np.uint8)
        # ensure no all-zero Lz rows and distinct pivots for a strict check
        for i in range(k):
            Lz[i, i] = 1
            Lz[i, :i] = 0
        return Lx, Lz

    def test_couples_move_together(self):
        """The SAME permutation must be applied to both matrices: every
        output (Lx2[i], Lz2[i]) couple must be an input couple, and the
        output must be a permutation of the inputs."""
        Lx, Lz = self._random_pair()
        Lx2, Lz2 = reorder_by_pivot(Lx, Lz, ref="Lz")
        in_couples = {(tuple(a), tuple(b)) for a, b in zip(Lx, Lz)}
        out_couples = {(tuple(a), tuple(b)) for a, b in zip(Lx2, Lz2)}
        assert in_couples == out_couples
        # and the ref pivots are ascending
        pivots = [_pivot(r) for r in Lz2]
        assert pivots == sorted(pivots)

    def test_ref_lx_sorts_by_lx(self):
        Lx, Lz = self._random_pair(seed=5)
        # give Lx the structured pivots instead
        Lx, Lz = Lz, Lx
        Lx2, Lz2 = reorder_by_pivot(Lx, Lz, ref="Lx")
        pivots = [_pivot(r) for r in Lx2]
        assert pivots == sorted(pivots)
        in_couples = {(tuple(a), tuple(b)) for a, b in zip(Lx, Lz)}
        out_couples = {(tuple(a), tuple(b)) for a, b in zip(Lx2, Lz2)}
        assert in_couples == out_couples

    def test_zero_row_sorts_last(self):
        Lz = np.array([[0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0]], dtype=np.uint8)
        Lx = np.array([[1, 1, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 1]], dtype=np.uint8)
        Lx2, Lz2 = reorder_by_pivot(Lx, Lz, ref="Lz")
        np.testing.assert_array_equal(Lz2[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(Lz2[1], [0, 0, 1, 0])
        np.testing.assert_array_equal(Lz2[2], [0, 0, 0, 0])   # zero row last
        # couples preserved
        np.testing.assert_array_equal(Lx2[2], [1, 1, 0, 0])

    def test_identity_pairing_preserved_after_row_shuffle(self):
        """Take a paired basis (Lx @ Lz.T = I), shuffle rows with one fixed
        permutation (pairing stays I), reorder — pairing must STILL be I."""
        Hx, Hz = direct_sum([steane(), css422()])
        Lx, Lz = find_logical_basis_pivot_aligned(Hx, Hz)
        k = Lx.shape[0]
        perm = np.array([2, 0, 1])
        Lx_s, Lz_s = Lx[perm], Lz[perm]
        Lx2, Lz2 = reorder_by_pivot(Lx_s, Lz_s, ref="Lz")
        P = (Lx2.astype(int) @ Lz2.T.astype(int)) % 2
        np.testing.assert_array_equal(P, np.eye(k, dtype=int))


# ─────────────────────────────────────────────────────────────────
# G-orbit path (find_logical_basis) over hand-rolled shims — GAP-free
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
class TestOrbitBasisShim:
    @pytest.mark.parametrize("n", [3, 6], ids=["C3", "C6"])
    def test_single_orbit_cyclic_contract_and_uniform_weights(self, n):
        """1x2 LP code over C_n with A = B = [[1, x + x^2]]: k = |G| (a
        single orbit). Full paired-basis contract PLUS the documented
        orbit-preserving property: on the primary path both Lx and Lz have
        UNIFORM row weight; the orbit metadata is consistent."""
        r = lp_code(CyclicShim(n), [[(0,), (1, 2)]], [[(0,), (1, 2)]])
        Hx, Hz = r["Hx"], r["Hz"]
        b = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], n, (1, 2))
        assert_paired_logical_basis(Hx, Hz, b["Lx"], b["Lz"])
        assert b["k"] == n and b["n"] == n and b["n_groups"] == 1
        # documented: uniform operator weight per direction (single orbit)
        assert np.unique(b["Lx"].sum(axis=1)).size == 1
        assert np.unique(b["Lz"].sum(axis=1)).size == 1
        # orbit metadata
        assert b["Lz_groups"].shape == (1, n, Hx.shape[1])
        np.testing.assert_array_equal(
            b["Lz_groups"].reshape(b["k"], -1), b["Lz"])
        assert len(b["Z_group_tags"]) == 1 and len(b["X_group_tags"]) == 1
        # no accidental canonicalize side-channel
        assert b["Hx_canonical"] is None and b["perm_a"] is None

    def test_single_orbit_dihedral_nonabelian(self):
        """Same contract on the NON-abelian dihedral group of order 8
        (existing suites only cover S3 among non-abelian groups)."""
        gd = DihedralShim(4)
        r = lp_code(gd, [[(0,), (1, 4)]], [[(0,), (1, 4)]])
        Hx, Hz = r["Hx"], r["Hz"]
        b = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 8, (1, 2))
        assert_paired_logical_basis(Hx, Hz, b["Lx"], b["Lz"])
        assert b["k"] == 8 and b["n_groups"] == 1
        assert np.unique(b["Lx"].sum(axis=1)).size == 1
        assert np.unique(b["Lz"].sum(axis=1)).size == 1

    def test_multi_orbit_fallback_path_full_contract(self):
        """1x3 LP code over C3 (k = 12 = 4 orbits): n_groups > 1 forces the
        ``P^-1`` fallback pairing, which NO existing test exercises. The
        full paired-basis contract must still hold; Lz must stay orbit-pure
        (uniform weight inside each Lz_groups slice); tags match n_groups."""
        A = [[(0,), (1, 2), (0, 1)]]
        r = lp_code(CyclicShim(3), A, A)
        Hx, Hz = r["Hx"], r["Hz"]
        b = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 3,
            shape_a=(1, 3), shape_b=(1, 3))
        assert b["k"] == 12 and b["n_groups"] == 4
        assert_paired_logical_basis(Hx, Hz, b["Lx"], b["Lz"])
        assert b["Lz_groups"].shape == (4, 3, Hx.shape[1])
        np.testing.assert_array_equal(
            b["Lz_groups"].reshape(12, -1), b["Lz"])
        for j in range(4):   # Lz stays orbit-pure: uniform weight per orbit
            assert np.unique(b["Lz_groups"][j].sum(axis=1)).size == 1
        assert len(b["Z_group_tags"]) == 4
        assert len(b["X_group_tags"]) == 4

    def test_shape_b_defaults_to_shape_a(self):
        """Passing shape_b=None must be identical to shape_b=shape_a."""
        r = lp_code(CyclicShim(3), [[(0,), (1, 2)]], [[(0,), (1, 2)]])
        Hx, Hz = r["Hx"], r["Hz"]
        args = (Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 3)
        b_default = find_logical_basis(*args, (1, 2))
        b_explicit = find_logical_basis(*args, (1, 2), (1, 2))
        np.testing.assert_array_equal(b_default["Lx"], b_explicit["Lx"])
        np.testing.assert_array_equal(b_default["Lz"], b_explicit["Lz"])

    def test_all_three_basis_functions_agree_on_k(self):
        """Orbit path, RREF path, and pivot-aligned path must report the
        same k on the same code — and all satisfy the contract."""
        r = lp_code(CyclicShim(6), [[(0,), (1, 2)]], [[(0,), (1, 2)]])
        Hx, Hz = r["Hx"], r["Hz"]
        b = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 6, (1, 2))
        for Lx, Lz in [
            (b["Lx"], b["Lz"]),
            find_logical_noncanonical_RREF(Hx, Hz),
            find_logical_basis_pivot_aligned(Hx, Hz),
        ]:
            assert Lx.shape[0] == b["k"]
            assert_paired_logical_basis(Hx, Hz, Lx, Lz)


# ─────────────────────────────────────────────────────────────────
# GAP-built GroupData must reproduce the shim-built invariants
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
class TestOrbitBasisGapCrossCheck:
    def test_gap_c6_matches_shim_invariants(self):
        """GroupData('CyclicGroup(6)') and the hand-rolled C6 shim may order
        group elements differently (the codes differ by a qubit relabeling),
        but every relabeling-invariant quantity must agree: k, n_groups, and
        the uniform orbit weight profiles of Lx and Lz."""
        from core.group import GroupData
        results = {}
        for tag, gd in [("gap", GroupData("CyclicGroup(6)")),
                        ("shim", CyclicShim(6))]:
            A = [[(gd.identity,), (1, 2)]]
            r = lp_code(gd, A, A)
            b = find_logical_basis(
                r["Hx"], r["Hz"], r["A_bin_canonical"], r["B_bin_canonical"],
                6, (1, 2))
            assert_paired_logical_basis(r["Hx"], r["Hz"], b["Lx"], b["Lz"])
            results[tag] = (
                b["k"], b["n_groups"],
                sorted(set(b["Lx"].sum(axis=1).tolist())),
                sorted(set(b["Lz"].sum(axis=1).tolist())),
            )
        assert results["gap"] == results["shim"]
        assert results["gap"][0] == 6 and results["gap"][1] == 1

    def test_gap_dihedral8_contract(self):
        """Non-abelian order-8 group via GAP (DihedralGroup(8)): the orbit
        path must produce a contract-satisfying, uniform-weight basis."""
        from core.group import GroupData
        gd = GroupData("DihedralGroup(8)")
        A = [[(gd.identity,), (1, 2)]]
        r = lp_code(gd, A, A)
        b = find_logical_basis(
            r["Hx"], r["Hz"], r["A_bin_canonical"], r["B_bin_canonical"],
            8, (1, 2))
        assert_paired_logical_basis(r["Hx"], r["Hz"], b["Lx"], b["Lz"])
        assert b["k"] == 8 and b["n_groups"] == 1
        assert np.unique(b["Lx"].sum(axis=1)).size == 1
        assert np.unique(b["Lz"].sum(axis=1)).size == 1
