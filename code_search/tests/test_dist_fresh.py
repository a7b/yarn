"""Fresh-eye black-box tests for the distance-estimation stack (``core/dist``).

Everything here treats the implementation as UNTRUSTED: expected values come
from a self-contained brute-force oracle written in this file (independent
GF(2) linear algebra — no ``core.f2`` reuse) that exhaustively enumerates
codewords on codes small enough for that (kernel dimension <= ~18).

Cross-checks:
  * classical BP+OSD and sqetch estimates == exact brute-force distance;
  * quantum BP+OSD and sqetch (dx, dz) == exact per-direction brute force,
    on codes with dx != dz (cube [[8,3,2]] has (dx,dz)=(4,2); the hypergraph
    product of rep(3) x rep(2) has (2,3)) so direction swaps cannot hide;
  * the Shor code guards degeneracy: its Z side has weight-2 STABILIZERS
    while the true dz is 3 — an estimator that counts stabilizers as
    logicals reports 2 and fails here;
  * BP+OSD vs sqetch vs brute force on the same code;
  * ``stab_symplectic`` (non-CSS sampler) == brute-force symplectic-weight
    enumeration over centralizer \\ stabilizer-span, including a scrambled
    (genuinely non-CSS) presentation.

Estimates are upper bounds that converge on tiny instances: equality is
asserted only with generous trial budgets (verified stable during authoring);
low-budget runs assert soundness (``est is None or est >= d_true``) only.

Markers: ``fast`` (pure numpy), ``bposd`` (ldpc), ``gpu`` (sqetch/CUDA).
No GAP needed anywhere in this file — group-algebra codes are built over a
hand-rolled cyclic-group shim (only ``n / mult / inv / identity`` are read).
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 1
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="no CUDA device")


# ─────────────────────────────────────────────────────────────────
# Independent GF(2) oracle (deliberately NOT core.f2)
# ─────────────────────────────────────────────────────────────────


def _oracle_rref(M):
    """RREF over GF(2). Returns (R, pivot_cols); R keeps nonzero rows only."""
    M = (np.atleast_2d(np.asarray(M)) % 2).astype(np.uint8).copy()
    rows, cols = M.shape
    piv: list[int] = []
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
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], piv


def _oracle_rank(M) -> int:
    M = np.atleast_2d(np.asarray(M))
    if M.size == 0:
        return 0
    return _oracle_rref(M)[0].shape[0]


def _oracle_nullspace(M) -> np.ndarray:
    """Basis of ker(M) over GF(2), shape (dim, n). Verified M @ basis.T == 0."""
    M = (np.atleast_2d(np.asarray(M)) % 2).astype(np.uint8)
    _, n = M.shape
    R, piv = _oracle_rref(M)
    free = [c for c in range(n) if c not in piv]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, fc in enumerate(free):
        basis[i, fc] = 1
        for ri, pc in enumerate(piv):
            if R[ri, fc]:
                basis[i, pc] = 1
    assert not ((M.astype(np.int64) @ basis.T.astype(np.int64)) % 2).any()
    return basis


def _all_span(basis) -> np.ndarray:
    """All 2^r vectors in rowspan(basis) (r <= 20 guard)."""
    basis = np.atleast_2d(np.asarray(basis)).astype(np.uint8)
    r, n = basis.shape
    if r == 0:
        return np.zeros((1, n), dtype=np.uint8)
    assert r <= 20, "brute-force span too large"
    idx = np.arange(1 << r, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(r)[None, :]) & 1).astype(np.int64)
    return ((bits @ basis.astype(np.int64)) % 2).astype(np.uint8)


def _in_rowspan(rows, v) -> bool:
    rows = np.atleast_2d(rows)
    return _oracle_rank(np.vstack([rows, np.atleast_2d(v)])) == _oracle_rank(rows)


def brute_classical_distance(H):
    """Exact d(ker H) by full enumeration; None for a trivial kernel."""
    ker = _oracle_nullspace(H)
    if ker.shape[0] == 0:
        return None
    w = _all_span(ker).sum(axis=1)
    w = w[w > 0]
    return int(w.min())


def brute_css_distances(Hx, Hz):
    """Exact (dx, dz): dx = min weight over ker(Hz) \\ rowspan(Hx); dually dz."""
    def one(H_ker, H_span):
        ker = _oracle_nullspace(H_ker)
        if ker.shape[0] == 0:
            return None
        R, piv = _oracle_rref(H_span)
        best = None
        for v in _all_span(ker):
            w = int(v.sum())
            if w == 0 or (best is not None and w >= best):
                continue
            vv = v.copy()
            for ri, pc in enumerate(piv):
                if vv[pc]:
                    vv ^= R[ri]
            if vv.any():
                best = w
        return best
    return one(Hz, Hx), one(Hx, Hz)


def brute_stab_distance(HX, HZ):
    """Exact stabilizer-code distance: min symplectic weight over the
    centralizer minus the stabilizer rowspan. Centralizer = ker of the
    Omega-swapped generator matrix [HZ | HX]."""
    HX = np.atleast_2d(HX).astype(np.uint8)
    HZ = np.atleast_2d(HZ).astype(np.uint8)
    W = HX.shape[1]
    cent = _oracle_nullspace(np.hstack([HZ, HX]))
    Rs, piv = _oracle_rref(np.hstack([HX, HZ]))
    best = None
    for v in _all_span(cent):
        w = int((v[:W] | v[W:]).sum())
        if w == 0 or (best is not None and w >= best):
            continue
        vv = v.copy()
        for ri, pc in enumerate(piv):
            if vv[pc]:
                vv ^= Rs[ri]
        if vv.any():
            best = w
    return best


def brute_coset_min_weight(GX, GZ, vx, vz):
    """Exact min symplectic weight over the affine coset v + rowspan(G)."""
    GX = np.atleast_2d(GX).astype(np.uint8)
    GZ = np.atleast_2d(GZ).astype(np.uint8)
    W = GX.shape[1]
    Rs, _ = _oracle_rref(np.hstack([GX, GZ]))
    v = np.concatenate([vx, vz]).astype(np.uint8)
    ws = [int(((v ^ g)[:W] | (v ^ g)[W:]).sum()) for g in _all_span(Rs)]
    return min(ws)


# ─────────────────────────────────────────────────────────────────
# Code zoo (hand-written; all sizes brute-forceable)
# ─────────────────────────────────────────────────────────────────


def rep_code(n) -> np.ndarray:
    """[n, 1, n] repetition code parity checks."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = H[i, i + 1] = 1
    return H


HAMMING = np.array([[1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)


def ext_hamming() -> np.ndarray:
    """[8, 4, 4] extended Hamming: Hamming(7,4) plus an overall parity row."""
    H = np.zeros((4, 8), dtype=np.uint8)
    H[:3, :7] = HAMMING
    H[3, :] = 1
    return H


def steane():
    """[[7, 1, 3]]: Hx = Hz = Hamming."""
    return HAMMING.copy(), HAMMING.copy()


def shor():
    """[[9, 1, 3]] Shor code. The Z side is DEGENERATE: Hz rows are weight-2
    stabilizers, strictly lighter than the true dz = 3. Any estimator that
    counts stabilizers as logicals reports dz = 2 here."""
    Hz = np.zeros((6, 9), dtype=np.uint8)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]):
        Hz[i, a] = Hz[i, b] = 1
    Hx = np.zeros((2, 9), dtype=np.uint8)
    Hx[0, 0:6] = 1
    Hx[1, 3:9] = 1
    return Hx, Hz


def css422():
    """[[4, 2, 2]]: Hx = Hz = [1 1 1 1]."""
    return np.ones((1, 4), dtype=np.uint8), np.ones((1, 4), dtype=np.uint8)


def cube832():
    """[[8, 3, 2]] cube code (qubits on cube vertices = bit triples):
    Hx = X on all 8 vertices; Hz = 4 independent weight-4 faces.
    ASYMMETRIC distances: (dx, dz) = (4, 2)."""
    Hx = np.ones((1, 8), dtype=np.uint8)
    faces = [[v for v in range(8) if ((v >> axis) & 1) == 0] for axis in range(3)]
    faces.append([v for v in range(8) if ((v >> 2) & 1) == 1])
    Hz = np.zeros((4, 8), dtype=np.uint8)
    for i, f in enumerate(faces):
        Hz[i, f] = 1
    return Hx, Hz


def hgp(H1, H2):
    """Hypergraph product: Hx = [H1 x I | I x H2^T], Hz = [I x H2 | H1^T x I]."""
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    Hx = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]).astype(np.uint8)
    Hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]).astype(np.uint8)
    return Hx, Hz


def hgp32():
    """HGP(rep3, rep2): [[8, 1]] with ASYMMETRIC (dx, dz) = (2, 3)."""
    return hgp(rep_code(3), rep_code(2))


def five_qubit():
    """[[5, 1, 3]] perfect code (cyclic XZZXI), genuinely non-CSS."""
    s = "XZZXI"
    gens = [s[-i:] + s[:-i] for i in range(4)]
    HX = np.array([[1 if c == "X" else 0 for c in g] for g in gens], np.uint8)
    HZ = np.array([[1 if c == "Z" else 0 for c in g] for g in gens], np.uint8)
    return HX, HZ


def stab_pair(Hx, Hz):
    """Stack a CSS pair into stabilizer (HX, HZ) generator-part format."""
    HX = np.vstack([Hx, np.zeros_like(Hz)]).astype(np.uint8)
    HZ = np.vstack([np.zeros_like(Hx), Hz]).astype(np.uint8)
    return HX, HZ


class CyclicShim:
    """Minimal GroupData-like object for C_n (identity = 0). The LP code
    builders read only ``n``, ``mult``, ``inv``, ``identity``, so no GAP
    is needed."""

    def __init__(self, n: int):
        self.n = n
        self.mult = [[(i + j) % n for j in range(n)] for i in range(n)]
        self.inv = [(-i) % n for i in range(n)]
        self.identity = 0


def lp_code_c6():
    """LP code over C6, A = B = [[1, x + x^2]]: [[30, 6]] with exact
    (dx, dz) = (3, 3) (brute-forced; kernel dims are 18)."""
    from core.quantum_code import build_quantum_code
    r = build_quantum_code([[(0,), (1, 2)]], [[(0,), (1, 2)]], CyclicShim(6))
    return r


def random_hgp():
    """Seeded random HGP instance: [[16, 2]] with exact (dx, dz) = (2, 1)."""
    rng = np.random.default_rng(7)
    H1 = (rng.random((2, 4)) < 0.5).astype(np.uint8)
    H2 = (rng.random((2, 3)) < 0.5).astype(np.uint8)
    return hgp(H1, H2)


CSS_CODES = [
    # (id, builder, exact (dx, dz) — re-derived by the oracle inside each test)
    ("steane", steane, (3, 3)),
    ("shor", shor, (3, 3)),
    ("css422", css422, (2, 2)),
    ("cube832", cube832, (4, 2)),
    ("hgp32", hgp32, (2, 3)),
]


# ─────────────────────────────────────────────────────────────────
# Oracle self-tests (anchor the oracle itself to textbook values)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
class TestOracle:
    def test_classical_textbook_values(self):
        assert brute_classical_distance(rep_code(5)) == 5
        assert brute_classical_distance(HAMMING) == 3
        assert brute_classical_distance(ext_hamming()) == 4
        assert brute_classical_distance(np.eye(4, dtype=np.uint8)) is None
        # zero column => e_i is a codeword => d = 1
        assert brute_classical_distance(
            np.array([[1, 1, 0], [0, 1, 0]], np.uint8)) == 1
        # single parity row => even-weight code => d = 2
        assert brute_classical_distance(np.ones((1, 6), np.uint8)) == 2

    @pytest.mark.parametrize("name,builder,expect",
                             CSS_CODES, ids=[c[0] for c in CSS_CODES])
    def test_css_textbook_values(self, name, builder, expect):
        Hx, Hz = builder()
        # all zoo codes must really be CSS
        assert not ((Hx.astype(int) @ Hz.T.astype(int)) % 2).any()
        assert brute_css_distances(Hx, Hz) == expect

    def test_stab_five_qubit(self):
        assert brute_stab_distance(*five_qubit()) == 3

    def test_stab_css_stack_matches_css_min(self):
        # A CSS code viewed as a stabilizer code has d = min(dx, dz).
        for _, builder, (dx, dz) in CSS_CODES:
            HX, HZ = stab_pair(*builder())
            assert brute_stab_distance(HX, HZ) == min(dx, dz)


# ─────────────────────────────────────────────────────────────────
# Classical distance — BP+OSD backend
# ─────────────────────────────────────────────────────────────────


@pytest.mark.bposd
class TestClassicalBposdFresh:
    @pytest.mark.parametrize("H,expect", [
        (rep_code(5), 5),
        (ext_hamming(), 4),
        (np.array([[1, 1, 0], [0, 1, 0]], np.uint8), 1),   # zero column, d=1
        (np.ones((1, 6), np.uint8), 2),                    # single row, d=2
    ], ids=["rep5", "ext_hamming", "zero_col_d1", "single_row_d2"])
    def test_matches_brute_force(self, H, expect):
        from core.dist.classical import estimate_classical_distance_bposd
        assert brute_classical_distance(H) == expect   # oracle agrees
        d = estimate_classical_distance_bposd(
            H, num_trials=300, n_workers=2, osd_order=0)
        assert d == expect

    def test_random_codes_match_brute_force(self):
        """Seeded random parity checks; expected d derived by enumeration."""
        from core.dist.classical import estimate_classical_distance_bposd
        rng = np.random.default_rng(12345)
        for _ in range(4):
            H = (rng.random((5, 11)) < 0.35).astype(np.uint8)
            d_true = brute_classical_distance(H)
            assert d_true is not None
            d = estimate_classical_distance_bposd(
                H, num_trials=300, n_workers=2, osd_order=0)
            assert d == d_true

    def test_d_target_boundary_does_not_corrupt_result(self):
        """Strict-`<` semantics: d_target == true distance must neither stop
        the search before the weight-d codeword is reported nor change the
        returned value."""
        from core.dist.classical import estimate_classical_distance_bposd
        d = estimate_classical_distance_bposd(
            rep_code(5), num_trials=300, n_workers=2, d_target=5, osd_order=0)
        assert d == 5


# ─────────────────────────────────────────────────────────────────
# Quantum distances — BP+OSD backend
# ─────────────────────────────────────────────────────────────────


@pytest.mark.bposd
class TestQuantumBposdFresh:
    @pytest.mark.parametrize("name,builder,expect",
                             CSS_CODES, ids=[c[0] for c in CSS_CODES])
    def test_matches_brute_force_per_direction(self, name, builder, expect):
        """(dx, dz) each equal the exact per-direction brute-force value.
        cube832 (4,2) and hgp32 (2,3) are asymmetric, so a direction swap
        cannot pass; shor's Z side is degenerate (weight-2 stabilizers,
        dz=3), so counting stabilizers as logicals cannot pass."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = builder()
        assert brute_css_distances(Hx, Hz) == expect
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=400, n_workers=2, osd_order=5)
        assert (dx, dz) == expect

    def test_auto_basis_matches_brute_force(self):
        """Omitting Lx/Lz (auto-derived basis) must give the same exact
        distances — on the asymmetric cube code."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        Hx, Hz = cube832()
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, num_trials=400, n_workers=2, osd_order=5)
        assert (dx, dz) == (4, 2)

    def test_random_hgp_matches_brute_force(self):
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = random_hgp()
        expect = brute_css_distances(Hx, Hz)
        assert None not in expect
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=400, n_workers=2, osd_order=5)
        assert (dx, dz) == expect

    def test_low_budget_is_still_sound(self):
        """With a tiny trial budget the estimate may be missing (None) or
        heavy, but it must NEVER be lighter than the true distance — a
        lighter report would mean a stabilizer or invalid vector was
        counted as a logical."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        for _, builder, (dx_true, dz_true) in CSS_CODES:
            Hx, Hz = builder()
            Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
            dx, dz = estimate_quantum_distances_bposd(
                Hx, Hz, Lx, Lz, num_trials=5, n_workers=1, osd_order=0)
            assert dx is None or dx >= dx_true
            assert dz is None or dz >= dz_true

    def test_k0_auto_basis_returns_none_pair(self):
        """A k = 0 code with auto-derived logicals: both directions None."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        Hx = np.eye(4, dtype=np.uint8)
        Hz = np.zeros((0, 4), dtype=np.uint8)
        assert estimate_quantum_distances_bposd(
            Hx, Hz, num_trials=10, n_workers=1) == (None, None)

    def test_lp_c6_shim_matches_brute_force(self):
        """LP code over a hand-rolled C6 shim ([[30, 6]], exact (3, 3))."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from logical_basis.logical_basis import find_logical_basis
        r = lp_code_c6()
        Hx, Hz = r["Hx"], r["Hz"]
        assert brute_css_distances(Hx, Hz) == (3, 3)
        basis = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 6, (1, 2))
        dx, dz = estimate_quantum_distances_bposd(
            Hx, Hz, basis["Lx"], basis["Lz"],
            num_trials=400, n_workers=2, osd_order=5)
        assert (dx, dz) == (3, 3)

    def test_basis_independence_of_estimates(self):
        """Distances are basis-independent: the orbit-preserving, the RREF,
        and the pivot-aligned logical bases must all yield the same exact
        (dx, dz) on the same code."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from core.quantum_code import build_quantum_code
        from logical_basis.logical_basis import (
            find_logical_basis,
            find_logical_basis_pivot_aligned,
            find_logical_noncanonical_RREF,
        )
        r = build_quantum_code([[(0,), (1, 2)]], [[(0,), (1, 2)]], CyclicShim(3))
        Hx, Hz = r["Hx"], r["Hz"]
        expect = brute_css_distances(Hx, Hz)
        assert expect == (3, 3)
        b = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 3, (1, 2))
        bases = [
            (b["Lx"], b["Lz"]),
            find_logical_noncanonical_RREF(Hx, Hz),
            find_logical_basis_pivot_aligned(Hx, Hz),
        ]
        for Lx, Lz in bases:
            got = estimate_quantum_distances_bposd(
                Hx, Hz, Lx, Lz, num_trials=400, n_workers=2, osd_order=5)
            assert got == expect


# ─────────────────────────────────────────────────────────────────
# Classical distance — sqetch backend (GPU)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@requires_cuda
class TestClassicalSqetchFresh:
    @pytest.mark.parametrize("H,expect", [
        (rep_code(5), 5),
        (ext_hamming(), 4),
        (np.array([[1, 1, 0], [0, 1, 0]], np.uint8), 1),
        (np.ones((1, 6), np.uint8), 2),
    ], ids=["rep5", "ext_hamming", "zero_col_d1", "single_row_d2"])
    def test_matches_brute_force(self, H, expect):
        from core.dist.classical import estimate_classical_distance_sqetch
        assert brute_classical_distance(H) == expect
        d = estimate_classical_distance_sqetch(H, num_trials=300, seed=0)
        assert d == expect

    def test_full_rank_kernel_trivial_returns_none(self):
        from core.dist.classical import estimate_classical_distance_sqetch
        d = estimate_classical_distance_sqetch(
            np.eye(4, dtype=np.uint8), num_trials=100, seed=0)
        assert d is None

    def test_d_target_boundary_does_not_corrupt_result(self):
        from core.dist.classical import estimate_classical_distance_sqetch
        d = estimate_classical_distance_sqetch(
            rep_code(5), num_trials=300, d_target=5, seed=1)
        assert d == 5

    def test_return_codeword_contract(self):
        """The witnessing codeword must be a REAL minimum-weight codeword:
        in ker(H), weight == reported distance == brute-force distance."""
        from core.dist.classical import estimate_classical_distance_sqetch
        H = ext_hamming()
        d, cw = estimate_classical_distance_sqetch(
            H, num_trials=2000, seed=0, return_codeword=True)
        assert cw is not None
        assert cw.shape == (8,) and cw.dtype == np.uint8
        assert not ((H.astype(int) @ cw.astype(int)) % 2).any()
        assert int(cw.sum()) == d
        assert d == 4


# ─────────────────────────────────────────────────────────────────
# Quantum distances — sqetch backend (GPU)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@requires_cuda
class TestQuantumSqetchFresh:
    @pytest.mark.parametrize("name,builder,expect",
                             CSS_CODES, ids=[c[0] for c in CSS_CODES])
    def test_matches_brute_force_per_direction(self, name, builder, expect):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = builder()
        assert brute_css_distances(Hx, Hz) == expect
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=600, seed=0)
        assert (dx, dz) == expect

    def test_auto_basis_matches_brute_force(self):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        Hx, Hz = cube832()
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, num_trials=600, seed=0)
        assert (dx, dz) == (4, 2)

    def test_lp_c6_shim_matches_brute_force(self):
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        from logical_basis.logical_basis import find_logical_basis
        r = lp_code_c6()
        Hx, Hz = r["Hx"], r["Hz"]
        assert brute_css_distances(Hx, Hz) == (3, 3)
        basis = find_logical_basis(
            Hx, Hz, r["A_bin_canonical"], r["B_bin_canonical"], 6, (1, 2))
        dx, dz = estimate_quantum_distances_sqetch(
            Hx, Hz, basis["Lx"], basis["Lz"], num_trials=800, seed=3)
        assert (dx, dz) == (3, 3)

    def test_empty_direction_wiring_is_cross_type(self):
        """dx hinges on Lz (cross-type augmentation) and dz on Lx: emptying
        exactly one of them must kill exactly the matching direction while
        the other still converges to the exact value."""
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = hgp32()                       # exact (dx, dz) = (2, 3)
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        empty = np.zeros((0, 8), dtype=np.uint8)
        assert estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, empty, num_trials=400, seed=0) == (None, 3)
        assert estimate_quantum_distances_sqetch(
            Hx, Hz, empty, Lz, num_trials=400, seed=0) == (2, None)

    def test_return_logical_soundness_and_witness_validity(self):
        """Recovery-kernel path. The reported weights are upper bounds (the
        recovery kernel was observed to converge more slowly than the fast
        kernel and is not seed-deterministic, so exact equality is NOT
        asserted here); the witnesses must be genuine nontrivial logicals of
        exactly the reported weight, verified against the ORACLE, not
        against the basis the estimator was fed."""
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = hgp32()
        dx_true, dz_true = brute_css_distances(Hx, Hz)
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        dx, dz, vec_dx, vec_dz = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=5000, seed=0, return_logical=True)
        assert dx is not None and dz is not None
        assert dx >= dx_true and dz >= dz_true      # upper-bound soundness
        # dx witness: an X-logical = in ker(Hz), NOT in rowspan(Hx).
        assert vec_dx.shape == (8,) and vec_dx.dtype == np.uint8
        assert not ((Hz.astype(int) @ vec_dx.astype(int)) % 2).any()
        assert not _in_rowspan(Hx, vec_dx)
        assert int(vec_dx.sum()) == dx
        # dz witness: a Z-logical = in ker(Hx), NOT in rowspan(Hz).
        assert vec_dz.shape == (8,) and vec_dz.dtype == np.uint8
        assert not ((Hx.astype(int) @ vec_dz.astype(int)) % 2).any()
        assert not _in_rowspan(Hz, vec_dz)
        assert int(vec_dz.sum()) == dz

    def test_tailored_sampler_finds_exact_low_weight_set(self):
        """sample_low_weight_logicals_sqetch on the [8,4,4] extended Hamming
        code with L = I_8 (every nonzero codeword qualifies) and
        target_weight = 4 must return EXACTLY the set of weight-4 codewords
        (there are 14 of them, and no nonzero codeword is lighter)."""
        from core.dist.quantum_sqetch import sample_low_weight_logicals_sqetch
        H = ext_hamming()
        words = _all_span(_oracle_nullspace(H))
        expect = {tuple(w.tolist()) for w in words if 0 < w.sum() <= 4}
        assert len(expect) == 14
        got = sample_low_weight_logicals_sqetch(
            H, np.eye(8, dtype=np.uint8),
            num_trials=800, target_weight=4, seed=4)
        assert got.dtype == np.uint8 and got.shape[1] == 8
        # contracts on every returned row
        assert not ((H.astype(int) @ got.T.astype(int)) % 2).any()
        assert ((got.sum(axis=1) > 0) & (got.sum(axis=1) <= 4)).all()
        got_set = {tuple(r.tolist()) for r in got}
        assert len(got_set) == got.shape[0]          # dedup honored
        assert got_set == expect                     # completeness + exactness


@pytest.mark.bposd
@pytest.mark.gpu
@requires_cuda
class TestEstimatorAgreement:
    def test_bposd_and_sqetch_agree_with_brute_force(self):
        """Both backends on the same seeded random HGP code equal the exact
        brute-force answer (hence each other)."""
        from core.dist.quantum_bposd import estimate_quantum_distances_bposd
        from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Hx, Hz = random_hgp()
        expect = brute_css_distances(Hx, Hz)
        assert None not in expect
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
        got_bposd = estimate_quantum_distances_bposd(
            Hx, Hz, Lx, Lz, num_trials=400, n_workers=2, osd_order=5)
        got_sqetch = estimate_quantum_distances_sqetch(
            Hx, Hz, Lx, Lz, num_trials=600, seed=1)
        assert got_bposd == expect
        assert got_sqetch == expect


# ─────────────────────────────────────────────────────────────────
# Non-CSS stabilizer sampler (stab_symplectic) — pure numpy
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
class TestStabSymplecticFresh:
    def test_symplectic_weight_counts_y_once(self):
        from core.dist.stab_symplectic import symplectic_weight
        # qubit 1 carries both X and Z (a Y) and must count once
        assert symplectic_weight(np.array([1, 1, 0], np.uint8),
                                 np.array([0, 1, 1], np.uint8)) == 3
        assert symplectic_weight(np.zeros(4, np.uint8),
                                 np.zeros(4, np.uint8)) == 0

    def test_stab_k_new_codes_and_redundant_rows(self):
        from core.dist.stab_symplectic import stab_k
        HXs, HZs = stab_pair(*shor())
        assert stab_k(HXs, HZs) == 1
        HXc, HZc = stab_pair(*cube832())
        assert stab_k(HXc, HZc) == 3
        # duplicating a generator row must not change k
        assert stab_k(np.vstack([HXs, HXs[0:1]]),
                      np.vstack([HZs, HZs[0:1]])) == 1

    def test_centralizer_basis_contract_cube(self):
        from core.dist.stab_symplectic import centralizer_basis
        HX, HZ = stab_pair(*cube832())
        CX, CZ = centralizer_basis(HX, HZ)
        W, k = 8, 3
        assert CX.shape == (W + k, W) and CZ.shape == (W + k, W)
        # every basis element commutes with every generator
        comm = (HX.astype(int) @ CZ.T.astype(int)
                + HZ.astype(int) @ CX.T.astype(int)) % 2
        assert not comm.any()
        # the basis rows are independent
        assert _oracle_rank(np.hstack([CX, CZ])) == W + k

    @pytest.mark.parametrize("case", ["shor", "cube", "scrambled422"])
    def test_estimate_matches_brute_force_with_witness(self, case):
        """d_est equals the exact brute-force distance and the witness is a
        genuine logical: commutes with all generators, is NOT in the
        stabilizer span, and has symplectic weight == d_est.

        shor-as-stab is degenerate (weight-2 stabilizer generators, d = 3);
        scrambled422 is [[4,2,2]] conjugated by qubit-local H/S maps —
        a genuinely non-CSS presentation (weight-preserving, so d = 2)."""
        from core.dist.stab_symplectic import (estimate_stab_distance,
                                               symplectic_weight)
        if case == "shor":
            HX, HZ = stab_pair(*shor())
        elif case == "cube":
            HX, HZ = stab_pair(*cube832())
        else:
            HX = np.array([[0, 1, 1, 1], [1, 0, 1, 0]], np.uint8)
            HZ = np.array([[1, 1, 1, 0], [0, 1, 0, 1]], np.uint8)
        # generators must commute (sanity on the fixture itself)
        comm = (HX.astype(int) @ HZ.T.astype(int)
                + HZ.astype(int) @ HX.T.astype(int)) % 2
        assert not comm.any()
        d_true = brute_stab_distance(HX, HZ)
        r = estimate_stab_distance(HX, HZ, num_trials=300, seed=11)
        assert r["d_est"] == d_true
        wx, wz = r["witness_x"], r["witness_z"]
        assert not ((HX.astype(int) @ wz + HZ.astype(int) @ wx) % 2).any()
        stab_rows = np.hstack([HX, HZ])
        assert not _in_rowspan(stab_rows, np.concatenate([wx, wz]))
        assert symplectic_weight(wx, wz) == r["d_est"]

    def test_redundant_generator_rows_same_distance(self):
        from core.dist.stab_symplectic import estimate_stab_distance
        HX, HZ = stab_pair(*shor())
        r = estimate_stab_distance(np.vstack([HX, HX[0:1]]),
                                   np.vstack([HZ, HZ[0:1]]),
                                   num_trials=300, seed=11)
        assert r["d_est"] == 3 and r["k"] == 1

    def test_seed_determinism(self):
        """Pure numpy sampler: identical seeds give identical results,
        including the witness vectors."""
        from core.dist.stab_symplectic import estimate_stab_distance
        HX, HZ = stab_pair(*cube832())
        r1 = estimate_stab_distance(HX, HZ, num_trials=120, seed=99)
        r2 = estimate_stab_distance(HX, HZ, num_trials=120, seed=99)
        assert r1["d_est"] == r2["d_est"]
        assert r1["trials_run"] == r2["trials_run"]
        np.testing.assert_array_equal(r1["witness_x"], r2["witness_x"])
        np.testing.assert_array_equal(r1["witness_z"], r2["witness_z"])

    def test_early_stop_semantics_and_result_fields(self):
        from core.dist.stab_symplectic import estimate_stab_distance
        HX, HZ = stab_pair(*cube832())          # d = 2
        # d_target = 3: a weight-2 witness (< 3) stops the run early.
        r = estimate_stab_distance(HX, HZ, num_trials=500, seed=1, d_target=3)
        assert r["early_stopped"] is True
        assert r["trials_run"] < 500
        assert r["d_est"] == 2
        assert r["seed"] == 1 and r["num_trials"] == 500 and r["k"] == 3
        # d_target = 2: strict `<` — a weight-2 witness must NOT stop early.
        r = estimate_stab_distance(HX, HZ, num_trials=80, seed=1, d_target=2)
        assert r["early_stopped"] is False
        assert r["trials_run"] == 80
        assert r["d_est"] == 2

    def test_pair_passes_zero_still_converges(self):
        from core.dist.stab_symplectic import estimate_stab_distance
        HX, HZ = stab_pair(*cube832())
        r = estimate_stab_distance(HX, HZ, num_trials=300, seed=5,
                                   pair_passes=0)
        assert r["d_est"] == 2

    def test_shape_mismatch_raises(self):
        from core.dist.stab_symplectic import estimate_stab_distance
        HX, HZ = stab_pair(*cube832())
        with pytest.raises(ValueError, match="must match"):
            estimate_stab_distance(HX, HZ[:, :-1], num_trials=5, seed=0)

    @pytest.mark.parametrize("case", ["cube", "five_qubit"])
    def test_pure_sector_css_reduction_matches_brute_force(self, case):
        """pure_sector_css contract, checked by enumeration:
          * S_pure and reps are pure-sector centralizer elements
            (orthogonal to every generator's opposite part);
          * reps are independent modulo S_pure;
          * min weight over ker(opp) \\ rowspan(S_pure) — the pure-sector
            nontrivial-logical minimum — matches the promised CSS pair.
        cube832 has pure-X min 4 and pure-Z min 2; the five-qubit code has
        an EMPTY pure-sector stabilizer basis and pure min 5 in both sectors
        (only the transversal logicals are pure)."""
        from core.dist.stab_symplectic import pure_sector_css
        if case == "cube":
            HX, HZ = stab_pair(*cube832())
            expect = {"x": 4, "z": 2}
        else:
            HX, HZ = five_qubit()
            expect = {"x": 5, "z": 5}
        for sector in ("x", "z"):
            S_pure, opp, reps = pure_sector_css(HX, HZ, sector)
            opp_expected = HZ if sector == "x" else HX
            np.testing.assert_array_equal(opp, opp_expected)
            if S_pure.shape[0]:
                assert not ((opp.astype(int) @ S_pure.T.astype(int)) % 2).any()
            assert reps.shape[0] > 0
            assert not ((opp.astype(int) @ reps.T.astype(int)) % 2).any()
            assert (_oracle_rank(np.vstack([S_pure, reps]))
                    == _oracle_rank(S_pure) + reps.shape[0])
            # brute-force the pure-sector minimum
            best = None
            R, piv = _oracle_rref(S_pure) if S_pure.shape[0] else (
                np.zeros((0, HX.shape[1]), np.uint8), [])
            for v in _all_span(_oracle_nullspace(opp)):
                w = int(v.sum())
                if w == 0 or (best is not None and w >= best):
                    continue
                vv = v.copy()
                for ri, pc in enumerate(piv):
                    if vv[pc]:
                        vv ^= R[ri]
                if vv.any():
                    best = w
            assert best == expect[sector]

    def test_pure_sector_bad_sector_raises(self):
        from core.dist.stab_symplectic import pure_sector_css
        HX, HZ = five_qubit()
        with pytest.raises(ValueError, match="sector"):
            pure_sector_css(HX, HZ, "y")

    def test_coset_min_weight_matches_brute_force(self):
        """Five-qubit code, v = transversal X(11111):
        (a) coset over the stabilizer span alone;
        (b) 'masked' coset with the transversal Z logical added to G.
        Both minima are brute-forced (16 / 32 coset elements). Witness must
        lie in the coset (witness XOR v in rowspan(G))."""
        from core.dist.stab_symplectic import estimate_coset_min_weight
        HX, HZ = five_qubit()
        vx = np.ones(5, np.uint8)
        vz = np.zeros(5, np.uint8)
        for GX, GZ in [
            (HX, HZ),
            (np.vstack([HX, np.zeros((1, 5), np.uint8)]),
             np.vstack([HZ, np.ones((1, 5), np.uint8)])),
        ]:
            expect = brute_coset_min_weight(GX, GZ, vx, vz)
            assert expect == 3
            r = estimate_coset_min_weight(GX, GZ, vx, vz,
                                          num_trials=300, seed=8)
            assert r["w_est"] == expect
            wx, wz = r["witness_x"], r["witness_z"]
            assert int((wx | wz).sum()) == r["w_est"]
            delta = np.concatenate([wx ^ vx, wz ^ vz])
            assert _in_rowspan(np.hstack([GX, GZ]), delta)
            assert r["seed"] == 8 and r["num_trials"] == 300

    def test_coset_seed_determinism(self):
        from core.dist.stab_symplectic import estimate_coset_min_weight
        HX, HZ = five_qubit()
        vx = np.ones(5, np.uint8)
        vz = np.zeros(5, np.uint8)
        r1 = estimate_coset_min_weight(HX, HZ, vx, vz, num_trials=100, seed=8)
        r2 = estimate_coset_min_weight(HX, HZ, vx, vz, num_trials=100, seed=8)
        assert r1["w_est"] == r2["w_est"]
        np.testing.assert_array_equal(r1["witness_x"], r2["witness_x"])
        np.testing.assert_array_equal(r1["witness_z"], r2["witness_z"])

    def test_coset_early_stop_strict_semantics(self):
        from core.dist.stab_symplectic import estimate_coset_min_weight
        HX, HZ = five_qubit()
        vx = np.ones(5, np.uint8)
        vz = np.zeros(5, np.uint8)
        # coset min is 3: d_target=3 must never stop early (strict <)
        r = estimate_coset_min_weight(HX, HZ, vx, vz, num_trials=60, seed=2,
                                      d_target=3)
        assert r["early_stopped"] is False and r["trials_run"] == 60
        assert r["w_est"] == 3
        # d_target=4 stops as soon as the weight-3 element is seen
        r = estimate_coset_min_weight(HX, HZ, vx, vz, num_trials=500, seed=2,
                                      d_target=4)
        assert r["early_stopped"] is True and r["trials_run"] < 500
        assert r["w_est"] == 3

    def test_coset_v_in_span_raises_for_nontrivial_member(self):
        """The in-span guard must catch v in rowspan(G) even when v is a
        PRODUCT of generators, not a literal generator row."""
        from core.dist.stab_symplectic import estimate_coset_min_weight
        HX, HZ = five_qubit()
        with pytest.raises(ValueError, match="rowspan"):
            estimate_coset_min_weight(HX, HZ, HX[0] ^ HX[1], HZ[0] ^ HZ[1],
                                      num_trials=10, seed=0)
