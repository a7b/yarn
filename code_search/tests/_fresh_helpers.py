"""Independent oracle helpers for the ``test_canonical_fresh*`` suite.

Everything in this module is deliberately re-derived from the *documented*
contracts (search/canonical/README.md and module docstrings) using only
numpy + stdlib — it never imports ``core``/``search``/``logical_basis``.
That way the fresh tests check the implementation against an independent
reimplementation, not against itself.

Conventions re-implemented here (from the README):
    L[g]·e_h = e_{g·h}          (left-regular rep)
    R[g]·e_h = e_{h·g⁻¹}        (right-regular rep)
A ring element is a set of distinct 0-based group-element indices.
"""

import itertools

import numpy as np


# ─────────────────────────────────────────────────────────────────
# GF(2) linear algebra (independent implementation)
# ─────────────────────────────────────────────────────────────────


def rref2(M):
    """(R, pivot_cols) — reduced row echelon form over GF(2)."""
    R = (np.array(M, dtype=np.uint8, copy=True) & 1)
    if R.ndim == 1:
        R = R[None, :]
    rows, cols = R.shape
    piv = []
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if R[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        R[[r, pivot]] = R[[pivot, r]]
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        piv.append(c)
        r += 1
    return R, piv


def rank2(M):
    """GF(2) rank."""
    return len(rref2(M)[1])


def nullspace2(M):
    """Basis (k × n uint8) of the right null space of ``M`` over GF(2)."""
    M = np.atleast_2d(np.asarray(M))
    R, piv = rref2(M)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in piv]
    B = np.zeros((len(free), cols), np.uint8)
    for i, fc in enumerate(free):
        B[i, fc] = 1
        for r_, pc in enumerate(piv):
            if R[r_, fc]:
                B[i, pc] = 1
    return B


def span_all(rows, max_dim=20):
    """Every GF(2) combination of ``rows`` as a (2^k, n) uint8 array."""
    B = np.atleast_2d(np.asarray(rows, np.uint8))
    # Reduce to an independent basis first so 2^k is the true span size.
    R, piv = rref2(B)
    basis = R[: len(piv)]
    if len(piv) > max_dim:
        raise ValueError(f"span dimension {len(piv)} > guard {max_dim}")
    out = np.zeros((1, B.shape[1]), np.uint8)
    for row in basis:
        out = np.vstack([out, out ^ row])
    return out


def pack_rows(V):
    """Pack binary rows (n ≤ 64 columns) into uint64 keys."""
    V = np.atleast_2d(np.asarray(V, np.uint64))
    n = V.shape[1]
    if n > 64:
        raise ValueError("pack_rows supports n <= 64")
    w = np.uint64(1) << np.arange(n, dtype=np.uint64)
    return (V * w).sum(axis=1)


def true_classical_distance(H):
    """Exact d(ker H) by enumerating the whole kernel. None if ker = {0}."""
    K = span_all(nullspace2(H))
    w = K.sum(axis=1)
    nz = w[w > 0]
    return int(nz.min()) if nz.size else None


def true_quantum_distance_one_direction(H_stab_same, H_check_other):
    """Exact min weight of ``v`` with ``H_check_other · v = 0`` and
    ``v ∉ rowspan(H_stab_same)``.

    For dx pass (H_stab_same=Hx, H_check_other=Hz); for dz swap the roles.
    Returns None if every kernel vector is a stabilizer (k = 0).
    """
    K = span_all(nullspace2(H_check_other))
    Rst, piv = rref2(H_stab_same)
    stab = span_all(Rst[: len(piv)])
    stab_keys = set(pack_rows(stab).tolist())
    keys = pack_rows(K)
    w = K.sum(axis=1)
    best = None
    for i in range(K.shape[0]):
        if int(keys[i]) not in stab_keys:
            wi = int(w[i])
            best = wi if best is None else min(best, wi)
    return best


# ─────────────────────────────────────────────────────────────────
# GAP-free groups (Cayley-table shims) + regular representations
# ─────────────────────────────────────────────────────────────────


class GroupShimFresh:
    """Minimal duck-typed GroupData stand-in: Cayley table only.

    Exposes exactly the attribute surface the canonical funnel documents as
    sufficient: ``n``, ``mult``, ``inv``, ``identity``, ``is_abelian``.
    """

    def __init__(self, mult, identity=0):
        mult = np.asarray(mult, dtype=np.int64)
        self.mult = [list(int(v) for v in row) for row in mult]
        self.n = len(self.mult)
        self.identity = int(identity)
        inv = [None] * self.n
        for g in range(self.n):
            for h in range(self.n):
                if self.mult[g][h] == self.identity:
                    inv[g] = h
        self.inv = inv
        self.is_abelian = all(
            self.mult[a][b] == self.mult[b][a]
            for a in range(self.n) for b in range(self.n)
        )


def dihedral_shim(m):
    """Dihedral group of order 2m built from first principles.

    Element (k, f) = rotation^k · flip^f is stored at index ``k + m·f``;
    (k1,f1)·(k2,f2) = (k1 + (-1)^{f1} k2 mod m, f1 xor f2). Identity = 0.
    """
    n = 2 * m
    mult = np.zeros((n, n), np.int64)
    for k1 in range(m):
        for f1 in range(2):
            for k2 in range(m):
                for f2 in range(2):
                    k = (k1 + (k2 if f1 == 0 else -k2)) % m
                    mult[k1 + m * f1, k2 + m * f2] = k + m * (f1 ^ f2)
    return GroupShimFresh(mult)


def check_group_axioms(gd):
    """Raise AssertionError unless ``gd``'s Cayley table is a real group
    with identity at index 0 (closure, identity, inverses, associativity)."""
    n = gd.n
    assert gd.identity == 0
    for a in range(n):
        assert gd.mult[0][a] == a and gd.mult[a][0] == a
        assert gd.mult[a][gd.inv[a]] == 0 and gd.mult[gd.inv[a]][a] == 0
        for b in range(n):
            assert 0 <= gd.mult[a][b] < n
    for a in range(n):
        for b in range(n):
            for c in range(n):
                assert gd.mult[gd.mult[a][b]][c] == gd.mult[a][gd.mult[b][c]]


def my_left(x, gd):
    """Independent L[x]: column h has a 1 at row g·h for each g in x (mod 2)."""
    n = gd.n
    M = np.zeros((n, n), np.uint8)
    for g in x:
        for h in range(n):
            M[gd.mult[g][h], h] ^= 1
    return M


def my_right(x, gd):
    """Independent R[x]: column h has a 1 at row h·g⁻¹ for each g in x (mod 2)."""
    n = gd.n
    M = np.zeros((n, n), np.uint8)
    for g in x:
        gi = gd.inv[g]
        for h in range(n):
            M[gd.mult[h][gi], h] ^= 1
    return M


def my_A_bin(entries, gd):
    """Independent [L[x_0] | L[x_1] | ...] lift for a 1-row ring matrix."""
    return np.hstack([my_left(x, gd) for x in entries])


def my_B_bin(entries, gd):
    """Independent [R[x_0] | R[x_1] | ...] lift for a 1-row ring matrix."""
    return np.hstack([my_right(x, gd) for x in entries])


def identity_unit_supports(gd, weight):
    """All identity-containing weight-``weight`` supports whose LEFT lift is
    invertible — an independent re-derivation of the brute anchor pool."""
    out = []
    for rest in itertools.combinations(range(1, gd.n), weight - 1):
        x = (0,) + rest
        if rank2(my_left(x, gd)) == gd.n:
            out.append(x)
    return out


def decode_A_side(M_bin, n):
    """Read the ring entries back off an A-side (left-rep) lift.

    Column 0 of each n×n block is the support (identity at index 0)."""
    n_blocks = M_bin.shape[1] // n
    return tuple(
        tuple(int(r) for r in np.where(M_bin[:, j * n])[0])
        for j in range(n_blocks)
    )


def decode_B_side(M_bin, n):
    """Read the ring entries back off a B-side (right-rep) lift.

    Row 0 of each n×n block is the support (identity at index 0)."""
    n_blocks = M_bin.shape[1] // n
    return tuple(
        tuple(int(c) for c in np.where(M_bin[0, j * n:(j + 1) * n])[0])
        for j in range(n_blocks)
    )
