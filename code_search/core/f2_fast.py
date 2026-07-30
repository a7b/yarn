"""Fast bit-packed GF(2) linear algebra for the search hot path.

The canonical/screen pipeline computes a logical basis *per candidate pair*,
which is the dominant cost (~448 ms/pair with the naive ``core.f2`` Python
row-loop). These routines pack rows into ``uint64`` words and vectorize the
elimination, giving ~30-60x.

Scope: used ONLY by the screen (``search.canonical``). ``core.f2`` and the
canonical ``find_logical_basis`` are untouched — the saved deliverable's basis
still goes through the audited path. The screen only needs *a* valid logical
basis (distance is basis-independent), so a fast non-canonical basis is fine.

All functions take/return unpacked ``uint8`` {0,1} matrices.
"""

import numpy as np

_U64 = np.uint64


def _pack(M: np.ndarray):
    """(m,n) binary → (m, W) uint64 (little-endian bit j of col index)."""
    M = (np.asarray(M) != 0).astype(_U64)
    m, n = M.shape
    W = (n + 63) // 64
    pad = W * 64 - n
    if pad:
        M = np.hstack([M, np.zeros((m, pad), dtype=_U64)])
    M = M.reshape(m, W, 64)
    weights = (_U64(1) << np.arange(64, dtype=_U64))
    return (M * weights).sum(axis=2).astype(_U64), n


def _unpack(P: np.ndarray, n: int) -> np.ndarray:
    """(m, W) uint64 → (m, n) uint8."""
    m, W = P.shape
    bits = ((P[:, :, None] >> np.arange(64, dtype=_U64)) & _U64(1)).astype(np.uint8)
    return bits.reshape(m, W * 64)[:, :n].copy()


def _rref_packed(P: np.ndarray, n: int):
    """In-place reduced RREF on packed rows. Returns (rref_rows, pivot_cols)."""
    P = P.copy()
    m = P.shape[0]
    rank = 0
    pivots = []
    for col in range(n):
        w = col >> 6
        mask = _U64(1) << _U64(col & 63)
        sub = P[rank:, w] & mask
        nz = np.nonzero(sub)[0]
        if nz.size == 0:
            continue
        piv = rank + int(nz[0])
        if piv != rank:
            P[[rank, piv]] = P[[piv, rank]]
        has = (P[:, w] & mask) != 0
        has[rank] = False
        if has.any():
            P[has] ^= P[rank]          # vectorized GF(2) elimination (above+below)
        pivots.append(col)
        rank += 1
        if rank == m:
            break
    return P[:rank].copy(), pivots


def f2_rank_fast(M: np.ndarray) -> int:
    P, n = _pack(M)
    _, piv = _rref_packed(P, n)
    return len(piv)


def f2_nullspace_fast(M: np.ndarray) -> np.ndarray:
    """Right null space basis of ``M`` over GF(2), shape ``(n - rank, n)``."""
    P, n = _pack(M)
    R, piv = _rref_packed(P, n)
    Run = _unpack(R, n)
    pivset = set(piv)
    free = [c for c in range(n) if c not in pivset]
    null = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        null[i, f] = 1
        for r, pc in enumerate(piv):
            null[i, pc] = Run[r, f]
    return null


def _complement_in_kernel(A: np.ndarray, H_check: np.ndarray, n: int) -> np.ndarray:
    """Rows that extend rowspan(A) to a basis of ker(H_check).

    Requires ``rowspan(A) ⊆ ker(H_check)`` (true for CSS: ``A=Hx, H_check=Hz``).
    Returns a ``(k, n)`` complement basis, ``k = (n − rank H_check) − rank A``.
    """
    NK = f2_nullspace_fast(H_check)              # basis of ker(H_check)
    if NK.shape[0] == 0:
        return np.zeros((0, n), dtype=np.uint8)
    PA, _ = _pack(A)
    RA, pivA = _rref_packed(PA, n)               # rowspan(A) in RREF
    NKp, _ = _pack(NK)
    if RA.shape[0]:
        for r, pc in enumerate(pivA):            # reduce NK by rowspan(A)
            w = pc >> 6
            mask = _U64(1) << _U64(pc & 63)
            has = (NKp[:, w] & mask) != 0
            if has.any():
                NKp[has] ^= RA[r]
    RC, _ = _rref_packed(NKp, n)                 # independent remainders
    return _unpack(RC, n)


def screen_basis(Hx: np.ndarray, Hz: np.ndarray):
    """Fast (Lx, Lz, k) for the distance screen.

    ``Lx`` spans ``ker(Hz) / rowspan(Hx)`` (X logicals), ``Lz`` spans
    ``ker(Hx) / rowspan(Hz)`` (Z logicals). Each is ``(k, n_phys)`` uint8 with
    ``k = n_phys − rank(Hx) − rank(Hz)``. NOT symplectically paired
    (``Lx·Lzᵀ`` need not be ``I``) — irrelevant for distance estimation, which
    only needs a valid logical basis per direction.
    """
    Hx = np.asarray(Hx, dtype=np.uint8)
    Hz = np.asarray(Hz, dtype=np.uint8)
    n = Hx.shape[1]
    Lx = _complement_in_kernel(Hx, Hz, n)        # ker(Hz) \ rowspan(Hx)
    Lz = _complement_in_kernel(Hz, Hx, n)        # ker(Hx) \ rowspan(Hz)
    return Lx, Lz, Lx.shape[0]
