"""LP code construction at the classical-code level.

Forward (ring → binary):
    build_A_bin(A, gd)   : (ma·n) × (na·n)   — uses left_rep
    build_B_bin(B, gd)   : (mb·n) × (nb·n)   — uses right_rep

Inverse (binary → ring):
    A_from_A_bin(A_bin, gd, shape_a)   — recover A from its binary lift
    B_from_B_bin(B_bin, gd, shape_b)   — recover B from its binary lift

Weight matrix (forgets group structure — just per-entry support sizes):
    weight_matrix(M)     : shape(rows, cols) int — W[i,j] = |M[i][j]|

Canonical block-col form (permute block-cols so an invertible block-col subset
is at the LAST positions — required by find_logical_basis / surgery):
    canonical_form_A(A, gd)                    — on ring matrix
    canonical_form_B(B, gd)
    canonical_form_A_bin(A_bin, gd, shape_a)   — chains via A_from_A_bin
    canonical_form_B_bin(B_bin, gd, shape_b)

Hx / Hz / compute_k / check_css / AB_from_Hx are NOT included here yet —
this module covers only the classical-code-level pieces.

Convention vs new/: shape parameters are `shape_a = (ma, na)` and
`shape_b = (mb, nb)` — both match the corresponding `build_*` input shape.
(new/ used `shape_bt = (nb, mb)` for the B side; we use `shape_b = (mb, nb)`.)
"""

import itertools

import numpy as np

from core.f2 import f2_rank
from core.group import GroupData, canonicalize, left_rep, right_rep


# ─────────────────────────────────────────────────────────────────
# Forward: ring → binary
# ─────────────────────────────────────────────────────────────────


def build_A_bin(A: list, gd: GroupData) -> np.ndarray:
    """A_bin: (ma·n) × (na·n) binary matrix using left-regular rep.

    Block [ia, ja] = L[A[ia][ja]]:
        A_bin[ia*n:(ia+1)*n, ja*n:(ja+1)*n] = left_rep(A[ia][ja], gd)
    """
    ma, na = len(A), len(A[0])
    n = gd.n
    M = np.zeros((ma * n, na * n), dtype=np.uint8)
    for ia in range(ma):
        for ja in range(na):
            M[ia * n:(ia + 1) * n, ja * n:(ja + 1) * n] = left_rep(A[ia][ja], gd)
    return M


def weight_matrix(M: list) -> np.ndarray:
    """Integer support-size matrix of a ring matrix.

    For a ring matrix ``M`` of shape ``(rows, cols)`` with each entry a
    ``tuple`` of distinct group-element indices (the ring element's
    support), returns an integer numpy array ``W`` of the same shape with
    ``W[i, j] = len(M[i][j])``.

    Forgets the actual ring elements — keeps only their weights. Useful
    for cheap classical filters that depend on weights only (e.g.
    ``abelian/weight_distance_bound``), search-config sparsity caps, and
    quantum check-weight derivation via ``core.quantum_code.quantum_check_weights``.

    Args:
        M: ring matrix (``list[list[tuple]]``). Works for both ``A`` and ``B``.

    Returns:
        ``(rows, cols)`` int numpy array.
    """
    rows = len(M)
    cols = len(M[0])
    W = np.empty((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            W[i, j] = len(M[i][j])
    return W


def build_B_bin(B: list, gd: GroupData) -> np.ndarray:
    """B_bin: (mb·n) × (nb·n) binary matrix using right-regular rep.

    Block [ib, jb] = R[B[ib][jb]]:
        B_bin[ib*n:(ib+1)*n, jb*n:(jb+1)*n] = right_rep(B[ib][jb], gd)
    """
    mb, nb = len(B), len(B[0])
    n = gd.n
    M = np.zeros((mb * n, nb * n), dtype=np.uint8)
    for ib in range(mb):
        for jb in range(nb):
            M[ib * n:(ib + 1) * n, jb * n:(jb + 1) * n] = right_rep(B[ib][jb], gd)
    return M


# ─────────────────────────────────────────────────────────────────
# Inverse: binary → ring
# ─────────────────────────────────────────────────────────────────


def A_from_A_bin(A_bin: np.ndarray, gd: GroupData, shape_a: tuple) -> list:
    """Recover the ring matrix A from A_bin = build_A_bin(A, gd).

    Under the package convention `gd.identity == 0`, **column 0** of each
    n×n block has 1s at exactly the support of the underlying ring element:
        L[x]_{k, 0} = 1  iff  (some g ∈ x) g·g_0 = k.
    With g_0 = identity (enforced by GroupData), this gives k = g, so col 0
    reads off x directly.

    Always rebuilds A_bin from the recovered A and asserts equality — inputs
    that are not a valid L-lift will raise. There is no opt-out: callers that
    want to skip the check can call `build_A_bin` themselves and compare.

    Args:
        A_bin: (ma·n) × (na·n) binary matrix.
        gd: GroupData.
        shape_a: (ma, na).

    Returns:
        ma × na list of lists of canonical sorted tuples (ring elements).

    Raises:
        ValueError if A_bin shape is inconsistent with shape_a, or if the
        rebuilt A_bin differs from the input.
    """
    ma, na = shape_a
    n = gd.n
    if A_bin.shape != (ma * n, na * n):
        raise ValueError(f"A_bin shape {A_bin.shape} != ({ma*n}, {na*n})")
    A = []
    for ia in range(ma):
        row = []
        for ja in range(na):
            block = A_bin[ia * n:(ia + 1) * n, ja * n:(ja + 1) * n]
            entry = tuple(int(k) for k in np.where(block[:, 0])[0])
            row.append(entry)
        A.append(row)
    if not np.array_equal(A_bin, build_A_bin(A, gd)):
        raise ValueError(
            "A_bin is not a valid left-rep lift of any ring matrix "
            "(rebuild from recovered A differs from input)."
        )
    return A


def B_from_B_bin(B_bin: np.ndarray, gd: GroupData, shape_b: tuple) -> list:
    """Recover the ring matrix B from B_bin = build_B_bin(B, gd).

    Under the package convention `gd.identity == 0`, **row 0** of each n×n
    block has 1s at exactly the support of the underlying ring element:
        R[x]_{0, h} = 1  iff  (some g ∈ x) h·g⁻¹ = g_0.
    With g_0 = identity (enforced by GroupData), this gives h = g, so row 0
    reads off x directly.

    Always rebuilds B_bin from the recovered B and asserts equality — inputs
    that are not a valid R-lift will raise. There is no opt-out: callers that
    want to skip the check can call `build_B_bin` themselves and compare.

    Args:
        B_bin: (mb·n) × (nb·n) binary matrix.
        gd: GroupData.
        shape_b: (mb, nb).

    Returns:
        mb × nb list of lists of canonical sorted tuples.

    Raises:
        ValueError if B_bin shape is inconsistent with shape_b, or if the
        rebuilt B_bin differs from the input.
    """
    mb, nb = shape_b
    n = gd.n
    if B_bin.shape != (mb * n, nb * n):
        raise ValueError(f"B_bin shape {B_bin.shape} != ({mb*n}, {nb*n})")
    B = []
    for ib in range(mb):
        row = []
        for jb in range(nb):
            block = B_bin[ib * n:(ib + 1) * n, jb * n:(jb + 1) * n]
            entry = tuple(int(h) for h in np.where(block[0, :])[0])
            row.append(entry)
        B.append(row)
    if not np.array_equal(B_bin, build_B_bin(B, gd)):
        raise ValueError(
            "B_bin is not a valid right-rep lift of any ring matrix "
            "(rebuild from recovered B differs from input)."
        )
    return B


# ─────────────────────────────────────────────────────────────────
# Canonical block-col form
# ─────────────────────────────────────────────────────────────────


def _canonicalize_entries(M: list) -> list:
    """Return a copy of M with each entry passed through canonicalize."""
    return [[canonicalize(M[i][j]) for j in range(len(M[0]))]
            for i in range(len(M))]


def _find_invertible_block_col_subset(M_bin: np.ndarray, n: int, m_rows: int,
                                       n_cols: int):
    """Find a size-``m_rows`` subset of the ``n_cols`` block-cols of ``M_bin``
    whose binary submatrix is full rank (= ``m_rows·n``).

    Idempotency rule: the trailing block-col subset ``(n_cols - m_rows, …,
    n_cols - 1)`` is tested FIRST. If that's already invertible, return it
    so callers using the result as a permutation get the identity. Only if
    the trailing subset is rank-deficient do we fall back to the lex-first
    invertible subset (`combinations(range(n_cols), m_rows)`).

    Returns the subset as a sorted tuple, or ``None`` if no such subset exists.
    """
    target_rank = m_rows * n

    def _rank_of(subset):
        col_idx = np.fromiter(
            (j * n + k for j in subset for k in range(n)),
            dtype=np.int64,
            count=m_rows * n,
        )
        return f2_rank(M_bin[:, col_idx])

    # 1. Trailing positions first — gives idempotent canonical_form_*.
    trailing = tuple(range(n_cols - m_rows, n_cols))
    if _rank_of(trailing) == target_rank:
        return trailing

    # 2. Otherwise the lex-first invertible subset.
    for subset in itertools.combinations(range(n_cols), m_rows):
        if subset == trailing:
            continue  # already checked
        if _rank_of(subset) == target_rank:
            return tuple(subset)
    return None


def canonical_form_A(A: list, gd: GroupData):
    """Canonical block-col form of A.

    Permutes the na block-cols of A so that an invertible block-col subset of
    A_bin lands at the LAST ma positions `[na-ma, …, na-1]`. The trailing
    subset is preferred when it is already invertible (keeping the form
    idempotent); otherwise the lex-first invertible subset (over
    `combinations(range(na), ma)`) is chosen. Entries are canonicalized via
    `core.group.canonicalize` regardless.

    Args:
        A: ma × na ring matrix.
        gd: GroupData.

    Returns:
        4-tuple (A_canonical, A_bin_canonical, perm, has_full_rank):
          A_canonical: ma × na ring matrix with canonical entries; if
                       has_full_rank is True, additionally block-col-permuted.
          A_bin_canonical: (ma·n) × (na·n) = build_A_bin(A_canonical, gd).
          perm: length-na list. A_canonical[i][k] == A_canon_entries[i][perm[k]],
                where A_canon_entries is A with each entry canonicalized.
          has_full_rank: True iff an invertible block-col subset was found
                         and moved. If False, perm == list(range(na)).
    """
    ma = len(A)
    na = len(A[0])
    n = gd.n

    A_canon_entries = _canonicalize_entries(A)
    A_bin = build_A_bin(A_canon_entries, gd)

    subset = _find_invertible_block_col_subset(A_bin, n, ma, na)
    if subset is None:
        return A_canon_entries, A_bin, list(range(na)), False

    others = [j for j in range(na) if j not in subset]
    perm = others + list(subset)
    A_canonical = [[A_canon_entries[i][perm[k]] for k in range(na)]
                   for i in range(ma)]
    A_bin_canonical = build_A_bin(A_canonical, gd)
    return A_canonical, A_bin_canonical, perm, True


def canonical_form_B(B: list, gd: GroupData):
    """Canonical block-col form of B.

    Permutes the nb block-cols of B so that an invertible block-col subset of
    B_bin lands at the LAST mb positions `[nb-mb, …, nb-1]`.

    Args:
        B: mb × nb ring matrix.
        gd: GroupData.

    Returns:
        4-tuple (B_canonical, B_bin_canonical, perm, has_full_rank), same
        semantics as canonical_form_A.
    """
    mb = len(B)
    nb = len(B[0])
    n = gd.n

    B_canon_entries = _canonicalize_entries(B)
    B_bin = build_B_bin(B_canon_entries, gd)

    subset = _find_invertible_block_col_subset(B_bin, n, mb, nb)
    if subset is None:
        return B_canon_entries, B_bin, list(range(nb)), False

    others = [j for j in range(nb) if j not in subset]
    perm = others + list(subset)
    B_canonical = [[B_canon_entries[i][perm[k]] for k in range(nb)]
                   for i in range(mb)]
    B_bin_canonical = build_B_bin(B_canonical, gd)
    return B_canonical, B_bin_canonical, perm, True


def canonical_form_A_bin(A_bin: np.ndarray, gd: GroupData, shape_a: tuple):
    """Canonical form starting from binary A_bin.

    Chain: A = A_from_A_bin(A_bin); then canonical_form_A(A, gd).
    Useful when only A_bin is available (e.g. loaded from a saved code).

    Returns the same 4-tuple as canonical_form_A.
    """
    A = A_from_A_bin(A_bin, gd, shape_a)
    return canonical_form_A(A, gd)


def canonical_form_B_bin(B_bin: np.ndarray, gd: GroupData, shape_b: tuple):
    """Canonical form starting from binary B_bin.

    Chain: B = B_from_B_bin(B_bin); then canonical_form_B(B, gd).

    Returns the same 4-tuple as canonical_form_B.
    """
    B = B_from_B_bin(B_bin, gd, shape_b)
    return canonical_form_B(B, gd)
