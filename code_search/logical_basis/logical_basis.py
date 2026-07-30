"""Paired X / Z logical basis for LP qLDPC codes (G-orbit-preserving).

Public API:
    find_logical_basis(Hx, Hz, A_bin, B_bin, n, shape_a, shape_b=None,
                       save_dir=None) -> dict

Given a CSS LP code (Hx, Hz) plus its classical binary lifts (A_bin, B_bin),
this constructs k paired X/Z logical operators such that
``Lx · Lzᵀ = I_k`` (mod 2), preserving the G-orbit structure of the code.

Algorithm (primary G-orbit-preserving path only — no fallback):
  1. Find an invertible ``ma·n × ma·n`` block-col subset of ``A_bin``
     (``any_block_col_full_rank`` ⟹ ``ma`` block-cols form a square invertible
     binary matrix). Same on ``B_bin``.
  2. Build ``ker(A_bin)``: for each free block, place a unit vector and solve
     the full-rank block. Group the n resulting codewords by G-orbit.
  3. Embed each G-orbit into ``ker(Hx)``'s left section over the ``nb`` fiber
     positions. Symmetrically for X-logicals via ker(B_bin) into ker(Hz).
  4. Greedily select complete G-orbit groups until ``k`` logicals are found.
  5. Pair WITHOUT breaking orbits (primary, ``_orbit_identity_pairing``): among the
     candidate orbits pick the pivot-matched (X-orbit, Z-orbit) pair whose
     ``P = Lx · Lzᵀ`` is a PERMUTATION matrix, then reorder the X rows so
     ``Lx · Lzᵀ = I_k``. A row-reorder keeps each of Lx, Lz a single G-orbit, so the
     operator weight stays UNIFORM per direction. Such a pair exists for every
     single-orbit (``k == |G|``) code tested here.
  5'. Fallback (``k > n`` / n_groups > 1, or no permutation pair): ``Lx ← P⁻¹ · Lx``.
     When ``P`` is a dense G-circulant this MIXES the X rows, so the returned ``Lx``
     is then NOT orbit-preserving (its row weights spread); ``Lz`` and the pre-pairing
     X_logicals stay orbit-pure. Reordering cannot avoid this for dense ``P``.

So the RETURNED basis is G-orbit-preserving (uniform weight) in BOTH directions
whenever the primary path applies (all single-orbit codes here); only the fallback
can yield a non-uniform ``Lx``. Distances are unaffected either way (basis-independent).

This function **raises** ``ValueError`` if the code does not have a
structured canonical basis (no full-rank ``ma``-block-col subset on either
``A_bin`` or ``B_bin``, or ``k % n != 0``). There is no fallback to a
non-orbit-preserving basis. Callers should pre-check via the
``has_full_rank_a/b`` flags from ``core.classical_code.canonical_form_A/B``.

Conventions:
    shape_a = (ma, na), shape_b = (mb, nb) — both match the corresponding
    ``build_*`` input shape (package convention; the legacy transposed
    ``shape_bt = (nb, mb)`` is NOT used here).
    shape_b defaults to (ma, na) when None.
    Returned Lx, Lz are flat: shape (k, n_phys). G-orbit structure is
    surfaced via the ``Z_group_tags`` / ``X_group_tags`` metadata.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from core.f2 import f2_rank, f2_reduce, f2_rref, f2_solve


# ---------------------------------------------------------------------------
# Block-column search
# ---------------------------------------------------------------------------


def _find_full_rank_block_cols(bin_matrix: np.ndarray, n: int, n_blocks_needed: int) -> list[int]:
    """Find ``n_blocks_needed`` block-cols of ``bin_matrix`` (each n cols wide) whose
    combined ``n_blocks_needed·n × n_blocks_needed·n`` submatrix is invertible over GF(2).

    Returns the sorted list of block-column indices.

    Raises:
        ValueError if no such subset exists (the code has no structured
        canonical basis on this side).
    """
    _nrows, total_cols = bin_matrix.shape
    n_blocks = total_cols // n
    for cols in itertools.combinations(range(n_blocks), n_blocks_needed):
        col_indices: list[int] = []
        for c in cols:
            col_indices.extend(range(c * n, (c + 1) * n))
        sub = bin_matrix[:, col_indices]
        if sub.shape[0] == sub.shape[1] and f2_rank(sub) == sub.shape[0]:
            return list(cols)
    raise ValueError(
        f"No full-rank {n_blocks_needed}-block-column subset found in matrix of "
        f"shape {bin_matrix.shape}; code has no structured canonical basis."
    )


# ---------------------------------------------------------------------------
# Codeword construction
# ---------------------------------------------------------------------------


def _build_codewords(bin_matrix: np.ndarray, n: int,
                     full_rank_block_cols: list[int]) -> np.ndarray:
    """Construct G-orbit-organized null-space codewords of ``bin_matrix``.

    For each "free" (non-full-rank) block-col k and each of the n unit-vector
    placements inside that block, solve the full-rank block to satisfy
    ``bin_matrix · v = 0``.

    Returns:
        Array of shape ``(n_free, n, total_cols)``:
            axis 0 = free block index,
            axis 1 = group-element index within the orbit,
            axis 2 = the codeword.
    """
    _nrows, total_cols = bin_matrix.shape
    n_total = total_cols // n
    free_cols = [c for c in range(n_total) if c not in full_rank_block_cols]
    n_free = len(free_cols)

    full_rank_bin_cols: list[int] = []
    for c in full_rank_block_cols:
        full_rank_bin_cols.extend(range(c * n, (c + 1) * n))
    free_bin_cols: list[int] = []
    for c in free_cols:
        free_bin_cols.extend(range(c * n, (c + 1) * n))

    A_right = bin_matrix[:, full_rank_bin_cols]    # (n_blocks_needed*n, n_blocks_needed*n) invertible
    A_left = bin_matrix[:, free_bin_cols]    # (n_blocks_needed*n, n_free*n)

    X = f2_solve(A_right, A_left)            # A_right @ X = A_left

    codewords = np.zeros((n_free, n, total_cols), dtype=np.uint8)
    for k in range(n_free):
        for g in range(n):
            i = k * n + g
            codewords[k, g, free_bin_cols[i]] = 1
            for fr_idx, fr_bin in enumerate(full_rank_bin_cols):
                codewords[k, g, fr_bin] = X[fr_idx, i]
    return codewords


# ---------------------------------------------------------------------------
# Embedding into ker(Hx) / ker(Hz) left sections
# ---------------------------------------------------------------------------


def _embed_z_logicals(codewords: np.ndarray, n: int, na: int, nb: int,
                      n_phys: int) -> tuple[list[np.ndarray], list[tuple]]:
    """Embed Z codewords from ker(A_bin) (length na·n) into ker(Hx) left
    section. Left section: na·nb col-blocks of size n, block index = ja·nb+ib.

    Returns:
        (candidate_groups, group_tags) — one (n × n_phys) group per
        (free_block, ib) pair, plus matching list of (k, ib) tags.
    """
    n_free = codewords.shape[0]
    candidate_groups: list[np.ndarray] = []
    group_tags: list[tuple] = []
    for k in range(n_free):
        for ib in range(nb):
            group = np.zeros((n, n_phys), dtype=np.uint8)
            for g in range(n):
                v = codewords[k, g]   # length na*n
                for ja in range(na):
                    block = ja * nb + ib
                    group[g, block * n: (block + 1) * n] = \
                        v[ja * n: (ja + 1) * n]
            candidate_groups.append(group)
            group_tags.append((k, ib))
    return candidate_groups, group_tags


def _embed_x_logicals(codewords: np.ndarray, n: int, na: int, nb: int,
                      n_phys: int) -> tuple[list[np.ndarray], list[tuple]]:
    """Embed X codewords from ker(B_bin) (length nb·n) into ker(Hz) left
    section. Left section: na·nb col-blocks, block index = ja·nb+jb.

    For each row ``ja ∈ [na]`` of the left section and each free block of
    ``B_bin``, fill the ``nb`` blocks at positions ``(ja, 0)..(ja, nb-1)``
    with the codeword (length ``nb·n``) and leave the right section zero.
    The row-``ja`` slice of Hz checks then becomes ``B · u^T = 0``, which
    is satisfied by construction.

    Returns:
        (candidate_groups, group_tags) — one (n × n_phys) group per
        (free_block, ja) pair, plus matching list of (k, ja) tags.
    """
    n_free = codewords.shape[0]
    candidate_groups: list[np.ndarray] = []
    group_tags: list[tuple] = []
    for k in range(n_free):
        for ja in range(na):
            group = np.zeros((n, n_phys), dtype=np.uint8)
            for g in range(n):
                u = codewords[k, g]   # length nb*n
                for jb in range(nb):
                    block = ja * nb + jb
                    group[g, block * n: (block + 1) * n] = \
                        u[jb * n: (jb + 1) * n]
            candidate_groups.append(group)
            group_tags.append((k, ja))
    return candidate_groups, group_tags


# ---------------------------------------------------------------------------
# Independence selection (G-orbit groups)
# ---------------------------------------------------------------------------


def _select_independent_groups(
    candidate_groups: list[np.ndarray],
    group_tags: list[tuple],
    seed_rows: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[tuple]]:
    """Greedily select complete length-n batches until ``k`` total logicals.

    Each "group" is an ``(n × n_phys)`` batch (the n vectors of one G-orbit,
    pre-embedded as binary rows). The selection is purely F₂ rank-driven:
    a batch is accepted iff appending its n rows to the current basis
    strictly increases its rank by exactly n. A batch that adds 0 rows is
    skipped (entirely absorbed by the basis). A batch that adds a partial
    count (``0 < delta < n``) raises — the orbit projects degenerately
    into ``ker(H_this) / rowspan(H_other)``, which would silently produce
    a rank-deficient logical basis if accepted.

    Args:
        candidate_groups: list of ``(n, n_phys)`` batches.
        group_tags: matching list of opaque tags.
        seed_rows: rowspace to quotient by — the OTHER CSS check matrix
            (e.g. ``Hz`` when selecting Z-logicals, ``Hx`` for X-logicals).
        k: target number of independent logicals to collect.

    Returns:
        (selected, selected_tags) where ``selected`` has shape
        ``(k // n, n, n_phys)`` and ``selected_tags`` is the matching list
        of group tags.

    Raises:
        RuntimeError if a batch adds a partial-but-nonzero number of
        pivots, or if fewer than ``k`` independent logicals are found
        after exhausting candidates.
    """
    basis_rref, pivot_cols = f2_rref(seed_rows)
    nonzero = np.any(basis_rref, axis=1)
    basis_rref = basis_rref[nonzero]
    pivot_cols = list(pivot_cols[:basis_rref.shape[0]])

    n = candidate_groups[0].shape[0]
    n_seed = basis_rref.shape[0]
    target_total = n_seed + k

    selected: list[np.ndarray] = []
    selected_tags: list[tuple] = []

    for group, tag in zip(candidate_groups, group_tags):
        if basis_rref.shape[0] == target_total:
            break
        # Append all n rows; let RREF count the actual rank gain.
        combined = np.vstack([basis_rref, group])
        new_basis, new_pivots = f2_rref(combined)
        nz = np.any(new_basis, axis=1)
        new_basis = new_basis[nz]
        new_pivots = list(new_pivots[:new_basis.shape[0]])

        delta = new_basis.shape[0] - basis_rref.shape[0]
        if delta == 0:
            continue
        if delta != n:
            raise RuntimeError(
                f"Batch at tag {tag} added {delta}/{n} pivots; the orbit "
                f"projects degenerately into ker(H_this) / rowspan(H_other), "
                f"so accepting it would yield a rank-deficient logical basis."
            )
        basis_rref = new_basis
        pivot_cols = new_pivots
        selected.append(group)
        selected_tags.append(tag)

    n_collected = basis_rref.shape[0] - n_seed
    if n_collected < k:
        raise RuntimeError(
            f"Only found {n_collected} independent logicals; needed {k}"
        )
    return np.array(selected, dtype=np.uint8), selected_tags


def _solve_f2_rectangular(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``M @ X = rhs`` over GF(2) for any X. ``M`` may be rectangular.

    Returns the particular solution with free variables set to 0. Raises
    ``RuntimeError`` if ``rhs`` is not in the column span of ``M``.
    """
    if rhs.ndim == 1:
        rhs_2d = rhs.reshape(-1, 1).astype(np.uint8)
    else:
        rhs_2d = rhs.astype(np.uint8)
    n_cols = M.shape[1]
    n_rhs_cols = rhs_2d.shape[1]
    aug = np.hstack([M.astype(np.uint8), rhs_2d])
    rref, pivots = f2_rref(aug)
    # Consistency: every pivot must land in the M-part (cols < n_cols).
    for p in pivots:
        if p >= n_cols:
            raise RuntimeError(
                "f2_rectangular solve: rhs is not in column span of M."
            )
    X = np.zeros((n_cols, n_rhs_cols), dtype=np.uint8)
    for r, p in enumerate(pivots):
        X[p, :] = rref[r, n_cols:]
    if rhs.ndim == 1:
        return X.ravel()
    return X


def reorder_by_pivot(Lx: np.ndarray, Lz: np.ndarray, *,
                     ref: str = "Lz") -> tuple[np.ndarray, np.ndarray]:
    """Reorder both ``Lx`` and ``Lz`` rows by their leading-pivot columns.

    The same permutation is applied to both matrices, so the pairing
    ``Lx · Lz.T`` is conjugated identically — in particular, if it was
    ``I_k`` before, it stays ``I_k`` after.

    Args:
        Lx, Lz: ``(k, n_phys)`` uint8 matrices.
        ref: ``"Lz"`` (default) sorts by the column index of each
            ``Lz[i]``'s first nonzero entry; ``"Lx"`` does the same with
            ``Lx``. Rows with no nonzero entry sort to the end.

    Returns:
        ``(Lx_sorted, Lz_sorted)``.
    """
    if Lz.shape[0] != Lx.shape[0]:
        raise ValueError(
            f"Lx and Lz must have the same row count; "
            f"got {Lx.shape[0]} and {Lz.shape[0]}."
        )
    src = Lz if ref == "Lz" else Lx if ref == "Lx" else None
    if src is None:
        raise ValueError(f"ref must be 'Lz' or 'Lx'; got {ref!r}.")
    n_cols = src.shape[1]

    def _first_one(row: np.ndarray) -> int:
        nz = np.where(row != 0)[0]
        return int(nz[0]) if nz.size else n_cols   # zero rows sort last

    pivots = np.array([_first_one(src[i]) for i in range(src.shape[0])])
    order = np.argsort(pivots, kind="stable")
    return Lx[order], Lz[order]


def find_logical_basis_pivot_aligned(Hx: np.ndarray, Hz: np.ndarray
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """Pivot-ordered logical basis with NATIVE symplectic pairing.

    Returns ``(Lx, Lz)`` such that ``Lx · Lz.T == I_k`` by construction —
    no post-hoc permutation or ``P⁻¹`` row-mix needed.

    Construction:

    1. Compute ``Lz`` via RREF extension of ``rowspan(Hz)`` to a basis of
       ``ker(Hx)``. The k chosen rows are pivot-ascending in the natural
       RREF ordering of ``Hx``'s null space.
    2. For each row index ``i ∈ [k]``, ``Lx[i]`` is the unique (up to
       stabilizer freedom) solution to ::

           Hz · Lx[i] = 0          (Lx[i] in ker(Hz))
           Lz · Lx[i] = e_i        (anti-commutes only with Lz[i])

       Stacked as ``[Hz; Lz] · Lx.T = [0; I_k]``. We solve the rectangular
       linear system over GF(2); free variables (the rowspan(Hx)
       stabilizer freedom) are set to 0.
    3. The resulting ``Lx`` has the SAME row order as ``Lz`` and ``Lx ·
       Lz.T = I_k`` natively.

    This avoids the orbit-identity permutation step in
    :func:`find_logical_basis` (which absorbs an L-vs-R rep-induced
    bijection ``φ`` via ``argmax(P, axis=0)``). The price: ``Lx`` is no
    longer guaranteed to be G-orbit-preserving (its rows are derived from
    a linear solve, not from an orbit construction).

    Args:
        Hx, Hz: binary parity-check matrices, ``(n_x_checks, n_phys)`` and
            ``(n_z_checks, n_phys)``.

    Returns:
        ``(Lx, Lz)`` — each ``(k, n_phys)`` uint8.

    Raises:
        ValueError: column-count mismatch between ``Hx`` and ``Hz``.
        RuntimeError: ``Hx``/``Hz`` aren't a valid CSS pair (the
            rectangular solve becomes inconsistent).
    """
    n_phys = int(Hx.shape[1])
    if Hz.shape[1] != n_phys:
        raise ValueError(
            f"Hx and Hz must have the same number of columns; "
            f"got {Hx.shape[1]} vs {Hz.shape[1]}"
        )
    k = n_phys - f2_rank(Hx) - f2_rank(Hz)
    if k <= 0:
        return (np.zeros((0, n_phys), dtype=np.uint8),
                np.zeros((0, n_phys), dtype=np.uint8))

    # Step 1: extract Lz in RREF-pivot order via the same _extract used by
    # find_logical_noncanonical_RREF.
    from core.f2 import f2_null_space  # local import to avoid cycle worries

    def _extract(H_check: np.ndarray, H_other: np.ndarray,
                 k_target: int) -> np.ndarray:
        null_basis = f2_null_space(H_check)
        basis_rref, pivots = f2_rref(H_other)
        nz = np.any(basis_rref, axis=1)
        basis_rref = basis_rref[nz]
        pivots = list(pivots[:basis_rref.shape[0]])
        rows: list[np.ndarray] = []
        for v in null_basis:
            v_u8 = v.astype(np.uint8)
            remainder = f2_reduce(v_u8, basis_rref, pivots)
            if not np.any(remainder):
                continue
            rows.append(v_u8)
            combined = np.vstack([basis_rref, v_u8[None, :]])
            basis_rref, new_p = f2_rref(combined)
            nz2 = np.any(basis_rref, axis=1)
            basis_rref = basis_rref[nz2]
            pivots = list(new_p[:basis_rref.shape[0]])
            if len(rows) == k_target:
                break
        if len(rows) < k_target:
            raise RuntimeError(
                f"only found {len(rows)} independent logicals; needed {k_target}"
            )
        return np.asarray(rows, dtype=np.uint8)

    Lz = _extract(Hx, Hz, k)   # Z-logicals: in ker(Hx), not in rowspan(Hz).

    # Step 2: solve [Hz; Lz] @ Lx.T = [0_block; I_k] for Lx.T.
    M = np.vstack([Hz.astype(np.uint8), Lz])
    rhs = np.zeros((M.shape[0], k), dtype=np.uint8)
    rhs[Hz.shape[0]:, :] = np.eye(k, dtype=np.uint8)
    Lx_T = _solve_f2_rectangular(M, rhs)
    Lx = Lx_T.T.astype(np.uint8)
    return Lx, Lz


def _orbit_identity_pairing(x_candidates, x_tags, z_candidates, z_tags, Hx, Hz, k, n):
    """Pick a single (X-orbit, Z-orbit) candidate pair whose pairing
    ``P = Lx · Lzᵀ`` is a *permutation* matrix, then reorder the X rows so
    ``Lx · Lzᵀ = I_k`` **without any F₂ row-mix**.

    This yields a basis that is simultaneously (a) **G-orbit-preserving** — each of
    Lx, Lz is a single G-orbit, so the operator weight is uniform per direction —
    and (b) symplectically paired (``Lx · Lzᵀ = I_k``). It works because the right
    (pivot-matched) orbit pair pairs to a permutation; the first-found orbit pair
    (what ``_select_independent_groups`` returns) often does not, which is why the
    legacy path then needs the orbit-breaking ``Lx ← P⁻¹·Lx`` mix.

    Scope / general structure: uniformity is a per-G-ORBIT property — each free block
    of ker(A_bin)/ker(B_bin) gives one orbit of n equal-weight logicals. With a SINGLE
    free block (k == n, i.e. n_groups == 1 — the 1×2 weight-3 codes here) one orbit
    fills the whole basis, so Lx (and Lz) is GLOBALLY uniform. With MULTIPLE free blocks
    (k > n, n_groups > 1) each orbit is internally uniform but different orbits may carry
    DIFFERENT weights, so the basis is only block-wise uniform; a full fix would pair each
    block to its pivot-matched partner (block-diagonal permutation P). The Lx and Lz orbit
    weights are independent and need not be equal (e.g. C3⋊D32 d14 is 44/64).

    Only the single-orbit case ``k == n`` is handled here; returns ``None`` otherwise
    (caller falls back to the P⁻¹ path). Empirically every 1×2 weight-3 code tested has
    such an identity pair.

    Returns ``(Lx, Lz, [x_tag], [z_tag])`` (uint8 (k, n_phys) each) or ``None``.
    """
    if k // n != 1:
        return None

    def _valid(group, seed):
        # group (n × n_phys) is a valid logical orbit iff appending it to rowspan(seed)
        # raises the F₂ rank by exactly n (same criterion as _select_independent_groups).
        br, _ = f2_rref(seed)
        br = br[np.any(br, axis=1)]
        base = br.shape[0]
        comb, _ = f2_rref(np.vstack([br, group]))
        return comb[np.any(comb, axis=1)].shape[0] - base == n

    valid_z = [(z, t) for z, t in zip(z_candidates, z_tags) if _valid(z, Hz)]
    valid_x = [(x, t) for x, t in zip(x_candidates, x_tags) if _valid(x, Hx)]
    for x, xt in valid_x:
        for z, zt in valid_z:
            P = (x.astype(np.int32) @ z.T.astype(np.int32)) % 2
            if (P.sum(0).max() == 1 and P.sum(0).min() == 1
                    and P.sum(1).max() == 1 and P.sum(1).min() == 1):
                # P is a permutation matrix: reorder X rows so x[perm]·zᵀ = I_k.
                perm = np.argmax(P, axis=0)         # perm[j] = X-row paired with Z-row j
                Lx = x[perm].astype(np.uint8)
                return Lx, z.astype(np.uint8), [xt], [zt]
    return None


# ---------------------------------------------------------------------------
# Quick RREF-based basis (no G-orbit structure, no LP info needed)
# ---------------------------------------------------------------------------


def find_logical_noncanonical_RREF(
    Hx: np.ndarray, Hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Any paired logical basis (Lx, Lz) using only Hx, Hz via RREF.

    For when the caller just needs *some* basis — e.g. to feed cross-type
    augmentation into a distance estimator — and doesn't care about weight,
    G-orbit structure, or LP provenance. Much faster than
    :func:`find_logical_basis` (no LP construction, no orbit selection)
    and works on any CSS pair.

    Algorithm:
      1. ``ker(Hz)`` and ``ker(Hx)`` from :func:`core.f2.f2_null_space`.
      2. For each direction, greedily pick null-space rows that extend
         the rowspan of the OTHER check matrix. Each accepted row is one
         logical; ``k = n_phys − rank(Hx) − rank(Hz)`` rows are kept.
      3. Pair: enforce ``Lx · Lz.T = I_k`` via ``Lx ← P⁻¹ · Lx`` where
         ``P = Lx · Lz.T`` (same step as the orbit-preserving path).

    Args:
        Hx, Hz: CSS parity-check matrices (uint8).

    Returns:
        ``(Lx, Lz)`` — each ``(k, n_phys)`` uint8. Paired so
        ``Lx · Lz.T == I_k (mod 2)``. ``Hx · Lz.T == 0``, ``Hz · Lx.T == 0``.

    Raises:
        RuntimeError if the pairing matrix is rank-deficient (would happen
        only if Hx, Hz aren't a valid CSS pair).
    """
    n_phys = int(Hx.shape[1])
    if Hz.shape[1] != n_phys:
        raise ValueError(
            f"Hx and Hz must have the same number of columns; "
            f"got {Hx.shape[1]} vs {Hz.shape[1]}"
        )
    k = n_phys - f2_rank(Hx) - f2_rank(Hz)
    if k <= 0:
        return (np.zeros((0, n_phys), dtype=np.uint8),
                np.zeros((0, n_phys), dtype=np.uint8))

    def _extract(H_check: np.ndarray, H_other: np.ndarray,
                 k_target: int) -> np.ndarray:
        """Find k_target vectors in ker(H_check) that extend rowspan(H_other)."""
        from core.f2 import f2_null_space
        null_basis = f2_null_space(H_check)
        basis_rref, pivots = f2_rref(H_other)
        nz = np.any(basis_rref, axis=1)
        basis_rref = basis_rref[nz]
        pivots = list(pivots[:basis_rref.shape[0]])
        rows: list[np.ndarray] = []
        for v in null_basis:
            v_u8 = v.astype(np.uint8)
            remainder = f2_reduce(v_u8, basis_rref, pivots)
            if not np.any(remainder):
                continue
            rows.append(v_u8)
            combined = np.vstack([basis_rref, v_u8[None, :]])
            basis_rref, new_p = f2_rref(combined)
            nz2 = np.any(basis_rref, axis=1)
            basis_rref = basis_rref[nz2]
            pivots = list(new_p[:basis_rref.shape[0]])
            if len(rows) == k_target:
                break
        if len(rows) < k_target:
            raise RuntimeError(
                f"only found {len(rows)} independent logicals; needed {k_target}"
            )
        return np.asarray(rows, dtype=np.uint8)

    Lx = _extract(Hz, Hx, k)   # X-logicals: in ker(Hz), not in rowspan(Hx).
    Lz = _extract(Hx, Hz, k)   # Z-logicals: in ker(Hx), not in rowspan(Hz).

    # Pair: Lx ← P⁻¹ · Lx so Lx · Lz.T = I_k.
    P = (Lx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
    if f2_rank(P) < k:
        raise RuntimeError(
            "Pairing matrix P = Lx · Lz.T is rank-deficient; Hx, Hz are "
            "not a valid CSS pair."
        )
    I_k = np.eye(k, dtype=np.uint8)
    Minv = f2_solve(P.astype(np.uint8), I_k)
    Lx = ((Minv.astype(np.int32) @ Lx.astype(np.int32)) % 2).astype(np.uint8)
    return Lx, Lz


# ---------------------------------------------------------------------------
# Public API (orbit-preserving)
# ---------------------------------------------------------------------------


def find_logical_basis(
    Hx: np.ndarray,
    Hz: np.ndarray,
    A_bin: np.ndarray,
    B_bin: np.ndarray,
    n: int,
    shape_a: tuple,
    shape_b: Optional[tuple] = None,
    *,
    canonicalize: bool = False,
    gd: Optional["GroupData"] = None,  # noqa: F821 — imported lazily
    save_dir: Optional[Path | str] = None,
    verbose: bool = False,
) -> dict:
    """Find a paired X / Z logical basis for an LP code, primary G-orbit path.

    Args:
        Hx, Hz: binary parity-check matrices, uint8. Shapes
            ``(ma·nb·n, n_phys)`` and ``(na·mb·n, n_phys)``.
        A_bin: ``build_A_bin(A, gd)`` — shape ``(ma·n, na·n)``.
        B_bin: ``build_B_bin(B, gd)`` — shape ``(mb·n, nb·n)``.
        n: group order ``|G|``.
        shape_a: ``(ma, na)``.
        shape_b: ``(mb, nb)``. Defaults to ``shape_a`` when ``None``.
        canonicalize: if ``True`` (default ``False``), the function first
            puts ``A_bin``, ``B_bin`` into canonical block-col form via
            :func:`core.classical_code.canonical_form_A_bin` /
            :func:`core.classical_code.canonical_form_B_bin`, then rebuilds
            ``Hx``, ``Hz`` from the canonical ring matrices and runs the
            basis search on the canonical pair. The canonical ``Hx``, ``Hz``
            are returned via the ``Hx_canonical`` / ``Hz_canonical`` keys.
            Requires ``gd`` to be provided.
        gd: ``GroupData`` instance. Required when ``canonicalize=True``;
            unused otherwise.
        save_dir: optional directory; if given, writes ``Lx.npy``,
            ``Lz.npy``, and ``logical_basis_meta.json``.

    Returns:
        dict with keys
            ``"Lx"``           : ``(k, n_phys)`` uint8, paired X-logicals.
            ``"Lz"``           : ``(k, n_phys)`` uint8, paired Z-logicals.
            ``"Lz_groups"``    : ``(n_groups, n, n_phys)`` uint8 — Lz reshaped
                                 so each axis-0 slice is one G-orbit of n
                                 logicals. Orbit structure on the Z side
                                 is preserved by pairing; the corresponding
                                 X side is intentionally NOT exposed as a
                                 3D array because the pairing transformation
                                 mixes Lx across orbits.
            ``"k"``            : int — number of logical qubits.
            ``"n"``            : int — group order.
            ``"n_groups"``     : int — ``k // n``.
            ``"Z_group_tags"`` : list of ``(free_block, ib)`` tuples.
            ``"X_group_tags"`` : list of ``(free_block, ja)`` tuples — these
                                 tag the X orbits BEFORE pairing.
            ``"full_rank_block_cols_A"``   : list[int].
            ``"full_rank_block_cols_B"``   : list[int].
            ``"Hx_canonical"`` : ``Hx`` rebuilt from canonical A, B if
                                 ``canonicalize=True``; ``None`` otherwise.
            ``"Hz_canonical"`` : same for Hz.
            ``"perm_a"``       : column permutation applied by
                                 ``canonical_form_A_bin`` if
                                 ``canonicalize=True``; ``None`` otherwise.
            ``"perm_b"``       : same for B.

    Invariants on the return value:
        ``Hx @ Lz.T == 0 (mod 2)``,
        ``Hz @ Lx.T == 0 (mod 2)``,
        ``Lx @ Lz.T == I_k (mod 2)``,
        ``Lz_groups.reshape(k, n_phys) == Lz``.

    Raises:
        ValueError if the code does not have a structured canonical basis
        (no full-rank ``ma``-block-col subset on either ``A_bin`` or
        ``B_bin``, or ``k % n != 0``). Pre-check via
        ``core.classical_code.canonical_form_A/B``'s ``has_full_rank_*`` flags.
        ValueError if ``canonicalize=True`` and ``gd`` is not provided.
    """
    ma, na = shape_a
    if shape_b is None:
        shape_b = (ma, na)
    mb, nb = shape_b

    # Optional: canonicalize A_bin, B_bin first and rebuild Hx, Hz on the
    # canonical pair. Forces the dep block-cols to land at the LAST positions,
    # which is the prerequisite for the orbit-pairing trick.
    Hx_canonical_out: Optional[np.ndarray] = None
    Hz_canonical_out: Optional[np.ndarray] = None
    perm_a_out: Optional[list] = None
    perm_b_out: Optional[list] = None
    if canonicalize:
        if gd is None:
            raise ValueError(
                "find_logical_basis: canonicalize=True requires `gd` "
                "(GroupData) to be passed."
            )
        from core.classical_code import (
            canonical_form_A_bin, canonical_form_B_bin,
        )
        from core.quantum_code import build_Hx, build_Hz

        A_can, A_bin_can, perm_a_out, has_a = canonical_form_A_bin(
            A_bin, gd, shape_a,
        )
        B_can, B_bin_can, perm_b_out, has_b = canonical_form_B_bin(
            B_bin, gd, shape_b,
        )
        if not (has_a and has_b):
            raise ValueError(
                "canonicalize=True: code does not have a structured "
                "canonical basis (no full-rank block-col subset on "
                f"{'A' if not has_a else ''}{'B' if not has_b else ''} side)."
            )
        # Use the canonical objects throughout the rest of the function.
        Hx = build_Hx(A_can, B_can, gd)
        Hz = build_Hz(A_can, B_can, gd)
        A_bin = A_bin_can
        B_bin = B_bin_can
        Hx_canonical_out = Hx
        Hz_canonical_out = Hz

    n_phys = int(Hx.shape[1])
    k = n_phys - f2_rank(Hx) - f2_rank(Hz)
    if k <= 0:
        raise ValueError(
            f"Code has k = {k} ≤ 0 logical qubits; nothing to find."
        )
    if k % n != 0:
        raise ValueError(
            f"k = {k} is not a multiple of |G| = {n}; the primary "
            f"G-orbit-preserving path requires k ≡ 0 (mod n). Code may "
            f"need the fallback null-space path (not supported here)."
        )

    # Eligibility checks — raise loudly if the code has no structured
    # canonical basis on either side.
    full_rank_block_cols_A = _find_full_rank_block_cols(A_bin, n, ma)     # raises ValueError on failure
    full_rank_block_cols_B = _find_full_rank_block_cols(B_bin, n, mb)

    n_groups = k // n

    # Build candidate G-orbits for both sides.
    z_codewords = _build_codewords(A_bin, n, full_rank_block_cols_A)
    z_candidates, z_cand_tags = _embed_z_logicals(z_codewords, n, na, nb, n_phys)
    x_codewords = _build_codewords(B_bin, n, full_rank_block_cols_B)
    x_candidates, x_cand_tags = _embed_x_logicals(x_codewords, n, na, nb, n_phys)

    I_k = np.eye(k, dtype=np.uint8)

    # Primary path: pick the pivot-matched orbit pair whose pairing P = Lx.Lz^T is a
    # permutation, then reorder X so Lx.Lz^T = I_k with NO row-mix. Keeps BOTH Lx and Lz
    # G-orbit-preserving (uniform operator weight per direction). Exists for every
    # single-orbit (k == |G|) code tested.
    _op = _orbit_identity_pairing(
        x_candidates, x_cand_tags, z_candidates, z_cand_tags, Hx, Hz, k, n,
    )
    if _op is not None:
        Lx, Lz, X_group_tags, Z_group_tags = _op
    else:
        if verbose:
            print(
                "[find_logical_basis] FALLBACK: orbit-identity pairing failed "
                "(no permutation-matched X/Z orbit pair found). Switching to "
                "_select_independent_groups + dense P^-1 row-mix on Lx. Lz stays "
                "G-orbit-pure; Lx is no longer orbit-preserving (weights spread)."
            )
        # Fallback (n_groups > 1, or no permutation pair): select independent orbits and
        # symplectically pair via Lx <- P^-1 . Lx. NOTE: dense P MIXES the X rows, so the
        # returned Lx is then not orbit-preserving (weights spread); Lz stays orbit-pure.
        Z_logicals, Z_group_tags = _select_independent_groups(
            z_candidates, z_cand_tags, Hz, k,
        )
        X_logicals, X_group_tags = _select_independent_groups(
            x_candidates, x_cand_tags, Hx, k,
        )
        Lz = Z_logicals.reshape(k, n_phys)
        Lx = X_logicals.reshape(k, n_phys)
        P = (Lx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
        P = P.astype(np.uint8)
        if f2_rank(P) < k:
            raise RuntimeError(
                "Pairing matrix P = Lx . Lz.T is rank-deficient over GF(2); the X and "
                "Z bases do not pair up (mismatch in Hx, Hz, A_bin, B_bin, or shape)."
            )
        Minv = f2_solve(P, I_k)
        Lx = ((Minv.astype(np.int32) @ Lx.astype(np.int32)) % 2).astype(np.uint8)

    # Cross-check invariants. These should hold by construction; raise loudly
    # if they don't (caught a bug, not a graceful fallback).
    if np.any((Hx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2):
        raise RuntimeError("Lz is not in ker(Hx); construction failed.")
    if np.any((Hz.astype(np.int32) @ Lx.T.astype(np.int32)) % 2):
        raise RuntimeError("Lx is not in ker(Hz); construction failed.")
    pairing = (Lx.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
    if not np.array_equal(pairing.astype(np.uint8), I_k):
        raise RuntimeError(
            "Pairing invariant Lx · Lz.T == I_k failed after Minv step."
        )

    result = {
        "Lx": Lx,
        "Lz": Lz,
        "Lz_groups": Lz.reshape(n_groups, n, n_phys),
        "k": int(k),
        "n": int(n),
        "n_groups": int(n_groups),
        "Z_group_tags": [tuple(int(x) for x in t) for t in Z_group_tags],
        "X_group_tags": [tuple(int(x) for x in t) for t in X_group_tags],
        "full_rank_block_cols_A": [int(c) for c in full_rank_block_cols_A],
        "full_rank_block_cols_B": [int(c) for c in full_rank_block_cols_B],
        "Hx_canonical": Hx_canonical_out,
        "Hz_canonical": Hz_canonical_out,
        "perm_a": perm_a_out,
        "perm_b": perm_b_out,
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "Lx.npy", Lx)
        np.save(save_path / "Lz.npy", Lz)
        meta = {key: result[key] for key in (
            "k", "n", "n_groups",
            "Z_group_tags", "X_group_tags",
            "full_rank_block_cols_A", "full_rank_block_cols_B",
            "perm_a", "perm_b",
        )}
        with open(save_path / "logical_basis_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return result
