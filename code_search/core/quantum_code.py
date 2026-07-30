"""LP code construction at the CSS quantum-code level.

Forward (ring → binary):
    build_Hx(A, B, gd) -> Hx                   — uses (A, B) as-is
    build_Hz(A, B, gd) -> Hz                   — uses (A, B) as-is
    build_quantum_code(A, B, gd)               -> dict {Hx, Hz, canonical info}

Code parameters:
    compute_k(Hx, Hz)      -> int       n_phys - rank(Hx) - rank(Hz)
    check_css(Hx, Hz)      -> bool      Hx · Hzᵀ == 0 mod 2

Check-weight derivation (no group machinery — purely arithmetic on
``weight_matrix(A)`` and ``weight_matrix(B)``):
    quantum_check_weights(W_A, W_B) -> {Hx_check_weight, Hz_check_weight}
        Each is the MAX row weight (sparsity) of Hx / Hz.

Inverse (binary → ring):
    A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd, shape_a, shape_b=None) -> (A_bin, B_bin)
        validates LP structure of each left section.
    AB_from_Hx_Hz(Hx, Hz, gd, shape_a, shape_b=None)          -> (A, B)
        chains the above through A_from_A_bin / B_from_B_bin, then rebuilds
        (Hx, Hz) and asserts both match — full cross-check.

Conventions:
- A : ma × na (left-regular rep in the left section of Hx).
- B : mb × nb (right-regular rep in the left section of Hz).
- Default mb = ma, nb = na (set when shape_b is None); non-default sizes
  are supported.
- build_Hx / build_Hz do NOT canonicalize. Use build_quantum_code if you
  want canonical-form Hx/Hz with their column permutations exposed.

Block placement (LP block-Kronecker ⊗_lp):
    Hx = [ L[A] ⊗_lp id_nb  |  id_ma ⊗_lp R[B†] ]
        shape (ma·nb·n) × (na·nb + ma·mb)·n
    Hz = [ id_na ⊗_lp R[B]  |  L[A†] ⊗_lp id_mb ]
        shape (na·mb·n) × (na·nb + ma·mb)·n
"""

import numpy as np

from core.classical_code import (
    A_from_A_bin,
    B_from_B_bin,
    canonical_form_A,
    canonical_form_B,
)
from core.f2 import f2_rank
from core.group import GroupData, dagger, left_rep, right_rep


# ─────────────────────────────────────────────────────────────────
# Forward: ring → binary
# ─────────────────────────────────────────────────────────────────


def _build_Hx_raw(A: list, B: list, gd: GroupData) -> np.ndarray:
    """Build Hx from (A, B) without any canonicalization.

    Hx = [ L[A] ⊗_lp id_nb | id_ma ⊗_lp R[B†] ]
    Shape: (ma·nb·n) × (na·nb + ma·mb)·n.
    """
    ma, na = len(A), len(A[0])
    mb, nb = len(B), len(B[0])
    if not all(len(row) == na for row in A):
        raise ValueError(
            f"A is not rectangular: row lengths {[len(r) for r in A]} "
            f"(expected all == {na})."
        )
    if not all(len(row) == nb for row in B):
        raise ValueError(
            f"B is not rectangular: row lengths {[len(r) for r in B]} "
            f"(expected all == {nb})."
        )
    n = gd.n

    rows = ma * nb * n
    cols = (na * nb + ma * mb) * n
    Hx = np.zeros((rows, cols), dtype=np.uint8)

    # Left section: L[A] ⊗_lp id_nb
    for ia in range(ma):
        for ja in range(na):
            LA = left_rep(A[ia][ja], gd)
            for ib in range(nb):
                r = (ia * nb + ib) * n
                c = (ja * nb + ib) * n
                Hx[r:r + n, c:c + n] ^= LA

    # Right section: id_ma ⊗_lp R[B†]
    # B†[ib][jb] = dagger(B[jb][ib])  for ib ∈ [nb], jb ∈ [mb]
    left_cols = na * nb * n
    for ib in range(nb):
        for jb in range(mb):
            bd = dagger(B[jb][ib], gd)
            RBd = right_rep(bd, gd)
            for ia in range(ma):
                r = (ia * nb + ib) * n
                c = left_cols + (ia * mb + jb) * n
                Hx[r:r + n, c:c + n] ^= RBd

    return Hx


def _build_Hz_raw(A: list, B: list, gd: GroupData) -> np.ndarray:
    """Build Hz from (A, B) without any canonicalization.

    Hz = [ id_na ⊗_lp R[B] | L[A†] ⊗_lp id_mb ]
    Shape: (na·mb·n) × (na·nb + ma·mb)·n.
    """
    ma, na = len(A), len(A[0])
    mb, nb = len(B), len(B[0])
    if not all(len(row) == na for row in A):
        raise ValueError(
            f"A is not rectangular: row lengths {[len(r) for r in A]} "
            f"(expected all == {na})."
        )
    if not all(len(row) == nb for row in B):
        raise ValueError(
            f"B is not rectangular: row lengths {[len(r) for r in B]} "
            f"(expected all == {nb})."
        )
    n = gd.n

    rows = na * mb * n
    cols = (na * nb + ma * mb) * n
    Hz = np.zeros((rows, cols), dtype=np.uint8)

    # Left section: id_na ⊗_lp R[B]
    for ib in range(mb):
        for jb in range(nb):
            RB = right_rep(B[ib][jb], gd)
            for ja in range(na):
                r = (ja * mb + ib) * n
                c = (ja * nb + jb) * n
                Hz[r:r + n, c:c + n] ^= RB

    # Right section: L[A†] ⊗_lp id_mb
    # A†[ja][ia] = dagger(A[ia][ja]) for ja ∈ [na], ia ∈ [ma]
    left_cols = na * nb * n
    for ja in range(na):
        for ia in range(ma):
            ad = dagger(A[ia][ja], gd)
            LAd = left_rep(ad, gd)
            for ib in range(mb):
                r = (ja * mb + ib) * n
                c = left_cols + (ia * mb + ib) * n
                Hz[r:r + n, c:c + n] ^= LAd

    return Hz


def build_Hx(A: list, B: list, gd: GroupData) -> np.ndarray:
    """Build the X parity-check matrix Hx of (A, B), using (A, B) as-is.

    Does not canonicalize. Use `build_quantum_code` if you want canonical-form
    Hx — that returns the permutations needed to interpret the column ordering.
    """
    return _build_Hx_raw(A, B, gd)


def build_Hz(A: list, B: list, gd: GroupData) -> np.ndarray:
    """Build the Z parity-check matrix Hz of (A, B), using (A, B) as-is.

    Does not canonicalize. Use `build_quantum_code` if you want canonical-form
    Hz — that returns the permutations needed to interpret the column ordering.
    """
    return _build_Hz_raw(A, B, gd)


def build_quantum_code(A: list, B: list, gd: GroupData) -> dict:
    """High-level CSS code builder.

    Always canonicalizes A and B via `canonical_form_A` / `canonical_form_B`,
    builds Hx and Hz from the canonical pair, and returns everything in one
    dict so callers can interpret the column ordering.

    Returns a dict with keys:
        Hx, Hz                            — binary parity-check matrices
        A_canonical, A_bin_canonical      — ring matrix + binary lift after canon
        B_canonical, B_bin_canonical
        perm_a, perm_b                    — column permutations applied
        has_full_rank_a, has_full_rank_b  — surgery-eligibility flags
    """
    A_can, A_bin_can, perm_a, found_a = canonical_form_A(A, gd)
    B_can, B_bin_can, perm_b, found_b = canonical_form_B(B, gd)
    Hx = _build_Hx_raw(A_can, B_can, gd)
    Hz = _build_Hz_raw(A_can, B_can, gd)
    return {
        "Hx": Hx,
        "Hz": Hz,
        "A_canonical": A_can,
        "A_bin_canonical": A_bin_can,
        "B_canonical": B_can,
        "B_bin_canonical": B_bin_can,
        "perm_a": perm_a,
        "perm_b": perm_b,
        "has_full_rank_a": found_a,
        "has_full_rank_b": found_b,
    }


# ─────────────────────────────────────────────────────────────────
# Code parameters
# ─────────────────────────────────────────────────────────────────


def compute_k(Hx: np.ndarray, Hz: np.ndarray) -> int:
    """Number of logical qubits k = n_phys - rank(Hx) - rank(Hz) (over GF(2)).

    The formula is only physically meaningful when Hx · Hzᵀ = 0 (CSS).
    The caller is responsible for verifying that — use `check_css` first
    if the input might not be a valid CSS pair.
    """
    return int(Hx.shape[1] - f2_rank(Hx) - f2_rank(Hz))


def check_css(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    """CSS orthogonality: Hx · Hzᵀ == 0 (mod 2).

    Nonzero entries count as 1 (the same convention as `core.f2`).
    """
    Hx01 = (np.asarray(Hx) != 0).astype(np.uint8)
    Hz01 = (np.asarray(Hz) != 0).astype(np.uint8)
    return bool(np.all((Hx01 @ Hz01.T) % 2 == 0))


# ─────────────────────────────────────────────────────────────────
# Check-weight derivation from weight matrices
# ─────────────────────────────────────────────────────────────────


def quantum_check_weights(W_A: np.ndarray, W_B: np.ndarray) -> dict:
    """Quantum check weights = MAX row weights of Hx and Hz.

    All rows in the ``(ia, ib)`` block of ``Hx`` have the same weight
    (every row ``k`` of ``L[x]`` has weight ``|x|`` — no cancellation).
    The block-row weight is::

        Hx[ia, ib] row weight = rowsum(W_A, ia) + colsum(W_B, ib)

    so the maximum row weight over all of Hx is::

        max_ia,ib (rowsum(W_A, ia) + colsum(W_B, ib))
            = max(rowsum(W_A))      [ia free]
            + max(colsum(W_B))      [ib free, independent]

    By the symmetric LP block-Kronecker on Hz::

        Hz[ja, ib] row weight = colsum(W_A, ja) + rowsum(W_B, ib)
        max over Hz             = max(colsum(W_A)) + max(rowsum(W_B))

    Returned as two scalars — these are the "sparsity caps" used by
    search configs and pairing filters. No GAP needed; purely integer
    arithmetic on W_A, W_B.

    Args:
        W_A: integer ndarray of shape ``(ma, na)`` (``weight_matrix(A)``).
        W_B: integer ndarray of shape ``(mb, nb)``.

    Returns:
        ``{"Hx_check_weight": int, "Hz_check_weight": int}`` — the max
        row weights of Hx and Hz. Returns 0 for an empty side.
    """
    W_A = np.asarray(W_A, dtype=int)
    W_B = np.asarray(W_B, dtype=int)
    if W_A.ndim != 2 or W_B.ndim != 2:
        raise ValueError(
            f"W_A and W_B must be 2D; got shapes {W_A.shape}, {W_B.shape}."
        )
    rowsum_A_max = int(W_A.sum(axis=1).max()) if W_A.size else 0
    colsum_A_max = int(W_A.sum(axis=0).max()) if W_A.size else 0
    rowsum_B_max = int(W_B.sum(axis=1).max()) if W_B.size else 0
    colsum_B_max = int(W_B.sum(axis=0).max()) if W_B.size else 0
    return {
        "Hx_check_weight": rowsum_A_max + colsum_B_max,
        "Hz_check_weight": colsum_A_max + rowsum_B_max,
    }


# ─────────────────────────────────────────────────────────────────
# Inverse: binary → ring (binary lifts → ring matrices)
# ─────────────────────────────────────────────────────────────────


def A_bin_B_bin_from_Hx_Hz(Hx: np.ndarray, Hz: np.ndarray, gd: GroupData,
                           shape_a: tuple,
                           shape_b: tuple = None) -> tuple:
    """Extract binary lifts (A_bin, B_bin) from (Hx, Hz), with LP validation.

    A_bin is read off the ib=0 fiber of Hx's left section:
        A_bin[ia·n:(ia+1)·n, ja·n:(ja+1)·n]
            = Hx[ia·nb·n : ia·nb·n+n, ja·nb·n : ja·nb·n+n]
    B_bin is read off the ja=0 fiber of Hz's left section:
        B_bin[ib·n:(ib+1)·n, jb·n:(jb+1)·n]
            = Hz[ib·n : (ib+1)·n, jb·n : (jb+1)·n]

    Validates that the entire left section of each matrix matches the LP
    block-Kronecker lift of the extracted fiber — every diagonal slice agrees
    and every off-diagonal block is zero. Raises ValueError on mismatch, so a
    successful return guarantees A_bin and B_bin are uniquely determined by
    Hx and Hz under the LP convention. (The right sections and Hx/Hz
    cross-consistency are NOT checked here — use AB_from_Hx_Hz for that.)

    Args:
        Hx: (ma·nb·n) × (na·nb + ma·mb)·n binary.
        Hz: (na·mb·n) × (na·nb + ma·mb)·n binary.
        gd: GroupData.
        shape_a: (ma, na).
        shape_b: (mb, nb); defaults to (ma, na).

    Returns:
        (A_bin, B_bin) — binary lifts of shape (ma·n, na·n) and (mb·n, nb·n).

    Raises:
        ValueError on shape mismatch or LP-inconsistency in either left
        section.
    """
    ma, na = shape_a
    if shape_b is None:
        shape_b = (ma, na)
    mb, nb = shape_b
    n = gd.n

    expected_hx_shape = (ma * nb * n, (na * nb + ma * mb) * n)
    expected_hz_shape = (na * mb * n, (na * nb + ma * mb) * n)
    if Hx.shape != expected_hx_shape:
        raise ValueError(
            f"Hx shape {Hx.shape} != expected {expected_hx_shape} for "
            f"shape_a={shape_a}, shape_b={shape_b}, n={n}."
        )
    if Hz.shape != expected_hz_shape:
        raise ValueError(
            f"Hz shape {Hz.shape} != expected {expected_hz_shape} for "
            f"shape_a={shape_a}, shape_b={shape_b}, n={n}."
        )

    # Read A_bin from the ib=0 fiber of Hx's left section.
    A_bin = np.zeros((ma * n, na * n), dtype=np.uint8)
    for ia in range(ma):
        for ja in range(na):
            A_bin[ia * n:(ia + 1) * n, ja * n:(ja + 1) * n] = \
                Hx[ia * nb * n:ia * nb * n + n,
                   ja * nb * n:ja * nb * n + n]

    # Validate: Hx's left section must equal the LP lift A_bin ⊗_lp I_nb.
    # Rebuilding and comparing catches both off-diagonal noise and disagreement
    # between ib slices — the silent failure modes that the old fiber-read had.
    hx_left_cols = na * nb * n
    expected_Hx_left = np.zeros((ma * nb * n, hx_left_cols), dtype=np.uint8)
    for ia in range(ma):
        for ja in range(na):
            block = A_bin[ia * n:(ia + 1) * n, ja * n:(ja + 1) * n]
            for ib in range(nb):
                r = (ia * nb + ib) * n
                c = (ja * nb + ib) * n
                expected_Hx_left[r:r + n, c:c + n] = block
    if not np.array_equal(Hx[:, :hx_left_cols], expected_Hx_left):
        raise ValueError(
            "Hx left section is not LP-consistent with its ib=0 fiber: "
            "different ib slices disagree, or off-diagonal blocks are nonzero. "
            "Hx is not a valid LP CSS code of the claimed shape."
        )

    # Read B_bin from the ja=0 fiber of Hz's left section.
    B_bin = np.zeros((mb * n, nb * n), dtype=np.uint8)
    for ib in range(mb):
        for jb in range(nb):
            B_bin[ib * n:(ib + 1) * n, jb * n:(jb + 1) * n] = \
                Hz[ib * n:(ib + 1) * n, jb * n:(jb + 1) * n]

    # Validate: Hz's left section must equal the LP lift I_na ⊗_lp B_bin
    # (using the Hz row-flatten: row-block (ja, ib), col-block (ja, jb)).
    hz_left_cols = na * nb * n
    expected_Hz_left = np.zeros((na * mb * n, hz_left_cols), dtype=np.uint8)
    for ib in range(mb):
        for jb in range(nb):
            block = B_bin[ib * n:(ib + 1) * n, jb * n:(jb + 1) * n]
            for ja in range(na):
                r = (ja * mb + ib) * n
                c = (ja * nb + jb) * n
                expected_Hz_left[r:r + n, c:c + n] = block
    if not np.array_equal(Hz[:, :hz_left_cols], expected_Hz_left):
        raise ValueError(
            "Hz left section is not LP-consistent with its ja=0 fiber: "
            "different ja slices disagree, or off-diagonal blocks are nonzero. "
            "Hz is not a valid LP CSS code of the claimed shape."
        )

    return A_bin, B_bin


def AB_from_Hx_Hz(Hx: np.ndarray, Hz: np.ndarray, gd: GroupData,
                  shape_a: tuple,
                  shape_b: tuple = None) -> tuple:
    """Recover ring matrices (A, B) from (Hx, Hz) with cross-check.

    Chains `A_bin_B_bin_from_Hx_Hz` → `A_from_A_bin` / `B_from_B_bin`, then
    rebuilds Hx and Hz from the recovered (A, B) and asserts both match the
    inputs. Raises ValueError if either rebuild fails — this catches Hx/Hz
    pairs that aren't a valid CSS LP code of the claimed shape.

    Args:
        Hx, Hz: binary parity-check matrices.
        gd: GroupData.
        shape_a: (ma, na).
        shape_b: (mb, nb); defaults to (ma, na).

    Returns:
        (A, B) — ring matrices in canonical form.

    Raises:
        ValueError on shape mismatch, invalid lift, or cross-check failure.
    """
    ma, na = shape_a
    if shape_b is None:
        shape_b = (ma, na)

    A_bin, B_bin = A_bin_B_bin_from_Hx_Hz(Hx, Hz, gd, shape_a, shape_b)
    A = A_from_A_bin(A_bin, gd, shape_a)
    B = B_from_B_bin(B_bin, gd, shape_b)

    Hx_rebuilt = _build_Hx_raw(A, B, gd)
    Hz_rebuilt = _build_Hz_raw(A, B, gd)
    if not np.array_equal(Hx, Hx_rebuilt):
        raise ValueError(
            "Hx cross-check failed: rebuild of Hx from recovered (A, B) "
            "differs from input. (Hx, Hz) may not be a valid LP CSS code "
            "of the claimed shape."
        )
    if not np.array_equal(Hz, Hz_rebuilt):
        raise ValueError(
            "Hz cross-check failed: rebuild of Hz from recovered (A, B) "
            "differs from input."
        )

    return A, B
