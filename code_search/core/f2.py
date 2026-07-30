"""GF(2) binary linear algebra.

All functions:
- Treat any nonzero input entry as ``1`` (so dtype doesn't matter).
- Do not mutate inputs.
- Return uint8 arrays.
"""

import numpy as np


def _to_binary(M: np.ndarray) -> np.ndarray:
    """Internal: copy + normalize any-dtype 2D array to a uint8 {0,1} matrix."""
    return (M != 0).astype(np.uint8)


def f2_rank(H: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) via Gaussian elimination.

    Nonzero entries are treated as 1; the input is not modified.
    """
    M = _to_binary(H)
    nrows, ncols = M.shape
    rank = 0
    for col in range(ncols):
        pivot = None
        for row in range(rank, nrows):
            if M[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(nrows):
            if row != rank and M[row, col]:
                M[row] ^= M[rank]
        rank += 1
    return rank


def f2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form of a binary matrix over GF(2).

    Returns:
        (R, pivot_cols) where R is the RREF of M (uint8) and pivot_cols is
        the list of column indices that hold a pivot, in row order. The
        rank equals ``len(pivot_cols)``.
    """
    R = _to_binary(M)
    nrows, ncols = R.shape
    pivot_cols: list[int] = []
    cur_row = 0
    for col in range(ncols):
        pivot = None
        for row in range(cur_row, nrows):
            if R[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        R[[cur_row, pivot]] = R[[pivot, cur_row]]
        pivot_cols.append(col)
        for row in range(nrows):
            if row != cur_row and R[row, col]:
                R[row] ^= R[cur_row]
        cur_row += 1
        if cur_row == nrows:
            break
    return R, pivot_cols


def f2_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve ``A · X = B`` over GF(2).

    Args:
        A: ``(n, n)`` square invertible binary matrix.
        B: ``(n, m)`` binary right-hand side.

    Returns:
        ``X``, the ``(n, m)`` solution.

    Raises:
        ValueError if ``A`` is not square, has incompatible shape with
        ``B``, or is not full rank over GF(2).
    """
    A_b = _to_binary(A)
    B_b = _to_binary(B)
    n = A_b.shape[0]
    if A_b.shape != (n, n):
        raise ValueError(f"A must be square; got shape {A_b.shape}")
    if B_b.ndim != 2 or B_b.shape[0] != n:
        raise ValueError(
            f"B must have shape ({n}, m); got {B_b.shape}"
        )
    AB = np.concatenate([A_b, B_b], axis=1)
    R, pivots = f2_rref(AB)
    if len(pivots) != n or pivots != list(range(n)):
        raise ValueError("A is not full rank over GF(2)")
    return R[:, n:]


def f2_reduce(v: np.ndarray, basis_rref: np.ndarray,
              pivot_cols: list[int]) -> np.ndarray:
    """Reduce binary vector ``v`` against an RREF basis.

    Subtracts (XORs) rows of ``basis_rref`` to zero out the pivot-column
    entries of ``v``. Returns the remainder.

    Args:
        v: length-``ncols`` binary vector.
        basis_rref: RREF matrix; usually from :func:`f2_rref`.
        pivot_cols: pivot columns of ``basis_rref``.

    Returns:
        The reduced binary vector (uint8). Length matches ``v``.
    """
    out = _to_binary(v.reshape(1, -1))[0]   # length-ncols 1-D
    for i, col in enumerate(pivot_cols):
        if col < out.shape[0] and out[col]:
            out ^= basis_rref[i]
    return out


def f2_null_space(M: np.ndarray) -> np.ndarray:
    """Right null space of a binary matrix ``M`` over GF(2).

    Returns:
        Array of shape ``(dim_ker, ncols)``; each row is a vector ``v``
        with ``M · v = 0 (mod 2)``. ``dim_ker = ncols - rank(M)``.

    Algorithm: RREF on ``[M.T | I_ncols]``; non-pivot rows of the RREF's
    right block are exactly the null-space basis.
    """
    M_b = _to_binary(M)
    nrows, ncols = M_b.shape
    aug = np.concatenate([M_b.T, np.eye(ncols, dtype=np.uint8)], axis=1)
    R, pivot_cols = f2_rref(aug)
    # Pivots in the M.T block (cols 0..nrows-1) count toward rank(M).
    # Pivots in the identity block (cols nrows..) are bookkeeping artifacts
    # and must not inflate the rank.
    rank = sum(1 for c in pivot_cols if c < nrows)
    return R[rank:, nrows:]
