"""Weight-matrix-only distance upper bound (abelian-only, cheap first step).

For a ``J×I`` ring matrix ``A`` over an abelian group ``G``, replace the
ring permanent of every J×J submatrix in :mod:`ring_distance_bound` with
the **integer permanent** of the corresponding submatrix of
``weight_matrix(A)``. No GAP, no ring multiplication.

The integer permanent upper-bounds the ring permanent's weight:
    |ring_permanent(A[:, S])| ≤ permanent(W_A[:, S])
because each term ``Π_i A[i][σ(i)]`` is a ring element of weight at most
``Π_i |A[i][σ(i)]|`` (and the sum over σ can only cancel / merge, not
grow). So this bound is **always valid** but only **tight** at ``J=1``,
where the J×J permanent collapses to a single entry's weight.

Use this as a CHEAP pre-filter; if it passes, run
:func:`ring_distance_bound` for the tighter (and possibly only correct)
test. For ``J=1`` the two are identical — keep both names so callers
can express intent.
"""

import itertools

import numpy as np

from core.group import GroupData


def _integer_permanent(W: np.ndarray) -> int:
    """Permanent of a square integer matrix via Ryser-style enumeration.

    Small ``J`` only (we expect ``J ≤ 4`` in practice). Pure-Python
    permutation sum is fine.
    """
    J = W.shape[0]
    if W.shape[1] != J:
        raise ValueError(f"_integer_permanent expects square; got {W.shape}.")
    total = 0
    for sigma in itertools.permutations(range(J)):
        prod = 1
        for i in range(J):
            prod *= int(W[i, sigma[i]])
        total += prod
    return total


def weight_distance_bound(W_A: np.ndarray, gd: GroupData) -> "int | float":
    """Weight-only upper bound on ``d(A_bin)`` (abelian).

    Args:
        W_A: integer weight matrix of shape ``(J, I)`` —
            ``W_A[i, j] = |A[i][j]|``. (Use
            :func:`core.classical_code.weight_matrix`.)
        gd: GroupData. Must be abelian.

    Returns:
        Integer bound, or ``float('inf')`` if ``J+1 > I``.

    Raises:
        ValueError if ``gd`` is non-abelian or ``W_A`` is not 2D.

    Identical to :func:`ring_distance_bound` for ``J=1`` (no ring
    multiplication occurs there). For ``J ≥ 2`` this returns a possibly
    looser bound (no cancellations / mergers tracked).
    """
    if not gd.is_abelian:
        raise ValueError(
            f"weight_distance_bound requires an abelian group; "
            f"got {gd.structure!r}, is_abelian=False"
        )
    W_A = np.asarray(W_A, dtype=int)
    if W_A.ndim != 2:
        raise ValueError(f"W_A must be 2D; got shape {W_A.shape}.")
    J, I = W_A.shape
    target_size = J + 1
    if target_size > I:
        return float("inf")

    min_bound: float = float("inf")
    for S in itertools.combinations(range(I), target_size):
        total = 0
        for i_idx in range(len(S)):
            cols_kept = [c for c in S if c != S[i_idx]]
            sub = W_A[:, cols_kept]
            total += _integer_permanent(sub)
        if total < min_bound:
            min_bound = total
    return int(min_bound) if min_bound < float("inf") else float("inf")
