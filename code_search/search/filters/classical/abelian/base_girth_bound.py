"""Upper bound on the Tanner girth of A_bin from the integer weight matrix alone.

**Abelian only.** Each rule below is justified by a relator that closes a
short bipartite cycle, e.g. for any 3-element abelian subset ``{g₁,g₂,g₃}``
the relator ``(g₁−g₂)+(g₃−g₁)+(g₂−g₃) = 0`` always yields a simple 6-cycle
in the Tanner graph of ``L[{g₁,g₂,g₃}]``. The analogous identities require
commutativity; in non-abelian groups they are not theorems, and a Moore-
type lower bound on girth can exceed the values returned here. Use only
when ``gd.is_abelian``; the dispatcher enforces this.

Needs only the integer weight matrix — no GAP, no ring sampling. Cheapest
filter.
"""

import numpy as np


def base_girth_bound(W) -> int | float:
    """Upper bound on Tanner girth from an integer weight matrix.

    Takes only support sizes — does NOT need GroupData. Applicable to any
    LP code regardless of whether G is abelian.

    Args:
        W: integer array of shape ``(ma, na)``. ``W[ia][ja]`` is the
           number of group elements in ring entry ``A[ia][ja]``.

    Returns:
        6, 8, 10, 12, or ``float('inf')`` (no theorem-based bound).

    Rules (earliest trigger wins):
      any entry ≥ 3                                  → girth ≤ 6
      two entries ≥ 2 in the same row or column      → girth ≤ 8
      2×2 all-nonzero sub-block with ≥1 entry ≥ 2    → girth ≤ 10
      2×3 or 3×2 all-nonzero sub-block               → girth ≤ 12
      otherwise                                       → ∞
    """
    W = np.asarray(W, dtype=int)
    rows, cols = W.shape

    if np.any(W >= 3):
        return 6

    for r in range(rows):
        if int(np.sum(W[r] >= 2)) >= 2:
            return 8
    for c in range(cols):
        if int(np.sum(W[:, c] >= 2)) >= 2:
            return 8

    # girth ≤ 10: 2×2 all-nonzero with at least one entry ≥ 2
    for r1 in range(rows):
        for r2 in range(r1 + 1, rows):
            shared = np.where((W[r1] >= 1) & (W[r2] >= 1))[0]
            if len(shared) >= 2:
                if np.any(W[r1, shared] >= 2) or np.any(W[r2, shared] >= 2):
                    return 10

    # girth ≤ 12: 2×3 or 3×2 all-nonzero sub-block
    for r1 in range(rows):
        for r2 in range(r1 + 1, rows):
            shared = np.where((W[r1] >= 1) & (W[r2] >= 1))[0]
            if len(shared) >= 3:
                return 12
    for c1 in range(cols):
        for c2 in range(c1 + 1, cols):
            shared = np.where((W[:, c1] >= 1) & (W[:, c2] >= 1))[0]
            if len(shared) >= 3:
                return 12

    return float("inf")
