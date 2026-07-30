"""Abelian-only distance upper bound via ring permanents.

For a ``J×I`` ring matrix ``A`` (``J = ma``, ``I = na``), abelian ``G``:

    d(A_bin) ≤ min over S ⊆ {0..I-1}, |S| = J+1
              of  Σ_{i ∈ S}  |ring_permanent(A[:, S \\ {i}])|

The permanent of the J×J submatrix formed by columns ``S \\ {i}`` is a
ring element in ``F₂[G]``; its weight (number of group elements)
contributes to the bound.

Lives in ``abelian/`` only (no re-export shell elsewhere) because the
permanent of a ring matrix over non-commutative ``F₂[G]`` does not have
the symmetry properties this bound relies on.
"""

import itertools

from core.group import GroupData, ring_permanent


def ring_distance_bound(A: list, gd: GroupData) -> int | float:
    """Upper bound on ``d(A_bin)`` via ring permanents.

    Args:
        A: ring matrix shape ``(J, I)``. Abelian ``gd`` required.
        gd: GroupData. Must satisfy ``gd.is_abelian``.

    Returns:
        Integer bound, or ``float('inf')`` if ``J+1 > I`` (no valid
        subsets).

    Raises:
        ValueError if ``gd`` is non-abelian.

    Special cases (no ring multiplication needed):
        J=1, I=2 → ``|A[0][0]| + |A[0][1]|``
        J=1, I=3 → min over the three pairwise sums
        J≥2     → genuine ``ring_permanent`` calls
    """
    if not gd.is_abelian:
        raise ValueError(
            f"ring_distance_bound requires an abelian group; "
            f"got {gd.structure!r}, is_abelian=False"
        )
    J = len(A)
    I = len(A[0])
    target_size = J + 1
    if target_size > I:
        return float("inf")

    min_bound: float = float("inf")
    for S in itertools.combinations(range(I), target_size):
        total = 0
        for i_idx in range(len(S)):
            cols_kept = [c for c in S if c != S[i_idx]]
            sub = [[A[row][col] for col in cols_kept] for row in range(J)]
            total += len(ring_permanent(sub, gd))
        if total < min_bound:
            min_bound = total

    return int(min_bound) if min_bound < float("inf") else float("inf")
