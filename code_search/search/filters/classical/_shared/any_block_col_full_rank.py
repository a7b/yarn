"""Structured-canonical-basis predicate on ``A_bin``.

Returns ``True`` iff there exists a subset of ``ma`` block-cols of
``A_bin`` whose combined ``ma·n × ma·n`` submatrix is invertible over
GF(2). This is the gating condition for the G-orbit-preserving logical
basis construction (`logical_basis.find_logical_basis`) and for surgery.

Strictly stronger than "A_bin has full row rank": a code can have full
row-rank A_bin without any ``ma``-sized block-col subset being invertible.
"""

import itertools

import numpy as np

from core.f2 import f2_rank


def any_block_col_full_rank(A_bin: np.ndarray, n: int, ma: int = 1) -> bool:
    """``True`` iff some ``ma``-subset of block-cols of ``A_bin`` is invertible.

    Args:
        A_bin: ``(ma·n, na·n)`` binary matrix.
        n: ``|G|`` (block-col width).
        ma: number of block-cols to combine (default ``1`` — the standard
            per-block check). For ``ma > 1`` we check ``C(na, ma)`` subsets.

    Returns:
        ``True`` iff there is a subset of ``ma`` block-cols whose combined
        submatrix is full-rank.
    """
    n_blocks = A_bin.shape[1] // n
    target = ma * n
    for cols in itertools.combinations(range(n_blocks), ma):
        idx = [c * n + k for c in cols for k in range(n)]
        sub = A_bin[:, idx]
        if sub.shape[0] == sub.shape[1] and f2_rank(sub) == target:
            return True
    return False
