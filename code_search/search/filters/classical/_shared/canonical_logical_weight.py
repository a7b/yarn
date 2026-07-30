"""Canonical (G-orbit) logical-operator weight of one classical side.

For the lifted-product codes here, the canonical paired logical basis
(``logical_basis.find_logical_basis``) is built from **structured kernel
codewords** of the classical lifts: each logical is one G-orbit codeword of
``ker(M_bin)`` — a unit vector in a *free* block-column plus the forced solve in
the *full-rank* block-column — embedded weight-preservingly into the quantum
code. Consequently the canonical logical weight is a property of a **single
classical side**, readable before any pairing or quantum build:

* the **A**-side fixes the **Z**-logical weight ( ``wt(Lz) = wt`` of the
  structured ``ker(A_bin)`` orbit codeword ),
* the **B**-side fixes the **X**-logical weight ( ``wt(Lx)`` from ``ker(B_bin)`` ).

(Verified exact against ``find_logical_basis`` on the 1x2 weight-3 family,
abelian and non-abelian.) A heavy canonical logical means heavy lattice-surgery
/ extractor gadgets, so a search can prune such sides up front.

This is the *canonical basis-representative* weight (gadget overhead), NOT the
code distance ``d`` (the *minimum* logical weight, which is ``<=`` this). Exact
for the single-orbit (``k = |G|``) canonical basis; ``None`` if the side has no
full-rank block-column (no structured canonical basis — handled by
``require_any_block_col_full_rank``).

Public deps: numpy + the kernel-codeword builder shared with ``find_logical_basis``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from logical_basis.logical_basis import _build_codewords, _find_full_rank_block_cols


def canonical_logical_weight(M_bin: np.ndarray, n: int) -> Optional[int]:
    """Weight of the canonical G-orbit logical operator produced by side ``M``.

    Args:
        M_bin: the binary lift of one ring matrix (``build_A_bin``/``build_B_bin``),
            shape ``(m*n, na*n)`` over GF(2).
        n: group order ``|G|`` (block size).

    Returns:
        The structured-orbit kernel-codeword weight (an ``int`` in
        ``[1, na*n]``), or ``None`` if ``M_bin`` has no full-rank block-column
        (then there is no canonical basis on this side).

    Notes:
        For a single-orbit code the orbit is weight-uniform, so this equals the
        canonical ``find_logical_basis`` logical weight exactly. We return the
        ``max`` over the constructed orbit(s), a conservative upper bound for the
        multi-orbit case (where ``find_logical_basis`` selects a subset).
    """
    M_bin = np.asarray(M_bin, np.uint8) & 1
    n_blocks = M_bin.shape[0] // n          # = m (rows of the ring matrix)
    try:
        frc = _find_full_rank_block_cols(M_bin, n, n_blocks)
    except (ValueError, RuntimeError):
        return None
    cw = _build_codewords(M_bin, n, frc)    # (n_free_blocks, n, na*n)
    flat = cw.reshape(-1, cw.shape[-1])
    if flat.size == 0:
        return None
    return int(flat.sum(axis=1).max())
