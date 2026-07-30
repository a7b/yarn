"""Classical distance estimation: BP+OSD and SQetch backends.

Both wrappers reuse the quantum CSS distance estimators with ``Hz = empty``
so ``rowspan(Hz) = {0}`` and the estimator's ``dz`` direction becomes the
classical minimum distance of ``ker(H)``.

Augmentation basis ``Lx``: we use the **standard basis ``I_n``** rather
than the auto-derived ``find_logical_noncanonical_RREF`` output. The
auto-derived basis lives in ``F₂ⁿ / rowspan(H)`` — it would miss
codewords that happen to lie in ``ker(H) ∩ rowspan(H)``. Identity rows
cover every nonzero v ∈ ker(H) (every nonzero vector has at least one
nonzero coordinate, so some ``e_i · v = 1``).

The ``dx`` direction is suppressed by passing ``Lz = empty``.

Two backends — pick by speed / hardware:

- :func:`estimate_classical_distance_bposd` — BP+OSD on CPU, parallel
  processes. Fast for small-to-medium ``n``, no GPU required.
- :func:`estimate_classical_distance_sqetch` — GPU random-ISD (QDistRnd-style) via
  the ``sqetch`` package. Faster for large ``n`` if a CUDA device is
  available.
"""

from typing import Iterable, Optional, Union

import numpy as np

from core.dist.quantum_bposd import estimate_quantum_distances_bposd
from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch


def _empty_setup(H: np.ndarray) -> tuple:
    """Build the ``(Hz_empty, Lx=I_n, Lz_empty)`` triple for classical use."""
    H = np.asarray(H, dtype=np.uint8)
    if H.ndim != 2:
        raise ValueError(f"H must be 2D; got shape {H.shape}.")
    n = H.shape[1]
    Hz_empty = np.zeros((0, n), dtype=np.uint8)
    Lx = np.eye(n, dtype=np.uint8)
    Lz_empty = np.zeros((0, n), dtype=np.uint8)
    return H, Hz_empty, Lx, Lz_empty


def estimate_classical_distance_bposd(
    H: np.ndarray,
    *,
    num_trials: int,
    n_workers: int,
    d_target: Optional[int] = None,
    osd_order: int = 5,
) -> Optional[int]:
    """Classical distance via BP+OSD (CPU, multi-process).

    Args:
        H: binary parity-check matrix, shape ``(m, n)``.
        num_trials: BP+OSD decode attempts, split across ``n_workers``.
        n_workers: parallel workers (>= 1).
        d_target: strict-``<`` early-stop threshold. ``None`` runs all trials.
        osd_order: OSD post-processing depth. Use ``0`` for very small ``n``
            (the LDPC library can crash on small matrices with high osd_order).

    Returns:
        Upper-bound estimate of ``d(ker(H))``, or ``None`` if no codeword
        was found in the budget — including the trivial case ``ker(H) = 0``
        (no nonzero codewords exist, so there is no distance to bound; the
        BP+OSD backend must not be called on that degenerate system).
    """
    H, Hz_empty, Lx, Lz_empty = _empty_setup(H)
    from core.f2 import f2_rank
    if f2_rank(H) == H.shape[1]:
        return None
    _dx, dz = estimate_quantum_distances_bposd(
        H, Hz_empty, Lx=Lx, Lz=Lz_empty,
        num_trials=num_trials, n_workers=n_workers,
        d_target=d_target, osd_order=osd_order,
    )
    return dz


def estimate_classical_distance_sqetch(
    H: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    return_codeword: bool = False,
    devices: Optional[Iterable[int]] = None,
    strategy: str = "auto",
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: Optional[int] = None,
) -> Union[Optional[int], tuple]:
    """Classical distance via SQetch (GPU random-ISD, QDistRnd-style).

    Args:
        H: binary parity-check matrix, shape ``(m, n)``.
        num_trials: GAP trials forwarded to sqetch.
        d_target: strict-``<`` early-stop threshold. ``None`` runs all trials.
        return_codeword: if ``True``, also recover the witnessing codeword
            via sqetch's recovery kernel. Slightly heavier kernel, same
            single-shot algorithm.
        devices, strategy, k_sub, batch_size, seed: forwarded to
            :func:`estimate_quantum_distances_sqetch`. See its docstring
            for semantics. ``strategy`` only matters when more than one
            CUDA device is visible.

    Returns:
        ``return_codeword=False`` (default): ``Optional[int]`` —
        upper-bound estimate of ``d(ker(H))``, or ``None`` if nothing was
        found.

        ``return_codeword=True``: ``(dist, codeword)`` where ``codeword``
        is a length-``n`` ``uint8`` array (or ``None``).

    Notes:
        Reuses the ``dz`` direction of the quantum sqetch estimator with
        empty Hz / empty Lz. The ``dx`` direction is unused (it would be
        meaningless for classical) but is still computed cheaply; we
        discard it. If a future sqetch update allows skipping a direction
        entirely, we can wire that in here.
    """
    H, Hz_empty, Lx, Lz_empty = _empty_setup(H)
    out = estimate_quantum_distances_sqetch(
        H, Hz_empty, Lx=Lx, Lz=Lz_empty,
        num_trials=num_trials, d_target=d_target,
        return_logical=return_codeword,
        devices=devices, strategy=strategy,
        k_sub=k_sub, batch_size=batch_size, seed=seed,
    )
    if return_codeword:
        _dx, dz, _dx_cw, dz_cw = out
        return dz, dz_cw
    else:
        _dx, dz = out
        return dz


# Backward-compat alias. The classical phase and tests use this name;
# new code should prefer ``estimate_classical_distance_bposd``.
estimate_classical_distance = estimate_classical_distance_bposd
