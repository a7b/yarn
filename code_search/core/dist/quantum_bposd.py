"""BP+OSD quantum distance estimation.

Estimates upper bounds on the X- and Z-distances of a CSS code via the
random-codeword trick with `BpOsdDecoder` as the minimum-weight solver. No
GAP subprocess overhead — suitable as a first-pass filter before higher-
confidence random-ISD (QDistRnd-style) verification.

Provides:
    estimate_quantum_distances_bposd(Hx, Hz, Lx, Lz, *, num_trials,
                                     n_workers, d_target=None, osd_order=5)
        -> (dx, dz)

Conventions:
- **Strict `<` early-stop.** A codeword of weight strictly less than
  `d_target` triggers all workers in that direction to halt. A codeword of
  weight == d_target does NOT trigger stop — the boundary case is a PASS
  by the downstream search-pipeline convention.
- **`int | None` per direction.** `None` means no codeword found in
  `num_trials` — callers should treat as PASS (no refutation of
  `d >= d_target`).
- **Cross-type augmentation** (required for correctness): for dx (X-logical
  weight), augment Hz with Lz. For dz, augment Hx with Lx. Same-type
  augmentation finds light stabilizers and falsely reports them as
  logicals.
- **Lx, Lz are required inputs.** Caller provides the logical basis;
  typically from `logical_basis.find_logical_basis`. The estimator does
  not compute its own logicals (decoupled from `bposd.css.css_code`).
"""

import multiprocessing
import types
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np


def _bposd_chunk(args):
    """Worker: run a chunk of BP+OSD trials in one direction.

    Returns the local minimum weight observed in this chunk, or the sentinel
    `large` if nothing was found.

    For each trial:
      1. Sample a random nonzero binary combination of `logicals`.
      2. Augment `H_check` with that combination as an extra row; target
         syndrome = [0,…,0,1] (only the augmented row is unsatisfied).
      3. Decode with BP+OSD. The result is a low-weight representative of
         the corresponding logical coset.
      4. Track local + shared minimum; honor strict-`<` early stop.
    """
    from ldpc import BpOsdDecoder

    (H_check, logicals, osd_order, chunk_trials, d_target,
     shared_stop, shared_best, large) = args

    n_log = logicals.shape[0]
    rng = np.random.default_rng()   # OS-seeded; independent per worker
    local_min = large
    _CHECK_INTERVAL = 50

    for i in range(chunk_trials):
        if i % _CHECK_INTERVAL == 0:
            if d_target is not None and shared_stop.value:
                break

        coeffs = rng.integers(0, 2, size=n_log, dtype=np.uint8)
        if not coeffs.any():
            continue
        logical_err = (coeffs @ logicals) % 2
        if not logical_err.any():
            # Combo lies in rowspan(H_check); not a logical. Skip.
            continue

        combined = np.vstack([H_check, logical_err[None, :]]).astype(np.uint8)
        syndrome = np.zeros(combined.shape[0], dtype=np.uint8)
        syndrome[-1] = 1

        decoder = BpOsdDecoder(
            combined,
            error_rate=0.05,
            max_iter=50,
            bp_method="min_sum",
            ms_scaling_factor=0.625,
            osd_method="osd_cs",
            osd_order=osd_order,
        )
        corr = decoder.decode(syndrome)

        # Validity: correction must satisfy the augmented syndrome.
        if not np.array_equal(
            (combined.astype(int) @ corr.astype(int)) % 2,
            syndrome.astype(int),
        ):
            continue

        w = int(np.sum(corr))
        if w == 0:
            continue

        if w < local_min:
            local_min = w
            if w < shared_best.value:
                shared_best.value = w

        # Strict `<` early stop: weight equal to d_target does NOT trigger
        # halt (matches QDistRnd's `mindist` semantics; boundary case = PASS).
        if d_target is not None and w < d_target:
            shared_stop.value = True
            break

    return int(local_min)


def estimate_quantum_distances_bposd(
    Hx: np.ndarray,
    Hz: np.ndarray,
    Lx: Optional[np.ndarray] = None,
    Lz: Optional[np.ndarray] = None,
    *,
    num_trials: int,
    n_workers: int,
    d_target: Optional[int] = None,
    osd_order: int = 5,
) -> tuple[Optional[int], Optional[int]]:
    """Estimate X and Z distances of a CSS code via BP+OSD.

    Args:
        Hx, Hz: CSS parity-check matrices, shape (m_x, n_phys), (m_z, n_phys),
            dtype uint8.
        Lx: X-logical basis, shape (k, n_phys). Rows in ker(Hz) \\ rowspan(Hx).
            If ``None`` (default), an RREF-based basis is auto-computed via
            :func:`logical_basis.find_logical_noncanonical_RREF` — any basis works
            for BP+OSD (weight doesn't matter for cross-type augmentation).
        Lz: Z-logical basis, shape (k, n_phys). Rows in ker(Hx) \\ rowspan(Hz).
            Same auto-compute semantics as ``Lx``. If exactly one of
            ``Lx``/``Lz`` is None, **both** are recomputed (the pairing
            ``Lx · Lz.T = I_k`` requires them to be derived together).
        num_trials: total decode attempts per direction (split across
            `n_workers`). Each worker runs `max(1, num_trials // n_workers)`.
        n_workers: parallel ProcessPoolExecutor workers. Required.
        d_target: strict-`<` early-stop threshold. If a codeword of weight
            < d_target is found, all workers in that direction halt.
            `None` (default) runs all trials.
        osd_order: OSD post-processing depth for BpOsdDecoder. Higher gives
            better coverage at O(n^3 · 2^order) per trial. Common values:
            5 (default, balanced) or 10 (more thorough, slower).

    Returns:
        (dx, dz). Each is an int (upper-bound weight estimate) or None
        (no codeword found in `num_trials`).

    Raises:
        ValueError if column counts disagree, or `num_trials < 1` or
        `n_workers < 1`.
    """
    Hx = np.asarray(Hx, dtype=np.uint8)
    Hz = np.asarray(Hz, dtype=np.uint8)

    # Auto-derive logicals when not provided. Always derive both at once so
    # the basis is internally consistent — Lx and Lz independently derived
    # would not be paired.
    if Lx is None or Lz is None:
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)
    Lx = np.asarray(Lx, dtype=np.uint8)
    Lz = np.asarray(Lz, dtype=np.uint8)

    n_phys = Hx.shape[1]
    if Hz.shape[1] != n_phys:
        raise ValueError(
            f"Hx and Hz column counts disagree: "
            f"{Hx.shape[1]} vs {Hz.shape[1]}"
        )
    if Lx.ndim == 1:
        Lx = Lx[None, :]
    if Lz.ndim == 1:
        Lz = Lz[None, :]
    if Lx.size and Lx.shape[1] != n_phys:
        raise ValueError(f"Lx has {Lx.shape[1]} cols; expected {n_phys}")
    if Lz.size and Lz.shape[1] != n_phys:
        raise ValueError(f"Lz has {Lz.shape[1]} cols; expected {n_phys}")
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1; got {num_trials}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1; got {n_workers}")

    large = n_phys + 1
    trials_per_worker = max(1, num_trials // n_workers)

    def _run(H_check: np.ndarray, logicals: np.ndarray) -> Optional[int]:
        if logicals.shape[0] == 0:
            return None
        if n_workers == 1:
            # Inline fast path: a Manager + ProcessPoolExecutor costs seconds
            # of wall time per call, which dominates small instances. The
            # chunk worker only needs `.value` on the two shared holders.
            stop = types.SimpleNamespace(value=False)
            best_holder = types.SimpleNamespace(value=large)
            best = _bposd_chunk(
                (H_check, logicals, osd_order, trials_per_worker,
                 d_target, stop, best_holder, large)
            )
            return None if best >= large else int(best)
        with multiprocessing.Manager() as manager:
            shared_stop = manager.Value("b", False)
            shared_best = manager.Value("i", large)
            tasks = [
                (H_check, logicals, osd_order, trials_per_worker,
                 d_target, shared_stop, shared_best, large)
                for _ in range(n_workers)
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_bposd_chunk, tasks))
        best = min(results) if results else large
        return None if best >= large else int(best)

    # dx: X-logicals live in ker(Hz) \ rowspan(Hx).
    # Augment Hz with Lz (lz ∈ ker(Hx) ⇒ lz · corr = 1 ⇒ corr ∉ rowspan(Hx)).
    dx = _run(Hz, Lz)
    # dz: Z-logicals live in ker(Hx) \ rowspan(Hz). Augment Hx with Lx.
    dz = _run(Hx, Lx)
    return dx, dz
