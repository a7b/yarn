"""SQetch-based CSS distance estimator — two-direction wrapper.

Mirrors the signature pattern of
:func:`core.dist.quantum_bposd.estimate_quantum_distances_bposd`:

    estimate_quantum_distances_sqetch(Hx, Hz, Lx, Lz, *,
        num_trials, d_target=None, return_logical=False,
        devices=None, strategy="auto",
        k_sub=64, batch_size=50_000, seed=None,
    ) -> tuple

Default returns ``(dx, dz)``. With ``return_logical=True`` returns a
4-tuple ``(dx, dz, dx_codeword, dz_codeword)`` where each codeword is a
length-``n_phys`` uint8 array or ``None``.

Cross-type augmentation: dx → ``(Hz, Lz)``; dz → ``(Hx, Lx)``.

Strict-``<`` early-stop. ``None`` per direction when nothing was found.

When ``return_logical=False``: calls sqetch's *fast* kernel
(:func:`sqetch.estimate_distance` / :func:`sqetch.estimate_distance_multi`)
which only tracks the minimum weight via atomicMin.

When ``return_logical=True``: calls sqetch's *recovery* kernel
(:func:`sqetch.recover_codeword`) which is the same algorithm but ALSO
writes the winning RREF row to GPU global memory on each new minimum.
One kernel — no two-phase "estimate then re-run with target_weight"
roundtrip. The recovery kernel pays slightly more shared memory and one
cooperative global-memory write per "new minimum" event.

sqetch is imported lazily so this module loads on machines without CUDA.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

import numpy as np


def _weight_or_none(w) -> Optional[int]:
    """Adapt sqetch's `DistanceResult.best_weight` (Optional[int]) to our
    `int | None` direction-result convention."""
    if w is None:
        return None
    return int(w)


def _direction_estimate_single(H_check, L_logical, *,
                               num_trials, d_target, device,
                               k_sub, batch_size, seed):
    """Light path: just estimate, no codeword recovery. Single GPU."""
    import torch
    import sqetch
    torch.cuda.set_device(device)
    res = sqetch.estimate_distance(
        H_check, L_logical,
        num_trials=num_trials,
        d_target=d_target,
        k_sub=k_sub,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    return _weight_or_none(res.best_weight), None


def _direction_estimate_multi(H_check, L_logical, *,
                              num_trials, d_target, devices,
                              k_sub, batch_size, seed):
    """Light path: just estimate, no codeword recovery. Trial-split across GPUs."""
    import sqetch
    res = sqetch.estimate_distance_multi(
        H_check, L_logical,
        num_trials=num_trials,
        d_target=d_target,
        devices=devices,
        k_sub=k_sub,
        batch_size=batch_size,
        seed=seed,
    )
    return _weight_or_none(res.best_weight), None


def _direction_recover_single(H_check, L_logical, *,
                              num_trials, d_target, device,
                              k_sub, batch_size, seed):
    """Heavy path: recovery kernel returns (weight, codeword). Single GPU."""
    import torch
    import sqetch
    torch.cuda.set_device(device)
    weight, codeword = sqetch.recover_codeword(
        H_check, L_logical,
        num_trials=num_trials,
        d_target=d_target,
        k_sub=k_sub,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    return weight, codeword


def estimate_quantum_distances_sqetch(
    Hx: np.ndarray,
    Hz: np.ndarray,
    Lx: Optional[np.ndarray] = None,
    Lz: Optional[np.ndarray] = None,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    return_logical: bool = False,
    devices: Optional[Iterable[int]] = None,
    strategy: str = "auto",
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: Optional[int] = None,
) -> tuple:
    """SQetch CSS-code distance estimator (two directions, optional multi-GPU).

    Args:
        Hx, Hz: CSS parity-check matrices, uint8.
        Lx: X-logical basis, ``(k, n_phys)``, rows in ker(Hz) \\ rowspan(Hx).
            If ``None`` (default), an RREF-based basis is auto-computed via
            :func:`logical_basis.find_logical_noncanonical_RREF` — any basis
            works for cross-type augmentation; weight doesn't matter.
        Lz: Z-logical basis, ``(k, n_phys)``, rows in ker(Hx) \\ rowspan(Hz).
            Same auto-compute semantics as ``Lx``. If exactly one of
            ``Lx``/``Lz`` is None, **both** are recomputed (the pairing
            ``Lx · Lz.T = I_k`` requires them derived together).
        num_trials: total decode attempts per direction.
        d_target: strict-``<`` early-stop threshold (forwarded to sqetch).
        return_logical: if True, also recover the codeword for each
            direction. Uses sqetch's recovery kernel (heavier, but still a
            single kernel — no two-phase round-trip).
        devices: optional iterable of CUDA device indices. ``None`` = all visible.
        strategy: ``"auto"``, ``"direction_split"``, or ``"trial_split"``.
            For ``return_logical=True``, multi-GPU is limited to
            ``direction_split`` (recovery kernel writes to one global
            buffer; trial_split is single-GPU per direction).
        k_sub, batch_size, seed: forwarded to sqetch.

    Returns:
        ``return_logical=False`` (default): ``(dx, dz)``,
            each ``int | None``. ``None`` means no codeword found.
        ``return_logical=True``: ``(dx, dz, dx_codeword, dz_codeword)``.
            Each codeword is a length-``n_phys`` uint8 array, or ``None``.

    Raises:
        ValueError on shape mismatches, ``num_trials < 1``, or unknown strategy.
        RuntimeError if no CUDA devices are visible.
    """
    Hx = np.asarray(Hx, dtype=np.uint8)
    Hz = np.asarray(Hz, dtype=np.uint8)

    # Auto-derive logicals when not provided. Always derive both at once so
    # the pair is internally consistent (Lx · Lz.T = I_k).
    if Lx is None or Lz is None:
        from logical_basis.logical_basis import find_logical_noncanonical_RREF
        Lx, Lz = find_logical_noncanonical_RREF(Hx, Hz)

    Lx = np.asarray(Lx, dtype=np.uint8)
    Lz = np.asarray(Lz, dtype=np.uint8)
    if Lx.ndim == 1:
        Lx = Lx[None, :]
    if Lz.ndim == 1:
        Lz = Lz[None, :]

    n_phys = Hx.shape[1]
    if Hz.shape[1] != n_phys:
        raise ValueError(
            f"Hx, Hz column counts disagree: {Hx.shape[1]} vs {Hz.shape[1]}"
        )
    if Lx.size and Lx.shape[1] != n_phys:
        raise ValueError(f"Lx has {Lx.shape[1]} cols; expected {n_phys}")
    if Lz.size and Lz.shape[1] != n_phys:
        raise ValueError(f"Lz has {Lz.shape[1]} cols; expected {n_phys}")
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1; got {num_trials}")
    if strategy not in ("auto", "direction_split", "trial_split"):
        raise ValueError(
            f"strategy must be 'auto', 'direction_split', or 'trial_split'; "
            f"got {strategy!r}"
        )

    import sqetch
    gpus = sqetch.detect_gpus(devices)
    n_gpus = len(gpus)
    if n_gpus == 0:
        raise RuntimeError("No CUDA devices detected for sqetch.")

    if strategy == "auto":
        strategy = "direction_split" if n_gpus == 2 else "trial_split"

    # Cross-type: dx augments Hz with Lz, dz augments Hx with Lx.
    # So dx-emptiness hinges on Lz (not Lx) and vice versa.
    dx_empty = Lz.shape[0] == 0
    dz_empty = Lx.shape[0] == 0

    # Resolve seed at the wrapper level so dx and dz get DISTINCT seeds
    # even with seed=None. Otherwise two concurrent direction_split threads
    # could fall through to sqetch's per-call `int(time.time() * 1e6)` and
    # collide on the same microsecond.
    if seed is None:
        seed = int(time.time() * 1e6) & ((1 << 63) - 1)
    seed_dx = seed
    seed_dz = seed + 1_000_003

    # Pick the runner per (return_logical, strategy, n_gpus).
    # Recovery kernel does not have a multi-GPU mode — under
    # return_logical=True we keep each direction on a single GPU.
    if return_logical:
        per_direction = _direction_recover_single
        use_threads = (strategy == "direction_split" and n_gpus >= 2)
    else:
        if strategy == "trial_split" and n_gpus > 1:
            per_direction = _direction_estimate_multi
            use_threads = False
        else:
            per_direction = _direction_estimate_single
            use_threads = (strategy == "direction_split" and n_gpus >= 2)

    if return_logical or per_direction is _direction_estimate_single:
        if use_threads:
            dev_dx = gpus[0]["idx"]
            dev_dz = gpus[1]["idx"]
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_dx = (None if dx_empty else
                        ex.submit(per_direction, Hz, Lz,
                                  num_trials=num_trials, d_target=d_target,
                                  device=dev_dx, k_sub=k_sub,
                                  batch_size=batch_size, seed=seed_dx))
                f_dz = (None if dz_empty else
                        ex.submit(per_direction, Hx, Lx,
                                  num_trials=num_trials, d_target=d_target,
                                  device=dev_dz, k_sub=k_sub,
                                  batch_size=batch_size, seed=seed_dz))
                dx_res = f_dx.result() if f_dx else (None, None)
                dz_res = f_dz.result() if f_dz else (None, None)
        else:
            dev = gpus[0]["idx"]
            dx_res = ((None, None) if dx_empty else
                      per_direction(Hz, Lz,
                                    num_trials=num_trials, d_target=d_target,
                                    device=dev, k_sub=k_sub,
                                    batch_size=batch_size, seed=seed_dx))
            dz_res = ((None, None) if dz_empty else
                      per_direction(Hx, Lx,
                                    num_trials=num_trials, d_target=d_target,
                                    device=dev, k_sub=k_sub,
                                    batch_size=batch_size, seed=seed_dz))
    else:
        # trial_split, return_logical=False.
        dev_list = [g["idx"] for g in gpus]
        dx_res = ((None, None) if dx_empty else
                  _direction_estimate_multi(Hz, Lz,
                                            num_trials=num_trials, d_target=d_target,
                                            devices=dev_list, k_sub=k_sub,
                                            batch_size=batch_size, seed=seed_dx))
        dz_res = ((None, None) if dz_empty else
                  _direction_estimate_multi(Hx, Lx,
                                            num_trials=num_trials, d_target=d_target,
                                            devices=dev_list, k_sub=k_sub,
                                            batch_size=batch_size, seed=seed_dz))

    dx, dx_vec = dx_res
    dz, dz_vec = dz_res

    if return_logical:
        return dx, dz, dx_vec, dz_vec
    return dx, dz


def sample_low_weight_logicals_sqetch(
    H_check: np.ndarray,
    L_logical: np.ndarray,
    *,
    num_trials: int,
    target_weight: int,
    max_results: int = 10_000,
    k_sub: int = 64,
    batch_size: int = 5_000,
    seed: Optional[int] = None,
    devices: Optional[Iterable[int]] = None,
    dedup: bool = True,
) -> np.ndarray:
    """Tailored sampling: every logical codeword of weight in ``(0, target_weight]``.

    Single-**direction** wrapper over :func:`sqetch.sample_low_weight_logicals`
    (the DistRandTailored analog).  This is the GPU replacement for the legacy
    ``gpu_gap_bridge.run_tailored_via_gpu`` used by adaptive-thickening surgery,
    whose inner loop needs to *see* which low-weight logicals exist in the
    current merged code so it can add ancilla columns that kill them.

    Surgery mapping (matching QDistRnd ``DistRandTailored(GX=Hz, WX=Lz)``):
    pass ``H_check = Hz`` (the merged Z-checks) and ``L_logical = Lz`` (the
    Z-logical witnesses).  Every returned row lies in ``ker(H_check)`` and has
    odd overlap with at least one ``L_logical`` row.

    Args:
        H_check: ``(m, n)`` uint8 parity-check matrix whose null space is sampled.
        L_logical: ``(k, n)`` uint8 logical-witness rows (the filter).
        num_trials: total random-ISD trials (split across ``devices`` if >1).
        target_weight: inclusive weight cap (``0 < wt <= target_weight``).
        max_results: per-launch (per-device) output-buffer capacity.
        k_sub, batch_size, seed: forwarded to sqetch.
        devices: CUDA device indices.  ``None`` → single GPU (device 0).
            A list of length >1 trial-splits ``num_trials`` across the devices
            and concatenates results (useful for the heavy initial-gadget call).
        dedup: return distinct rows only (deduped after merging shards).

    Returns:
        ``(n_found, n)`` uint8 array of codewords, or ``(0, n)`` if none.

    Raises:
        ValueError on shape mismatch / bad parameters; RuntimeError if no CUDA.
    """
    import sqetch

    H_check = np.asarray(H_check, dtype=np.uint8)
    L_logical = np.asarray(L_logical, dtype=np.uint8)
    if L_logical.ndim == 1:
        L_logical = L_logical[None, :]

    dev_list = None if devices is None else [int(d) for d in devices]

    # Single GPU (the common per-round case): one direct call.
    if dev_list is None or len(dev_list) <= 1:
        device = 0 if dev_list is None else dev_list[0]
        return sqetch.sample_low_weight_logicals(
            H_check, L_logical,
            num_trials=num_trials, target_weight=target_weight,
            max_results=max_results, k_sub=k_sub, batch_size=batch_size,
            seed=seed, device=device, dedup=dedup,
        )

    # Multi-GPU trial-split: shard num_trials, distinct seed per shard, merge.
    if seed is None:
        seed = int(time.time() * 1e6) & ((1 << 63) - 1)
    n_dev = len(dev_list)
    base = num_trials // n_dev
    shares = [base + (1 if i < num_trials % n_dev else 0) for i in range(n_dev)]

    def _shard(i):
        if shares[i] < 1:
            return None
        import torch
        torch.cuda.set_device(dev_list[i])
        return sqetch.sample_low_weight_logicals(
            H_check, L_logical,
            num_trials=shares[i], target_weight=target_weight,
            max_results=max_results, k_sub=k_sub, batch_size=batch_size,
            seed=seed + i * 1_000_003, device=dev_list[i], dedup=False,
        )

    with ThreadPoolExecutor(max_workers=n_dev) as ex:
        parts = [p for p in ex.map(_shard, range(n_dev)) if p is not None and p.shape[0]]

    if not parts:
        return np.zeros((0, H_check.shape[1]), dtype=np.uint8)
    out = np.vstack(parts).astype(np.uint8)
    if dedup and out.shape[0] > 1:
        out = np.unique(out, axis=0)
    return out
