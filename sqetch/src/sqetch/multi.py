"""Multi-GPU dispatch for sqetch.estimate_distance.

Shards ``num_trials`` across the visible CUDA devices (or an explicit
subset), runs each shard concurrently in its own thread, and reduces by
``min(best_weight)`` over shards that found any non-trivial logical.
Single-direction, matching :func:`sqetch.estimate_distance`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

import numpy as np

from .api import DistanceResult, estimate_distance


# Compute-capability -> max opt-in dynamic shared memory per block (bytes).
# Reporting only; the per-launch validation lives in api.py.
_CC_SHMEM = {
    (6, 0): 64 * 1024,
    (6, 1): 48 * 1024,
    (7, 0): 96 * 1024,
    (7, 2): 96 * 1024,
    (7, 5): 64 * 1024,
    (8, 0): 163 * 1024,
    (8, 6): 99 * 1024,
    (8, 7): 163 * 1024,
    (8, 9): 99 * 1024,
    (9, 0): 227 * 1024,
    (10, 0): 227 * 1024,
    (12, 0): 99 * 1024,
}


def detect_gpus(devices: Optional[Iterable[int]] = None) -> list[dict]:
    """Return a list of dicts describing each candidate GPU.

    Args:
        devices: explicit iterable of device indices, or ``None`` for all
            visible CUDA devices.

    Each dict has: idx, name, cc (tuple), cc_str ("sm_XX"), total_mem_gb,
    free_mem_gb, shmem_limit (bytes).  Empty list if no CUDA devices.
    """
    import torch
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    if n == 0:
        return []

    indices = list(range(n)) if devices is None else [int(i) for i in devices]
    out: list[dict] = []
    for i in indices:
        if i < 0 or i >= n:
            continue
        props = torch.cuda.get_device_properties(i)
        cc = (props.major, props.minor)
        try:
            free, _total = torch.cuda.mem_get_info(i)
            free_gb = free / 1024**3
        except Exception:
            free_gb = float("nan")
        out.append({
            "idx": int(i),
            "name": str(props.name),
            "cc": cc,
            "cc_str": f"sm_{cc[0]}{cc[1]}",
            "total_mem_gb": props.total_memory / 1024**3,
            "free_mem_gb": free_gb,
            "shmem_limit": _CC_SHMEM.get(cc, 48 * 1024),
        })
    return out


def _run_one_shard(
    H_check, L_logical,
    num_trials, d_target, seed, k_sub, batch_size, device_idx,
) -> DistanceResult:
    """Pin the calling thread to a device and run one shard."""
    import torch
    torch.cuda.set_device(device_idx)
    return estimate_distance(
        H_check, L_logical,
        num_trials=num_trials,
        d_target=d_target,
        seed=seed,
        k_sub=k_sub,
        batch_size=batch_size,
        device=device_idx,
    )


def estimate_distance_multi(
    H_check: np.ndarray,
    L_logical: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    devices: Optional[Iterable[int]] = None,
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: Optional[int] = None,
) -> DistanceResult:
    """Single-direction multi-GPU distance estimation (trial_split).

    Shards ``num_trials`` evenly across ``devices`` (or all visible GPUs),
    one thread per GPU, and reduces by ``min`` over the per-shard
    ``best_weight`` (ignoring shards that found no logical).

    Args:
        H_check, L_logical: as in :func:`sqetch.estimate_distance`.
        num_trials: total trials across all devices.
        d_target: early-stop threshold, applied per shard.
        devices: optional iterable of device indices; ``None`` uses all.
        k_sub, batch_size, seed: forwarded to :func:`estimate_distance`;
            per-shard seeds derive from ``seed``.

    Returns:
        A :class:`DistanceResult` aggregated across shards.

    Raises:
        RuntimeError: if no CUDA devices are visible.
    """
    gpus = detect_gpus(devices)
    if not gpus:
        raise RuntimeError(
            "No CUDA devices detected for sqetch.estimate_distance_multi. "
            "Install a CUDA build of torch and ensure a GPU is visible."
        )
    n_gpus = len(gpus)

    if seed is None:
        seed = int(time.time() * 1e6) & ((1 << 63) - 1)

    base = num_trials // n_gpus
    extra = num_trials - base * n_gpus

    t0 = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        for i, g in enumerate(gpus):
            t = base + (1 if i < extra else 0)
            if t == 0:
                continue
            shard_seed = seed + i * 1_000_003
            futures.append(ex.submit(
                _run_one_shard,
                H_check, L_logical,
                t, d_target, shard_seed,
                k_sub, batch_size, g["idx"],
            ))
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    if not results:
        return DistanceResult(
            best_weight=None,
            found=False,
            trials_run=0,
            elapsed_seconds=0.0,
            raw_iter_per_sec=0.0,
            n=int(H_check.shape[1]),
            k_null=0,
            k_sub=0,
        )

    non_none = [r.best_weight for r in results if r.best_weight is not None]
    best_weight = min(non_none) if non_none else None
    found = any(r.found for r in results)
    trials_run = sum(r.trials_run for r in results)
    raw_iter_per_sec = trials_run / elapsed if elapsed > 0 else 0.0

    return DistanceResult(
        best_weight=best_weight if best_weight is None else int(best_weight),
        found=bool(found),
        trials_run=int(trials_run),
        elapsed_seconds=float(elapsed),
        raw_iter_per_sec=float(raw_iter_per_sec),
        n=int(H_check.shape[1]),
        k_null=int(results[0].k_null),
        k_sub=int(results[0].k_sub),
    )
