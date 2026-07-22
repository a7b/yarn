"""GPU random-ISD distance estimation for CSS codes.

:func:`estimate_distance` runs a batched GPU loop of randomized
information-set decoder trials against ``ker(H_check)`` and returns the
smallest logical-coset weight observed.  Call it once per CSS direction:
``(H_X, L_X)`` for d_Z and ``(H_Z, L_Z)`` for d_X.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ._arch import arch_flag, build_dir_suffix, max_shmem_per_block_optin
from ._gf2 import gf2_null_space, pack_gf2_rows_gpu

_KERNEL_DIR = Path(__file__).parent / "kernels"
_KERNEL_PATH = _KERNEL_DIR / "kernel.cu"

_MODULE = None
_BUILD_ERROR: Optional[str] = None


@dataclass(frozen=True)
class DistanceResult:
    """Result of a single :func:`estimate_distance` call.

    Attributes:
        best_weight: Smallest logical-coset weight observed across all
            trials, or ``None`` if no non-trivial logical-coset row was
            ever found in the budget.
        found: True iff a candidate lighter than ``d_target`` was seen.
            Always False when ``d_target is None``.
        trials_run: Number of GPU trials executed (may be less than
            ``num_trials`` if early stop fired).
        elapsed_seconds: Wall-clock time of the GPU loop, excluding the
            CPU null-space computation.
        raw_iter_per_sec: ``trials_run / elapsed_seconds``.
        n: Number of physical qubits.
        k_null: Rank of the null space of ``H_check``.
        k_sub: Sketch dimension actually used.
    """

    best_weight: Optional[int]
    found: bool
    trials_run: int
    elapsed_seconds: float
    raw_iter_per_sec: float
    n: int
    k_null: int
    k_sub: int


def _get_module():
    """JIT-compile the CUDA kernel for the visible GPU (cached after first call)."""
    global _MODULE, _BUILD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _BUILD_ERROR is not None:
        raise RuntimeError(f"sqetch kernel compilation previously failed: {_BUILD_ERROR}")

    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError as e:
        raise RuntimeError(
            "sqetch requires PyTorch with CUDA extension support; "
            "install via `pip install sqetch[gpu]` or supply torch manually."
        ) from e

    cuda_src = _KERNEL_PATH.read_text()
    cpp_src = "// sqetch kernel binding\n"

    build_dir = os.path.join("/tmp", "sqetch_ext" + build_dir_suffix())
    os.makedirs(build_dir, exist_ok=True)

    try:
        _MODULE = load_inline(
            name="sqetch_ext",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            extra_cuda_cflags=["-O3", arch_flag(), "--use_fast_math"],
            build_directory=build_dir,
            verbose=False,
        )
        return _MODULE
    except Exception as e:
        _BUILD_ERROR = str(e)
        raise RuntimeError(f"sqetch kernel compilation failed: {e}") from e


def _require_cuda() -> None:
    """Raise a clear error when sqetch is called without a CUDA-enabled torch."""
    import torch
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "sqetch's kernel needs a CUDA-enabled PyTorch build with a "
        "visible GPU.  torch.cuda.is_available() is False -- either "
        "your `torch` is the CPU-only wheel (default on macOS and ARM "
        "Linux from PyPI), or no CUDA device is visible.  See the "
        "Install / Platform notes in sqetch's README."
    )


def _validate_inputs(H_check: np.ndarray, L_logical: np.ndarray) -> None:
    if H_check.ndim != 2 or L_logical.ndim != 2:
        raise ValueError("H_check and L_logical must be 2D arrays.")
    if H_check.shape[1] != L_logical.shape[1]:
        raise ValueError(
            f"H_check and L_logical must have the same number of columns "
            f"(got {H_check.shape[1]} vs {L_logical.shape[1]})."
        )
    if H_check.dtype != np.uint8 or L_logical.dtype != np.uint8:
        raise ValueError("H_check and L_logical must be uint8 arrays.")
    if not np.array_equal(H_check, H_check & 1) or not np.array_equal(L_logical, L_logical & 1):
        raise ValueError("H_check and L_logical entries must be in {0, 1}.")


def estimate_distance(
    H_check: np.ndarray,
    L_logical: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: Optional[int] = None,
    device: int = 0,
) -> DistanceResult:
    """Estimate the minimum logical-coset weight of a CSS code on the GPU.

    Computes ``W = ker(H_check)``, then for each trial draws a random
    column permutation and a ``k_sub``-dimensional sketch ``W_sub`` of
    ``W``, row-reduces ``W_sub`` in the permuted order, and for each
    non-trivial row checks whether its inner product with any row of
    ``L_logical`` is non-zero, tracking the minimum Hamming weight seen.

    Args:
        H_check: ``(m, n)`` uint8 parity-check matrix (``H_X`` or ``H_Z``).
        L_logical: ``(k, n)`` uint8 logical basis rows; must lie in
            ``ker(H_check)``.  Pass ``L_X`` to estimate ``d_Z``, ``L_Z``
            for ``d_X``.
        num_trials: Total number of random-ISD trials to run.
        d_target: Early-stop threshold.  A trial finding a codeword
            lighter than ``d_target`` stops the host loop at the next
            batch boundary.  ``None`` disables early stopping.
        k_sub: Sketch dimension, bounded above by ``rank(ker(H_check))``
            and by the device's opt-in shared-memory ceiling.
        batch_size: GPU trials per kernel launch.
        seed: RNG seed.  ``None`` uses ``int(time.time() * 1e6)``.
        device: CUDA device ordinal.

    Returns:
        :class:`DistanceResult`.
    """
    _validate_inputs(H_check, L_logical)

    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if k_sub <= 0:
        raise ValueError("k_sub must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if seed is None:
        seed = int(time.time() * 1e6) & ((1 << 63) - 1)

    _require_cuda()
    import torch
    torch.cuda.set_device(device)
    torch_device = torch.device(f"cuda:{device}")

    W_null = gf2_null_space(H_check)
    W_logical = L_logical

    k_null, n = W_null.shape
    nw = math.ceil(n / 64)

    if k_null == 0:
        return DistanceResult(
            best_weight=None,
            found=False,
            trials_run=0,
            elapsed_seconds=0.0,
            raw_iter_per_sec=0.0,
            n=n,
            k_null=0,
            k_sub=0,
        )

    k_sub_eff = min(k_sub, k_null)

    perm_bytes = (n * 2 + 7) & ~7
    wsub_bytes = k_sub_eff * nw * 8
    pivot_bytes = nw * 8
    total_shmem = perm_bytes + wsub_bytes + pivot_bytes + 8 + 128 * 4
    max_shmem = max_shmem_per_block_optin()
    if total_shmem > max_shmem:
        k_sub_cap = (max_shmem - perm_bytes - pivot_bytes - 8 - 512) // (nw * 8)
        raise ValueError(
            f"k_sub={k_sub_eff} requires {total_shmem} bytes shared memory "
            f"({total_shmem / 1024:.1f} KiB), exceeds device limit "
            f"{max_shmem / 1024:.0f} KiB. Max k_sub for nw={nw}: {k_sub_cap}."
        )

    mod = _get_module()
    W_null_gpu = pack_gf2_rows_gpu(W_null, torch_device)
    W_logical_gpu = pack_gf2_rows_gpu(W_logical, torch_device)

    best = n + 1
    trials_done = 0
    found_any = False
    d_target_int = d_target if d_target is not None else -1

    t0 = time.perf_counter()
    while trials_done < num_trials:
        B = min(batch_size, num_trials - trials_done)
        batch_seed = seed + trials_done
        result = mod.run_sqetch_ksub(
            W_null_gpu, W_logical_gpu, n, k_sub_eff, B, batch_seed, d_target_int
        )
        torch.cuda.synchronize()

        batch_best = int(result[0].item())
        found = int(result[1].item())

        if batch_best < best:
            best = batch_best
        trials_done += B

        if found:
            found_any = True
        if found or (d_target is not None and best < d_target):
            break
    elapsed = time.perf_counter() - t0

    best_weight: Optional[int] = int(best) if best <= n else None

    raw_iter_per_sec = trials_done / elapsed if elapsed > 0 else 0.0

    return DistanceResult(
        best_weight=best_weight,
        found=bool(found_any),
        trials_run=int(trials_done),
        elapsed_seconds=float(elapsed),
        raw_iter_per_sec=float(raw_iter_per_sec),
        n=int(n),
        k_null=int(k_null),
        k_sub=int(k_sub_eff),
    )
