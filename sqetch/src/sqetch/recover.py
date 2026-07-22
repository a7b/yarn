"""Codeword recovery: like estimate_distance but also returns the bit vector.

:func:`recover_codeword` mirrors :func:`sqetch.estimate_distance`'s
signature and early-stop semantics, but also reconstructs and returns the
minimum-weight codeword.  It is slower than :func:`estimate_distance`; use
it only when you need the bit pattern.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .api import _get_module, _require_cuda
from ._gf2 import gf2_null_space, pack_gf2_rows_gpu


def _unpack_row(packed_row: np.ndarray, n: int) -> np.ndarray:
    """Unpack a length-``nw`` uint64 packed row into a length-``n`` uint8 vector."""
    out = np.zeros(n, dtype=np.uint8)
    words = np.asarray(packed_row, dtype=np.uint64)
    for i in range(n):
        w = i >> 6
        b = i & 63
        out[i] = (int(words[w]) >> b) & 1
    return out


def _packed_to_codeword(packed: np.ndarray, perm: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct the codeword from the kernel's packed output.

    The kernel's output row is already in physical column order, so the
    permutation is unused here; we just unpack the bit-packed words.
    """
    del perm
    return _unpack_row(packed, n)


def recover_codeword(
    H_check: np.ndarray,
    L_logical: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: Optional[int] = None,
    device: int = 0,
) -> tuple[Optional[int], Optional[np.ndarray]]:
    """Find a low-weight logical codeword and return its weight and the vector.

    Each batch runs ``batch_size`` trials; the first block to find a
    codeword lighter than the current target claims the output slot and
    writes its row.  The host tightens the target across batches until
    ``num_trials`` is exhausted, or a codeword lighter than ``d_target``
    is found.

    Args:
        H_check, L_logical: as in :func:`sqetch.estimate_distance`.
        num_trials: total kernel trials across all batches.
        d_target: early-stop threshold; ``None`` disables it.
        k_sub, batch_size, seed, device: as in :func:`estimate_distance`.

    Returns:
        ``(best_weight, codeword)``, where ``codeword`` is a length-``n``
        uint8 array, or ``(None, None)`` if nothing was committed.

    Raises:
        ValueError on shape mismatch, ``num_trials < 1``, or ``k_sub < 1``.
    """
    if H_check.ndim != 2 or L_logical.ndim != 2:
        raise ValueError("H_check and L_logical must be 2D arrays.")
    if H_check.shape[1] != L_logical.shape[1]:
        raise ValueError(
            f"H_check and L_logical column counts disagree: "
            f"{H_check.shape[1]} vs {L_logical.shape[1]}"
        )
    if num_trials < 1:
        raise ValueError("num_trials must be positive.")
    if k_sub < 1:
        raise ValueError("k_sub must be positive.")

    if seed is None:
        seed = int(time.time() * 1e6) & ((1 << 63) - 1)

    _require_cuda()
    import torch
    torch.cuda.set_device(device)
    torch_device = torch.device(f"cuda:{device}")

    W_null = gf2_null_space(H_check)
    k_null, n = W_null.shape
    if k_null == 0:
        return None, None

    k_sub_eff = min(k_sub, k_null)

    mod = _get_module()
    W_null_gpu = pack_gf2_rows_gpu(W_null, torch_device)
    W_logical_gpu = pack_gf2_rows_gpu(L_logical, torch_device)

    current_target = n + 1
    best_weight: Optional[int] = None
    best_codeword: Optional[np.ndarray] = None
    trials_done = 0
    batch_seed = seed

    while trials_done < num_trials:
        B = min(batch_size, num_trials - trials_done)
        stats, out_vec, out_perm = mod.run_sqetch_ksub_recover(
            W_null_gpu, W_logical_gpu, n, k_sub_eff,
            B, batch_seed, current_target,
        )
        torch.cuda.synchronize()

        done = int(stats[2].item())
        trials_done += B
        batch_seed += B * 100003

        if not done:
            continue

        packed = out_vec.cpu().numpy().view(np.uint64)
        perm = out_perm.cpu().numpy()
        codeword = _packed_to_codeword(packed, perm, n)
        w = int(codeword.sum())

        if best_weight is None or w < best_weight:
            best_weight = w
            best_codeword = codeword
            current_target = w
            if d_target is not None and w < d_target:
                break

    return best_weight, best_codeword
