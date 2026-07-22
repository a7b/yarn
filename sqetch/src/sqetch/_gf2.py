"""GF(2) linear-algebra helpers.

* CPU null-space of a binary parity-check matrix ``H``, returned as the
  right null-space basis ``W`` with ``H @ W.T == 0 (mod 2)``.
* Bit-packed row layout: column ``j`` lives in word ``j // 64``, bit
  ``j % 64`` (LSB-first within each ``uint64``).  Packed rows are read
  directly by the CUDA kernel.
"""

from __future__ import annotations

import math

import numpy as np


def _gf2_gaussian_elim_fast(H_packed: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Bit-packed GF(2) Gaussian elimination returning RREF + pivot columns."""
    H = H_packed.copy()
    m, nw = H.shape
    pivot_cols: list[int] = []
    pivot_row = 0

    for col_word in range(nw):
        for col_bit in range(64):
            mask = np.uint64(1) << np.uint64(col_bit)
            col_vals = H[pivot_row:, col_word]
            set_bits = (col_vals & mask).astype(bool)
            if not np.any(set_bits):
                continue

            first_idx = int(np.argmax(set_bits))
            abs_first = pivot_row + first_idx
            if abs_first != pivot_row:
                H[[pivot_row, abs_first]] = H[[abs_first, pivot_row]]

            all_set = (H[:, col_word] & mask).astype(bool)
            all_set[pivot_row] = False
            H[all_set] ^= H[pivot_row]

            pivot_cols.append(col_word * 64 + col_bit)
            pivot_row += 1
            if pivot_row == m:
                return H, pivot_cols

    return H, pivot_cols


def pack_gf2_matrix(H: np.ndarray) -> np.ndarray:
    """Pack a ``(m, n)`` uint8 GF(2) matrix into ``(m, ceil(n/64))`` uint64."""
    assert H.ndim == 2
    m, n = H.shape
    nw = math.ceil(n / 64)
    n_padded = nw * 64

    if n_padded > n:
        H_padded = np.zeros((m, n_padded), dtype=np.uint8)
        H_padded[:, :n] = H
    else:
        H_padded = H.astype(np.uint8)

    H_reshaped = H_padded.reshape(m, nw, 64)
    powers = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    H_packed = (H_reshaped.astype(np.uint64) * powers[np.newaxis, np.newaxis, :]).sum(axis=2)
    return H_packed.astype(np.uint64)


def unpack_gf2_matrix(H_packed: np.ndarray, n_cols: int) -> np.ndarray:
    """Inverse of :func:`pack_gf2_matrix`."""
    assert H_packed.ndim == 2
    m, nw = H_packed.shape
    bits = np.arange(64, dtype=np.uint64)
    H_unpacked = ((H_packed[:, :, np.newaxis] >> bits[np.newaxis, np.newaxis, :]) & np.uint64(1))
    H_unpacked = H_unpacked.reshape(m, nw * 64).astype(np.uint8)
    return H_unpacked[:, :n_cols]


def gf2_null_space(H: np.ndarray) -> np.ndarray:
    """Right null space of ``H`` over GF(2).

    Returns ``W`` of shape ``(k, n)`` with ``k = n - rank(H)`` satisfying
    ``(H @ W.T) % 2 == 0``.  Returns ``(0, n)`` if the null space is trivial.
    """
    m, n = H.shape
    H_packed = pack_gf2_matrix(H)
    H_rref_packed, pivot_cols = _gf2_gaussian_elim_fast(H_packed)
    H_rref = unpack_gf2_matrix(H_rref_packed, n)

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    k = len(free_cols)
    if k == 0:
        return np.zeros((0, n), dtype=np.uint8)

    H_rref_trunc = H_rref[: len(pivot_cols), :]
    W = np.zeros((k, n), dtype=np.uint8)
    for i, fc in enumerate(free_cols):
        W[i, fc] = 1
        if pivot_cols:
            W[i, pivot_cols] = H_rref_trunc[:, fc]
    return W


def pack_gf2_rows_gpu(W: np.ndarray, device) -> "torch.Tensor":  # noqa: F821
    """Pack ``W`` to bit-packed ``int64`` and copy to ``device``.

    ``int64`` (not ``uint64``) is used because Torch lacks unsigned 64-bit
    tensors; the bit pattern is preserved.
    """
    import torch
    k, n = W.shape
    nw = math.ceil(n / 64)
    n_padded = nw * 64

    if n_padded > n:
        W_padded = np.zeros((k, n_padded), dtype=np.uint8)
        W_padded[:, :n] = W
    else:
        W_padded = W.astype(np.uint8)

    W_reshaped = W_padded.reshape(k, nw, 64)
    powers = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    W_packed = (W_reshaped.astype(np.uint64) * powers[np.newaxis, np.newaxis, :]).sum(axis=2)
    W_int64 = W_packed.view(np.int64)
    return torch.from_numpy(W_int64).to(device)
