"""Estimate the distance of the [[144, 12, 12]] bivariate-bicycle code.

Loads ``Hx.npy`` / ``Hz.npy`` from a directory the user supplies via
the ``SQETCH_BB_PATH`` environment variable, constructs an X-logical
basis via two GF(2) null-space computations, and runs
``estimate_distance`` for d_Z.

If ``SQETCH_BB_PATH`` is unset or the files are missing, the script
prints an instructive message and exits cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from sqetch import estimate_distance
from sqetch._gf2 import gf2_null_space


def logical_basis_x(H_X: np.ndarray, H_Z: np.ndarray) -> np.ndarray:
    """X-logical basis: rows of ker(H_Z) reduced modulo rowspan(H_X).

    Concretely we compute ker(H_Z), stack rowspan(H_X) on top, and pick
    rows of ker(H_Z) whose addition to the stack increases its rank.
    """
    null_Hz = gf2_null_space(H_Z)
    n = H_X.shape[1]
    basis = list(H_X.copy())
    rank_h = _rank_gf2(np.vstack(basis) if basis else np.zeros((0, n), dtype=np.uint8))
    logicals = []
    for row in null_Hz:
        trial = basis + [row]
        new_rank = _rank_gf2(np.vstack(trial))
        if new_rank > rank_h:
            basis.append(row)
            logicals.append(row)
            rank_h = new_rank
    if not logicals:
        return np.zeros((0, n), dtype=np.uint8)
    return np.array(logicals, dtype=np.uint8)


def _rank_gf2(M: np.ndarray) -> int:
    if M.size == 0:
        return 0
    A = M.copy().astype(np.uint8)
    m, n = A.shape
    r = 0
    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if A[i, c]:
                pivot = i
                break
        if pivot == -1:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break
    return r


def main() -> None:
    env_path = os.environ.get("SQETCH_BB_PATH")
    if env_path is None:
        print("Set SQETCH_BB_PATH to a directory containing Hx.npy and "
              "Hz.npy for a [[144, 12, 12]] bivariate-bicycle code, or "
              "any CSS code you want to evaluate.")
        sys.exit(0)

    bb_path = Path(env_path)
    hx_path = bb_path / "Hx.npy"
    hz_path = bb_path / "Hz.npy"
    if not (hx_path.exists() and hz_path.exists()):
        print(f"Hx.npy / Hz.npy not found under {bb_path}.")
        print("Check SQETCH_BB_PATH, or substitute your own CSS code.")
        sys.exit(0)

    H_X = np.load(hx_path).astype(np.uint8)
    H_Z = np.load(hz_path).astype(np.uint8)
    print(f"Loaded H_X: {H_X.shape}, H_Z: {H_Z.shape}")

    L_X = logical_basis_x(H_X, H_Z)
    print(f"X-logical basis: {L_X.shape[0]} rows")

    result = estimate_distance(
        H_X,
        L_X,
        num_trials=200_000,
        d_target=12,
        k_sub=16,
        seed=42,
    )
    print(f"d_Z bound:        {result.best_weight}")
    print(f"trials run:       {result.trials_run}")
    print(f"early-stop found: {result.found}")
    print(f"elapsed:          {result.elapsed_seconds:.2f} s")
    print(f"raw iter/s:       {result.raw_iter_per_sec:.0f}")


if __name__ == "__main__":
    main()
