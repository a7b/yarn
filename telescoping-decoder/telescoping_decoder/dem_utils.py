"""Utilities for converting and sampling Stim detector error models."""

from __future__ import annotations

from typing import FrozenSet, List

import numpy as np
from scipy import sparse
import stim


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract dense check and observable matrices from a DEM.

    Returns
    -------
    check_matrix : ndarray of shape (num_detectors, num_errors), uint8
        Binary matrix: check_matrix[d, e] = 1 if error e flips detector d.
    obs_matrix : ndarray of shape (num_observables, num_errors), uint8
        Binary matrix: obs_matrix[k, e] = 1 if error e flips observable k.
    priors : ndarray of shape (num_errors,), float64
        Error probabilities.
    """
    check_matrix, obs_matrix, priors = dem_to_sparse_matrices(dem)
    return (
        check_matrix.toarray().astype(np.uint8, copy=False),
        obs_matrix.toarray().astype(np.uint8, copy=False),
        priors,
    )


def dem_to_sparse_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Extract sparse check and observable matrices from a DEM.

    Hyperedges are kept intact. The conversion does not decompose them into
    graphlike edges or construct edge-level matrices.

    Returns
    -------
    check_matrix : csr_matrix of shape (num_detectors, num_errors), uint8
    obs_matrix : csr_matrix of shape (num_observables, num_errors), uint8
    priors : ndarray of shape (num_errors,), float64

    Raises
    ------
    ValueError
        If two error mechanisms have identical detector support but different
        logical-observable support. Their XOR is a detector-silent logical
        error made from at most two mechanisms.
    """
    hyperedge_ids: dict[frozenset[int], int] = {}
    hyperedge_obs_map: dict[int, frozenset[int]] = {}
    priors_dict: dict[int, float] = {}

    def handle_error(
        prob: float,
        detectors: list[list[int]],
        observables: list[list[int]],
    ) -> None:
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0
        hid = hyperedge_ids[hyperedge_dets]
        if (
            hid in hyperedge_obs_map
            and hyperedge_obs_map[hid] != hyperedge_obs
        ):
            previous_obs = sorted(hyperedge_obs_map[hid])
            incoming_obs = sorted(hyperedge_obs)
            raise ValueError(
                "DEM contains error mechanisms with identical detector "
                f"support {sorted(hyperedge_dets)} but different logical-"
                f"observable support {previous_obs} and {incoming_obs}. "
                "XORing them gives a detector-silent, logically nontrivial "
                "error, so the circuit-level distance is at most 2. Refusing "
                "to merge them."
            )
        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: list[list[int]] = [[]]
            obs: list[list[int]] = [[]]
            prob = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    obs[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    obs.append([])
            handle_error(prob, dets, obs)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError(f"Unsupported DEM instruction: {instruction.type}")

    num_hyperedges = len(hyperedge_ids)
    check_rows: list[int] = []
    check_cols: list[int] = []
    for dets, hid in hyperedge_ids.items():
        for d in dets:
            check_rows.append(d)
            check_cols.append(hid)

    obs_rows: list[int] = []
    obs_cols: list[int] = []
    for hid, obs in hyperedge_obs_map.items():
        for o in obs:
            obs_rows.append(o)
            obs_cols.append(hid)

    check_matrix = sparse.csr_matrix(
        (
            np.ones(len(check_rows), dtype=np.uint8),
            (np.asarray(check_rows, dtype=np.int32), np.asarray(check_cols, dtype=np.int32)),
        ),
        shape=(dem.num_detectors, num_hyperedges),
        dtype=np.uint8,
    )
    obs_matrix = sparse.csr_matrix(
        (
            np.ones(len(obs_rows), dtype=np.uint8),
            (np.asarray(obs_rows, dtype=np.int32), np.asarray(obs_cols, dtype=np.int32)),
        ),
        shape=(dem.num_observables, num_hyperedges),
        dtype=np.uint8,
    )
    priors = np.zeros(num_hyperedges, dtype=np.float64)
    for hid, prob in priors_dict.items():
        priors[hid] = prob

    return check_matrix, obs_matrix, priors


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    out = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def sample_dem_syndromes(
    dem_or_sampler,
    batch_size: int,
    *,
    seed: int | None = None,
    dtype: np.dtype = np.uint8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector syndromes and observables from a Stim DEM sampler."""
    if hasattr(dem_or_sampler, "sample"):
        if seed is not None:
            raise ValueError("seed cannot be provided when passing a precompiled sampler")
        sampler = dem_or_sampler
    else:
        sampler = dem_or_sampler.compile_sampler(seed=seed)

    syndromes, observables, _ = sampler.sample(
        shots=batch_size,
        bit_packed=False,
    )
    return np.asarray(syndromes, dtype=dtype), np.asarray(observables, dtype=dtype)


def binary_matmul_mod2(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray | sparse.spmatrix,
) -> np.ndarray:
    """Compute ``(left_matrix @ right_matrix.T) % 2`` for binary matrices.

    ``right_matrix`` may be dense or sparse. The result is returned as ``uint8``.
    """
    left = np.asarray(left_matrix, dtype=np.uint8)
    if sparse.issparse(right_matrix):
        product = left @ right_matrix.transpose().astype(np.uint8)
        return (np.asarray(product, dtype=np.uint8) & 1).astype(np.uint8, copy=False)

    right = np.asarray(right_matrix, dtype=np.uint8)
    return ((left @ right.T) & 1).astype(np.uint8, copy=False)


def _as_binary_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary matrix")
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError(f"{name} must contain only 0/1 entries")
    return matrix.astype(np.uint8, copy=False)


def _normalize_logicals(
    logicals: np.ndarray | None,
    num_qubits: int,
    name: str,
) -> np.ndarray | None:
    if logicals is None:
        return None

    logicals = np.asarray(logicals)
    if logicals.ndim == 1:
        logicals = logicals.reshape(1, -1)
    if logicals.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D binary matrix")
    if logicals.shape[1] != num_qubits:
        raise ValueError(
            f"{name} must have one column per data qubit: "
            f"got {logicals.shape[1]}, expected {num_qubits}"
        )
    if np.any((logicals != 0) & (logicals != 1)):
        raise ValueError(f"{name} must contain only 0/1 entries")
    return logicals.astype(np.uint8, copy=False)


def _validate_probability(probability: float, name: str) -> None:
    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {probability}")
