"""Stable identities for decoding systems and their cached artifacts."""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp


FINGERPRINT_VERSION = 1


def _update_array(digest, name: str, value, dtype) -> None:
    """Hash an array with an explicit name, shape, and portable dtype."""
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(name.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def source_fingerprint(
    H,
    L,
    priors,
    *,
    is_x_detector=None,
    init_basis=None,
) -> str:
    """Return a complete, stable identity for one physical decode system.

    The identity includes every input that changes an original, GARI, or
    init-detectors artifact. It intentionally identifies the source problem,
    rather than a particular transformed representation, so matching original
    and GARI NPZ files carry the same value.
    """
    digest = hashlib.sha256()
    digest.update(f"telescoping-decoder-system-v{FINGERPRINT_VERSION}\0".encode())
    H = sp.csc_matrix(H, dtype=np.uint8).copy()
    L = sp.csc_matrix(L, dtype=np.uint8).copy()
    H.sum_duplicates(); H.sort_indices()
    L.sum_duplicates(); L.sort_indices()
    priors = np.asarray(priors, dtype="<f8")
    if H.shape[1] != L.shape[1] or priors.shape != (H.shape[1],):
        raise ValueError("H, L, and priors must describe the same columns")
    _update_array(digest, "source_shape", (H.shape[0], L.shape[0], H.shape[1]),
                  "<i8")

    # GARI groups physical columns by X/Y/Z type, whereas the original NPZ
    # retains DEM insertion order. Hash complete column records and sort them
    # so those equivalent orderings have one source identity.
    column_ids = []
    for j in range(H.shape[1]):
        column = hashlib.sha256()
        hs = slice(H.indptr[j], H.indptr[j + 1])
        ls = slice(L.indptr[j], L.indptr[j + 1])
        _update_array(column, "h.indices", H.indices[hs], "<i8")
        _update_array(column, "h.data", H.data[hs], "u1")
        _update_array(column, "l.indices", L.indices[ls], "<i8")
        _update_array(column, "l.data", L.data[ls], "u1")
        _update_array(column, "prior", priors[j:j + 1], "<f8")
        column_ids.append(column.digest())
    for column_id in sorted(column_ids):
        digest.update(column_id)
    if is_x_detector is None:
        digest.update(b"is_x_detector:none\0")
    else:
        _update_array(digest, "is_x_detector", is_x_detector, "u1")
    digest.update(b"init_basis\0")
    digest.update(("" if init_basis is None else str(init_basis)).encode("ascii"))
    return digest.hexdigest()


__all__ = ["FINGERPRINT_VERSION", "source_fingerprint"]
