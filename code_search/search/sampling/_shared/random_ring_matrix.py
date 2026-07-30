"""Random ring-matrix sampler.

Calls :func:`random_ring_element` once per ``(i, j)`` entry with the
weight read from a weight matrix. Knobs have a two-level resolution:

1. **Matrix-level defaults** — keyword arguments to
   :func:`random_ring_matrix`. Applied to every entry.
2. **Per-entry overrides** — ``overrides[(i, j)] = {knob: value}``
   replaces the default for that specific entry only.

After sampling, the matrix is canonicalized by default
(``canonicalize=True``) via :func:`core.classical_code.canonical_form_A`
— per-entry sort-of-indices PLUS permutation of block-cols so that an
invertible block-col subset (if one exists) lands at the LAST ``ma``
positions. Saved codes are therefore always in canonical form, ready for
``find_logical_basis`` / surgery without further reshuffling.

The ring-matrix canonicalization is rep-independent: ``rank(L[x]) ==
rank(R[x])``, so the invertible subset under ``build_A_bin`` is the same
under ``build_B_bin``. We use ``canonical_form_A`` purely as the entry
point; callers later lift with either ``build_A_bin`` or ``build_B_bin``.

If any entry's per-entry sampler returns ``None`` (rejection budget
exhausted or eligible set too small), the whole call returns ``None``.
The caller can retry, relax constraints, or skip this weight pattern.
"""

from typing import Optional

import numpy as np

from core.group import GroupData

from .random_ring_element import random_ring_element


_KNOB_KEYS = {"include_identity", "min_element_order", "avoid_same_coset",
              "max_tries"}


def random_ring_matrix(
    gd: GroupData,
    weight_matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    include_identity: bool = True,
    min_element_order: int = 1,
    avoid_same_coset: bool = False,
    max_tries: int = 1000,
    overrides: Optional[dict] = None,
    canonicalize: bool = True,
) -> Optional[list]:
    """Sample a ring matrix given the per-entry weight matrix.

    Args:
        gd: GroupData.
        weight_matrix: integer ndarray of shape ``(rows, cols)``;
            ``weight_matrix[i, j]`` is the target weight for entry ``A[i][j]``.
        rng: numpy ``Generator``.
        include_identity, min_element_order, avoid_same_coset, max_tries:
            matrix-level defaults forwarded to every entry's
            :func:`random_ring_element` call.
        overrides: optional ``dict[(i, j), dict[str, Any]]`` of per-entry
            knob overrides. Keys of the inner dict must be a subset of
            ``{"include_identity", "min_element_order", "avoid_same_coset",
            "max_tries"}``. Unknown keys raise ``ValueError``.
        canonicalize: if ``True`` (default), pipe the result through
            :func:`core.classical_code.canonical_form_A` so block-cols
            are permuted to put an invertible subset at the LAST ``ma``
            positions (when one exists). The ring-matrix-level
            canonicalization is rep-independent — works as the canonical
            form for both A-side (``build_A_bin``) and B-side
            (``build_B_bin``) downstream lifts.

    Returns:
        Ring matrix as ``list[list[tuple]]``, or ``None`` if any entry
        could not be sampled within its budget.

    Raises:
        ValueError: invalid ``weight_matrix`` shape, or an override
            specifies an unknown knob.
    """
    W = np.asarray(weight_matrix, dtype=int)
    if W.ndim != 2:
        raise ValueError(f"weight_matrix must be 2D; got shape {W.shape}.")
    rows, cols = W.shape
    overrides = overrides or {}
    for (i, j), knobs in overrides.items():
        if not (0 <= i < rows and 0 <= j < cols):
            raise ValueError(
                f"overrides key ({i}, {j}) is out of bounds for weight_matrix "
                f"of shape ({rows}, {cols})."
            )
        bad = set(knobs.keys()) - _KNOB_KEYS
        if bad:
            raise ValueError(
                f"overrides[({i}, {j})] has unknown knob(s) {sorted(bad)}; "
                f"allowed: {sorted(_KNOB_KEYS)}."
            )

    defaults = {
        "include_identity": include_identity,
        "min_element_order": min_element_order,
        "avoid_same_coset": avoid_same_coset,
        "max_tries": max_tries,
    }

    A: list = []
    for i in range(rows):
        row: list = []
        for j in range(cols):
            w = int(W[i, j])
            entry_knobs = {**defaults, **overrides.get((i, j), {})}
            entry = random_ring_element(gd, w, rng=rng, **entry_knobs)
            if entry is None:
                return None
            row.append(entry)
        A.append(row)

    if canonicalize:
        # Local import to avoid a circular core.classical_code <-> search
        # dependency at import time.
        from core.classical_code import canonical_form_A
        A_canonical, _, _, _ = canonical_form_A(A, gd)
        return A_canonical
    return A
