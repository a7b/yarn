"""Random per-entry ring-element sampler with constraints.

One function dispatches by weight — weight 1 (monomial) and weight ≥ 2
(polynomial) are handled uniformly. Returns a canonical sorted tuple, or
``None`` if the rejection budget was exhausted.

Constraints
-----------
``include_identity`` (default ``True``)
    Whether the group identity element (index ``gd.identity == 0``) is
    allowed as one of the chosen group elements.

``min_element_order`` (default ``1`` = no constraint)
    Every chosen element ``g`` must satisfy ``order(g) ≥ this``. The
    weight-2 orbit bound ``d(A_bin) ≤ ord(g₁⁻¹g₂)`` motivates this knob
    (avoids pre-doomed entries).

``avoid_same_coset`` (default ``False``) — **non-abelian only**
    The chosen elements must lie in pairwise-distinct left cosets of
    ``[G,G]``. Improves the abelianization bound by ensuring the
    F₂[G/[G,G]] image of the ring element has the same weight as the
    ring element itself (no cancellation under abelianization).
    Raises ``ValueError`` if used with abelian ``G`` (the bound is
    trivial there since ``[G,G] = {e}``).

Strategy
--------
Pre-filter to the *eligible set* once (uses ``include_identity`` and
``min_element_order``); then rejection-sample subsets of size ``weight``
from that set. ``avoid_same_coset`` is enforced via per-subset rejection
(coset membership comes from ``gd.coset_id``).

If the eligible set is smaller than ``weight``, returns ``None``
immediately. If no satisfying subset is found within ``max_tries``,
returns ``None`` (caller can retry with a larger budget or relax
constraints).
"""

from typing import Optional

import numpy as np

from core.group import GroupData, element_order


def random_ring_element(
    gd: GroupData,
    weight: int,
    *,
    rng: np.random.Generator,
    include_identity: bool = True,
    min_element_order: int = 1,
    avoid_same_coset: bool = False,
    max_tries: int = 1000,
) -> Optional[tuple]:
    """Sample a single ring element satisfying the constraints.

    Args:
        gd: GroupData.
        weight: target number of group elements in the ring element.
            ``weight = 0`` is allowed and returns ``()`` (the zero
            ring element).
        rng: numpy ``Generator``.
        include_identity: see module docstring.
        min_element_order: see module docstring. ``1`` disables.
        avoid_same_coset: see module docstring. Raises on abelian ``G``.
        max_tries: rejection-sampling budget across all retries.

    Returns:
        Canonical sorted tuple of distinct 0-based group-element
        indices (length ``weight``), or ``None`` if the budget is
        exhausted or the eligible set is too small.

    Raises:
        ValueError: ``avoid_same_coset=True`` with abelian ``G``, or
            invalid ``weight``.
    """
    n = gd.n
    if weight < 0 or weight > n:
        raise ValueError(f"weight must be in [0, n]; got {weight}, n={n}.")
    if avoid_same_coset and gd.is_abelian:
        raise ValueError(
            "avoid_same_coset is non-abelian only; for abelian G the "
            "commutator subgroup is trivial so the constraint rejects "
            "every weight-≥2 element."
        )
    if weight == 0:
        return ()

    # Eligible set: pass include_identity and min_element_order filters.
    eligible = []
    for g in range(n):
        if not include_identity and g == gd.identity:
            continue
        if min_element_order > 1 and element_order(g, gd) < min_element_order:
            continue
        eligible.append(g)
    if len(eligible) < weight:
        return None

    # Feasibility pre-check for avoid_same_coset: if the eligible set
    # spans fewer than ``weight`` distinct cosets of [G,G], no subset of
    # size ``weight`` can have ``weight`` distinct cosets. Return None
    # immediately instead of burning the full ``max_tries`` budget.
    # NOTE: ``gd.coset_id`` is read ONLY here (not unconditionally) — GAP-free
    # duck-typed group objects may not carry it, and the default
    # avoid_same_coset=False path must work without it.
    if avoid_same_coset:
        coset_id = gd.coset_id   # list; len == n
        distinct_cosets = len({coset_id[g] for g in eligible})
        if distinct_cosets < weight:
            return None

    eligible_arr = np.asarray(eligible)
    for _ in range(max_tries):
        idx = rng.choice(len(eligible_arr), size=weight, replace=False)
        subset = sorted(int(eligible_arr[i]) for i in idx)
        if avoid_same_coset:
            cosets = {coset_id[g] for g in subset}
            if len(cosets) != weight:
                continue
        return tuple(subset)
    return None
