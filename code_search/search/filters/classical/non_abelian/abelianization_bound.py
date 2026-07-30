"""Abelianization bound on ``d(A_bin)`` (ma=1), non-abelian only.

For ``G`` non-abelian, the abelianization map ``π: G → G/[G,G]`` induces
``A → A^{ab}``, a code over ``F₂[G/[G,G]]``. Any codeword of ``A_bin^{ab}``
of weight ``w`` lifts to a codeword of ``A_bin`` of weight ``≤ w·|[G,G]|``.
The abelian-side distance is at most the total row weight, so:

    d(A_bin) ≤ (total entries in row) × |[G,G]|

This filter lives in ``non_abelian/`` because for abelian ``G`` we have
``[G,G] = {e}`` and the bound collapses to ``total_entries`` — strictly
dominated by ``abelian/ring_distance_bound`` (J=1 case is identical, and
``I ≥ 3`` the ring permanent picks the tighter pair-sum).
"""

from core.group import GroupData


def abelianization_bound(A: list, gd: GroupData) -> int:
    """Abelianization-derived upper bound on ``d(A_bin)`` for a single-row A.

    Bound:  ``d(A_bin) ≤ (total group elements in row) × |[G,G]|``.

    Derivation: the abelianization map ``G → G/[G,G]`` induces
    ``A_bin → A_bin^{ab}`` (a code over ``F₂[G/[G,G]]``). Any codeword of
    ``A_bin^{ab}`` of weight ``w`` lifts to a codeword of ``A_bin`` of
    weight ``≤ w × |[G,G]|``. The abelian-side distance is at most the
    total row weight, giving the bound.

    Args:
        A: ring matrix with ``ma = 1``.
        gd: GroupData. Non-abelian (abelian ``G`` gives a trivial bound;
            use ``abelian/ring_distance_bound`` instead).

    Returns:
        int upper bound.

    Raises:
        ValueError if ``ma != 1`` or ``gd.is_abelian``.
    """
    if len(A) != 1:
        raise ValueError(
            f"abelianization_bound requires ma=1; got ma={len(A)}"
        )
    if gd.is_abelian:
        raise ValueError(
            "abelianization_bound is non-abelian only; for abelian G use "
            "ring_distance_bound (the abelianization bound is trivial)."
        )
    total = sum(len(A[0][j]) for j in range(len(A[0])))
    return total * gd.commutator_order
