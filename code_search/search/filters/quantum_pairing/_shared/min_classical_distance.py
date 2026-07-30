"""Pair filter: both classical distances must meet a threshold."""

from typing import Optional


def min_classical_distance(
    dist_A: Optional[int],
    dist_B: Optional[int],
    d_min: int,
) -> bool:
    """``True`` iff both sides' classical distance is at least ``d_min``.

    A side's distance may be ``None`` if the search stage didn't compute
    it (or genuinely couldn't refute the lower-bound hypothesis). ``None``
    is treated **pessimistically** here — it does NOT pass; if you want
    to allow ``None``, drop this filter from the config.

    Args:
        dist_A, dist_B: classical distance for each side (``int`` or
            ``None``). For abelian search pass the same value twice.
        d_min: minimum required.

    Returns:
        ``True`` iff ``dist_A >= d_min`` AND ``dist_B >= d_min``.
    """
    if dist_A is None or dist_B is None:
        return False
    return dist_A >= d_min and dist_B >= d_min
