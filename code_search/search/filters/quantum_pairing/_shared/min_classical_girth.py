"""Pair filter: both classical Tanner girths must meet a threshold."""

from typing import Optional


def min_classical_girth(
    girth_A: Optional[int],
    girth_B: Optional[int],
    g_min: int,
) -> bool:
    """``True`` iff both sides' classical Tanner girth is at least ``g_min``.

    ``None`` means the side's Tanner graph is a forest (no cycles) —
    treated as **infinite girth** and PASSES the threshold.

    Args:
        girth_A, girth_B: Tanner girth for each side (``int`` or
            ``None``).
        g_min: minimum required girth.

    Returns:
        ``True`` iff each girth is either ``None`` or ``>= g_min``.
    """
    pass_A = girth_A is None or girth_A >= g_min
    pass_B = girth_B is None or girth_B >= g_min
    return pass_A and pass_B
