"""Pair filter: A's shape equals B's shape.

LP CSS codes are mathematically valid for any ``(ma, na, mb, nb)`` —
each binary block-Kronecker term is dimensionally consistent. In most
searches we still want ``(mb, nb) == (ma, na)``: it keeps Hx and Hz
balanced, the same weight-matrix family applies to both sides, and the
saved-code metadata uses one ``shape`` field for both. This filter
enforces that convention; drop it from the config when you genuinely
want mixed shapes.
"""


def same_shape(shape_A: tuple, shape_B: tuple) -> bool:
    """``True`` iff the two classical sides have identical ``(ma, na)``."""
    return tuple(shape_A) == tuple(shape_B)
