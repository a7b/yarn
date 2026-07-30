"""Upper bound on ``d(A_bin)`` from orders of weight-2 entries (ma=1)."""

from typing import Optional

from core.group import GroupData, element_order


def entry_order_bound(A: list, gd: GroupData) -> Optional[int]:
    """Upper bound on ``d(A_bin)`` from the orders of weight-2 entries.

    For any weight-2 entry ``{g1, g2}``, the kernel of ``L[g1+g2]`` equals
    ``ker(I + L[g1⁻¹·g2])``, whose elements are vectors constant on the
    left-multiplication orbits of ``g1⁻¹·g2`` (each orbit has size
    ``ord(g1⁻¹·g2)``). Taking the other column blocks to zero gives a
    codeword of ``ker(A_bin)`` of weight exactly ``ord(g1⁻¹·g2)``, so:

        d(A_bin) ≤ ord(g1⁻¹·g2)

    Args:
        A: ring matrix with ``ma = 1``. Each entry is an iterable of
           0-based group-element indices.
        gd: GroupData.

    Returns:
        Minimum of ``ord(g1⁻¹·g2)`` over all weight-2 entries of A, or
        ``None`` if no weight-2 entry exists.

    Raises:
        ValueError if ``ma != 1``.
    """
    if len(A) != 1:
        raise ValueError(
            f"entry_order_bound requires ma=1; got ma={len(A)}"
        )
    bound: Optional[int] = None
    for x in A[0]:
        if len(x) == 2:
            g1, g2 = tuple(x)
            ratio = gd.mult[gd.inv[g1]][g2]
            ord_ratio = element_order(ratio, gd)
            if bound is None or ord_ratio < bound:
                bound = ord_ratio
    return bound
