"""Non-abelian ``SmallGroup`` enumeration over an order range (GAP-backed).

The smallest shape-(1,2) code has ``n = 5|G|``, so the *smallest group order
that yields a verified target-distance code is provably optimal*. We
therefore sweep orders ascending and stop at the first success. A few orders
have an enormous number of non-abelian groups (96 → 224, 144 → 187,
160 → 231); those are **deferred** to a second pass so the small orders
resolve first.

This module needs ``gappy`` + GAP. It runs **on the driver host**; remote
workers never call GAP — they receive serialized Cayley tables.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# Orders with an unusually large number of non-abelian groups; deferred to a
# second pass on each band. (Counts measured via GAP: 96→224, 144→187,
# 160→231.)
DEFAULT_DEFER = (96, 144, 160)

# Target bands. Primary lists exclude the deferred
# giants; ``enumerate_band`` can fold them back in for a second pass.
D16_BAND = range(84, 101)          # distance 16: |G| in 84..100
D2224_BAND = range(140, 181)       # distance 22 & 24: |G| in 140..180


@dataclass(frozen=True)
class GroupSpec:
    """One non-abelian group to search."""

    order: int
    small_group_id: int

    @property
    def gap_expr(self) -> str:
        return f"SmallGroup({self.order},{self.small_group_id})"

    @property
    def tag(self) -> str:
        return f"Sg{self.order}_{self.small_group_id}"


def nonabelian_ids(order: int) -> List[int]:
    """Ascending ``SmallGroup`` IDs of every non-abelian group of ``order``.

    Returns ``[]`` for orders with no non-abelian group (e.g. primes,
    cyclic-only orders).
    """
    from gappy import gap

    n = int(gap.eval(f"NrSmallGroups({order})"))
    out: List[int] = []
    for i in range(1, n + 1):
        if not bool(gap.eval(f"IsAbelian(SmallGroup({order},{i}))")):
            out.append(i)
    return out


def enumerate_orders(
    orders: Iterable[int],
    *,
    defer: Sequence[int] = DEFAULT_DEFER,
    include_deferred: bool = False,
) -> List[GroupSpec]:
    """All non-abelian groups for ``orders``, ascending by ``(order, id)``.

    Args:
        orders: iterable of group orders to scan.
        defer: orders to hold back (too many non-abelian groups). On a
            normal (first) pass these are skipped entirely.
        include_deferred: if ``True``, the deferred orders are included
            **in their natural ascending position** (so a re-run that
            folds them in still resolves the smallest order first).

    Returns:
        Ascending list of :class:`GroupSpec`. With ``include_deferred=False``
        the deferred orders contribute nothing.
    """
    defer_set = set(defer)
    specs: List[GroupSpec] = []
    for o in sorted(orders):
        if o in defer_set and not include_deferred:
            continue
        for i in nonabelian_ids(o):
            specs.append(GroupSpec(order=o, small_group_id=i))
    return specs


def enumerate_band(
    band: str,
    *,
    include_deferred: bool = False,
) -> List[GroupSpec]:
    """Enumerate a named target band.

    Args:
        band: ``"d16"`` (orders 84..100) or ``"d2224"`` (orders 140..180).
        include_deferred: fold the deferred giant orders back in (second pass).
    """
    if band == "d16":
        orders: Iterable[int] = D16_BAND
    elif band == "d2224":
        orders = D2224_BAND
    else:
        raise ValueError(f"unknown band {band!r}; expected 'd16' or 'd2224'.")
    return enumerate_orders(orders, include_deferred=include_deferred)


def orders_in_band(band: str, *, include_deferred: bool = False) -> List[int]:
    """Distinct group orders that actually contain a non-abelian group, ascending.

    Useful for wave scheduling (process a few orders at a time, stop at the
    smallest that succeeds).
    """
    specs = enumerate_band(band, include_deferred=include_deferred)
    seen: List[int] = []
    for s in specs:
        if not seen or seen[-1] != s.order:
            if s.order not in seen:
                seen.append(s.order)
    return sorted(set(o for o in seen))


# ─────────────────────────────────────────────────────────────────
# Commutator-ratio group selection (for the random-sampling sweep)
# ─────────────────────────────────────────────────────────────────


def commutator_ratio(gap_expr: str) -> tuple:
    """``(|G|, |[G,G]|, |[G,G]|/|G|)`` for a GAP group expression (GAP-local)."""
    from gappy import gap

    n = int(gap.eval(f"Size({gap_expr})"))
    d = int(gap.eval(f"Order(DerivedSubgroup({gap_expr}))"))
    return n, d, d / n


def _pick_low_high(ranked: Sequence[tuple], n_low: int, n_high: int) -> list:
    """Pure helper: from ``[(ratio, item), ...]`` pick the ``n_low`` lowest
    ratios + the ``n_high`` highest, with no duplicates.

    Returns the chosen ``item`` objects, low-ratio ones first then high-ratio.
    If the pool has ``≤ n_low + n_high`` entries, every item is returned
    (ascending by ratio).
    """
    ordered = sorted(ranked, key=lambda t: (t[0], str(getattr(t[1], "tag", t[1]))))
    if len(ordered) <= n_low + n_high:
        return [item for _, item in ordered]
    low = ordered[:n_low]
    high = ordered[len(ordered) - n_high:]
    chosen = [item for _, item in low]
    low_ids = {id(item) for _, item in low}
    for _, item in high:
        if id(item) not in low_ids:
            chosen.append(item)
    return chosen


def select_groups_by_ratio(order: int, *, n_low: int = 3, n_high: int = 2) -> List[GroupSpec]:
    """Non-abelian groups of ``order``: the ``n_low`` lowest ``|[G,G]|/|G|``
    plus the ``n_high`` highest (GAP-local).

    Returns ``[]`` for orders with no non-abelian group. Fewer than
    ``n_low + n_high`` groups ⇒ all of them (ascending by ratio).
    """
    specs = [GroupSpec(order=order, small_group_id=i) for i in nonabelian_ids(order)]
    if not specs:
        return []
    ranked = [(commutator_ratio(s.gap_expr)[2], s) for s in specs]
    return _pick_low_high(ranked, n_low, n_high)


def sweep_orders(order_min: int, order_max: int) -> List[int]:
    """Orders in ``[order_min, order_max]`` that contain ≥1 non-abelian group
    (ascending, GAP-local). Used to drive the random-sampling sweep cursor."""
    return [o for o in range(order_min, order_max + 1) if nonabelian_ids(o)]
