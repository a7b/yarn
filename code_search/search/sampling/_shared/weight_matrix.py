"""Weight-pattern enumeration for the abelian two-stage classical search.

The abelian classical loop is cheaper if we first screen INTEGER weight
matrices ``W`` (no ring elements, no GAP) and then only sample ring
placements for surviving patterns. ``random_weight_patterns`` is the
random-sampling stream (used by the search phase); ``all_weight_patterns``
is the exhaustive iterator (for small spaces).

Both apply the same filter set::

    max_row_weight              every row sum  ≤ threshold
    max_col_weight              every col sum  ≤ threshold
    min_base_girth_bound        base_girth_bound(W)        ≥ threshold
    min_weight_distance_bound   weight_distance_bound(W, gd) ≥ threshold   (ma=1, abelian)

All optional (``None`` disables). The base / weight distance filters
reuse the existing filter functions in
``search/filters/classical/abelian/``.

Duplicate suppression is on by default: an internal ``set`` of seen
(tuple-of-tuples) patterns avoids yielding the same ``W`` twice. Useful
because random sampling over ``[0, entry_max]^(ma·na)`` will revisit
patterns at small ``entry_max``.
"""

import itertools
from typing import Iterator, Optional

import numpy as np

from core.group import GroupData
from search.filters.classical.abelian.base_girth_bound import base_girth_bound
from search.filters.classical.abelian.weight_distance_bound import (
    weight_distance_bound,
)


def _row_sums_ok(W: np.ndarray, threshold: Optional[int]) -> bool:
    if threshold is None:
        return True
    return int(W.sum(axis=1).max()) <= threshold


def _col_sums_ok(W: np.ndarray, threshold: Optional[int]) -> bool:
    if threshold is None:
        return True
    return int(W.sum(axis=0).max()) <= threshold


def _passes_weight_filters(
    W: np.ndarray,
    *,
    max_row_weight: Optional[int],
    max_col_weight: Optional[int],
    min_base_girth_bound_val: Optional[int],
    min_weight_distance_bound_val: Optional[int],
    gd: Optional[GroupData],
) -> bool:
    """Apply weight-pattern filters cheapest-first.

    Args:
        W: integer weight matrix.
        max_row_weight, max_col_weight, min_base_girth_bound_val,
            min_weight_distance_bound_val: filter thresholds.
        gd: GroupData; required if ``min_weight_distance_bound_val`` is
            set (the bound is abelian-only and consults ``gd.is_abelian``).
    """
    if not _row_sums_ok(W, max_row_weight):
        return False
    if not _col_sums_ok(W, max_col_weight):
        return False
    if min_base_girth_bound_val is not None:
        if base_girth_bound(W.tolist()) < min_base_girth_bound_val:
            return False
    if min_weight_distance_bound_val is not None:
        if gd is None:
            raise ValueError(
                "min_weight_distance_bound requires GroupData (passed as gd)."
            )
        if weight_distance_bound(W, gd) < min_weight_distance_bound_val:
            return False
    return True


def random_weight_patterns(
    shape: tuple,
    entry_max: int,
    num_samples: int,
    *,
    rng: np.random.Generator,
    entry_min: int = 0,
    max_row_weight: Optional[int] = None,
    max_col_weight: Optional[int] = None,
    min_base_girth_bound: Optional[int] = None,
    min_weight_distance_bound: Optional[int] = None,
    gd: Optional[GroupData] = None,
    max_tries: Optional[int] = None,
    dedupe: bool = True,
) -> Iterator[np.ndarray]:
    """Yield random unique integer weight matrices passing the filters.

    Each call returns an iterator that yields **at most** ``num_samples``
    surviving patterns. Internally samples one matrix at a time from
    ``[entry_min, entry_max]^(ma·na)``, applies filters, and (if
    ``dedupe=True``) skips already-yielded patterns.

    Args:
        shape: ``(ma, na)``.
        entry_max: max integer per entry (inclusive).
        num_samples: max patterns to yield. Iteration stops when reached.
        rng: numpy ``Generator``.
        entry_min: min integer per entry (inclusive). Defaults to 0;
            set to ``1`` to force every entry to be non-empty.
        max_row_weight, max_col_weight, min_base_girth_bound,
            min_weight_distance_bound: weight-pattern filter thresholds.
            All optional (``None`` disables).
        gd: GroupData; required when ``min_weight_distance_bound`` is set.
        max_tries: hard cap on raw sample draws. Defaults to
            ``num_samples * 100`` — enough for moderate filter rejection
            rates. Set higher if your filters reject aggressively.
        dedupe: skip patterns already yielded.

    Yields:
        ``np.ndarray`` of shape ``(ma, na)`` and dtype int.
    """
    ma, na = shape
    if entry_min < 0:
        raise ValueError(f"entry_min must be ≥ 0; got {entry_min}.")
    if entry_max < entry_min:
        raise ValueError(
            f"entry_max must be ≥ entry_min; got entry_min={entry_min}, "
            f"entry_max={entry_max}."
        )
    if num_samples < 0:
        raise ValueError(f"num_samples must be ≥ 0; got {num_samples}.")
    if num_samples == 0:
        return

    budget = num_samples * 100 if max_tries is None else max_tries
    seen: set = set()
    yielded = 0
    for _ in range(budget):
        if yielded >= num_samples:
            return
        W = rng.integers(entry_min, entry_max + 1, size=(ma, na))
        if dedupe:
            key = tuple(tuple(int(x) for x in row) for row in W)
            if key in seen:
                continue
            seen.add(key)
        if not _passes_weight_filters(
            W,
            max_row_weight=max_row_weight,
            max_col_weight=max_col_weight,
            min_base_girth_bound_val=min_base_girth_bound,
            min_weight_distance_bound_val=min_weight_distance_bound,
            gd=gd,
        ):
            continue
        yield W
        yielded += 1


def all_weight_patterns(
    shape: tuple,
    entry_max: int,
    *,
    entry_min: int = 0,
    max_row_weight: Optional[int] = None,
    max_col_weight: Optional[int] = None,
    min_base_girth_bound: Optional[int] = None,
    min_weight_distance_bound: Optional[int] = None,
    gd: Optional[GroupData] = None,
) -> Iterator[np.ndarray]:
    """Yield ALL integer weight matrices in
    ``[entry_min, entry_max]^(ma·na)`` passing the filters, in
    lexicographic order.

    Use only for small spaces — the iteration cost is
    ``(entry_max - entry_min + 1)^(ma·na)``.
    """
    ma, na = shape
    if entry_min < 0:
        raise ValueError(f"entry_min must be ≥ 0; got {entry_min}.")
    if entry_max < entry_min:
        raise ValueError(
            f"entry_max must be ≥ entry_min; got entry_min={entry_min}, "
            f"entry_max={entry_max}."
        )
    for flat in itertools.product(range(entry_min, entry_max + 1),
                                  repeat=ma * na):
        W = np.array(flat, dtype=int).reshape(ma, na)
        if _passes_weight_filters(
            W,
            max_row_weight=max_row_weight,
            max_col_weight=max_col_weight,
            min_base_girth_bound_val=min_base_girth_bound,
            min_weight_distance_bound_val=min_weight_distance_bound,
            gd=gd,
        ):
            yield W
