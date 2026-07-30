"""Pool of full-rank ring elements + sampler that uses the pool.

Background
----------
LP codes with a *structured canonical basis* require an invertible
``ma·n × ma·n`` block-col submatrix of ``A_bin`` (and similarly of
``B_bin``). The post-hoc filter ``any_block_col_full_rank`` rejects
candidates that fail this property — but at search time it's much more
efficient to **construct** A_bin so the property holds by design:

1. Build a pool of ring elements ``x ∈ F₂[G]`` whose binary lift
   ``L[x]`` is invertible (a "full-rank anchor").
2. Sample A by **placing one pool element as the LAST block-col of A**
   (so canonical-form orbit-pairing finds it at position ``na-1``) and
   sampling the remaining ``na-1`` block-cols freely.

This avoids rejection sampling: every constructed A has a structured
canonical basis by construction.

ma = 1 only
-----------
Initial implementation handles ``ma = 1`` only — the common case for the
search pipeline. For ``ma > 1`` a pool element would be an ``ma × ma``
ring-matrix snippet (and we'd check invertibility of the combined
``ma·n × ma·n`` binary submatrix); raise ``NotImplementedError`` until
needed.

Same pool for A and B
---------------------
``rank(L[x]) == rank(R[x])`` for any ``x ∈ F₂[G]`` (both reps are
faithful and intertwined by the inverse map), so the same pool of full-
rank ring elements works as the canonical anchor for **both** A_bin
(uses ``left_rep``) and B_bin (uses ``right_rep``). No need to build
separate A-side and B-side pools.

The builder accepts a ``rep_func`` override anyway, in case a caller
wants to insist on, e.g., ``right_rep`` checks.
"""

import itertools
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from core.f2 import f2_rank
from core.group import GroupData, canonicalize, left_rep

from .random_ring_element import random_ring_element


# ─────────────────────────────────────────────────────────────────
# Pool builder
# ─────────────────────────────────────────────────────────────────


def build_full_rank_block_pool(
    gd: GroupData,
    weight: int,
    *,
    max_pool_size: int,
    max_tries: int = 100_000,
    seed: Optional[int] = None,
    rep_func: Callable = left_rep,
    ma: int = 1,
) -> list:
    """Build a pool of full-rank ring elements (ma = 1).

    Samples ring elements ``x ∈ F₂[G]`` of the given weight, accepts
    those whose ``rep_func(x, gd)`` is invertible (rank ``n``), until
    either ``max_pool_size`` are accepted or ``max_tries`` samples are
    drawn. The returned list contains canonical sorted tuples; the order
    is the order of acceptance.

    Args:
        gd: GroupData.
        weight: number of group elements per ring entry. Must be in
            ``[1, n]``. ``weight=1`` always yields full-rank elements
            (each ``L[g]`` is a permutation matrix), so the pool fills
            quickly. **Only odd weights can yield units** of F₂[G] in
            general (the augmentation ε(x) = Σ α_g must equal 1 mod 2).
            For some groups, certain odd weights ALSO have no units —
            e.g. S₃ has units only at weights 1 and 5; weights 3 give
            zero units despite being odd. If your pool comes back empty,
            try a different weight.
        max_pool_size: stop once this many elements are accepted.
        max_tries: budget on the total number of samples drawn
            (independent of acceptance). Bound on runtime.
        seed: RNG seed for reproducibility. ``None`` → fresh entropy.
        rep_func: ``left_rep`` (default) or ``right_rep``. The accepted
            set is the same for both, but callers can override for
            documentation purposes or to be defensive against future
            convention changes.
        ma: currently must be ``1``. Larger ``ma`` would mean each pool
            entry is an ``ma × ma`` ring-matrix snippet whose combined
            binary submatrix is invertible; that case raises
            ``NotImplementedError`` for now.

    Returns:
        A list of canonical ring elements (sorted tuples of distinct
        0-based group-element indices). May be empty if no full-rank
        element was found within the budget. May be shorter than
        ``max_pool_size`` if the budget ran out.

    Raises:
        ValueError: invalid ``weight``.
        NotImplementedError: ``ma != 1``.
    """
    if ma != 1:
        raise NotImplementedError(
            f"build_full_rank_block_pool currently supports ma=1 only; "
            f"got ma={ma}. (For ma>1 each pool element would be a "
            f"ring-matrix snippet of shape (ma, ma); add the enumerator "
            f"when needed.)"
        )
    n = gd.n
    if not (1 <= weight <= n):
        raise ValueError(
            f"weight must be in [1, n]; got weight={weight}, n={n}."
        )

    rng = np.random.default_rng(seed)
    pool: list = []
    seen: set = set()
    target_rank = n
    for _ in range(max_tries):
        if len(pool) >= max_pool_size:
            break
        x = random_ring_element(gd, weight, rng=rng, max_tries=1)
        if x is None or x in seen:
            continue
        seen.add(x)
        M = rep_func(x, gd)
        if f2_rank(M) == target_rank:
            pool.append(canonicalize(x))
    return pool


# ─────────────────────────────────────────────────────────────────
# Brute-force (exhaustive) enumeration
# ─────────────────────────────────────────────────────────────────


def enumerate_weight_elements(
    gd: GroupData,
    weight: int,
    *,
    force_identity: bool = False,
) -> Iterator[tuple]:
    """Yield every weight-``weight`` ring element, as canonical sorted tuples.

    A ring element over F₂[G] is a set of ``weight`` distinct group-element
    indices (the support). This enumerates all ``C(n, weight)`` of them — or,
    with ``force_identity=True``, only those that **contain the identity**
    (index 0), i.e. identity + ``C(n-1, weight-1)`` of the rest.

    Args:
        gd: GroupData.
        weight: support size; must be in ``[1, n]``.
        force_identity: if ``True``, every yielded element contains index 0.

    Yields:
        Canonical ring elements (sorted tuples of distinct 0-based indices).

    Raises:
        ValueError: invalid ``weight``, or ``force_identity`` with ``weight<1``.
    """
    n = gd.n
    if not (1 <= weight <= n):
        raise ValueError(f"weight must be in [1, n]; got weight={weight}, n={n}.")
    if force_identity:
        # identity (0) is fixed; choose weight-1 others from {1..n-1}.
        for rest in itertools.combinations(range(1, n), weight - 1):
            yield (0,) + rest          # already sorted: 0 < every rest element
    else:
        for combo in itertools.combinations(range(n), weight):
            yield combo                # combinations yields ascending tuples


def count_weight_elements(n: int, weight: int, *, force_identity: bool = False) -> int:
    """Number of elements :func:`enumerate_weight_elements` would yield."""
    import math
    if force_identity:
        return math.comb(n - 1, weight - 1)
    return math.comb(n, weight)


def build_full_rank_block_pool_brute(
    gd: GroupData,
    weight: int,
    *,
    force_identity: bool = True,
    rep_func: Callable = left_rep,
    max_pool_size: Optional[int] = None,
) -> list:
    """**Exhaustively** enumerate the full-rank ring-element pool (ma = 1).

    Unlike :func:`build_full_rank_block_pool` (random sampling with a try
    budget), this walks *every* weight-``weight`` element — by default only
    those containing the identity (the canonical-anchor convention) — and
    keeps each one whose binary lift ``rep_func(x, gd)`` is a unit
    (rank ``n``). The result is the **complete** anchor pool.

    Cost: ``C(n-1, weight-1)`` F2 rank tests (``force_identity=True``). For the
    search bands this is ≤ ~16k tests, i.e. seconds per group on one core.

    Args:
        gd: GroupData.
        weight: support size of each anchor (odd weights only can be units;
            some groups have no units at a given odd weight — then the pool is
            empty and the caller should fall back to ``weight=1`` monomials).
        force_identity: restrict to elements containing the identity (index 0).
            This is the canonical normalization used by the search.
        rep_func: ``left_rep`` (A-side) or ``right_rep`` (B-side). The accepted
            set is identical for both (``rank L[x] == rank R[x]``); the override
            exists for defensiveness / documentation.
        max_pool_size: optional early stop once this many units are accepted
            (``None`` = enumerate all). The search uses ``None`` (full brute).

    Returns:
        Complete list of canonical full-rank ring elements (acceptance order =
        enumeration order). May be empty if the group has no unit at this
        weight.

    Raises:
        ValueError: invalid ``weight``.
    """
    n = gd.n
    pool: list = []
    for x in enumerate_weight_elements(gd, weight, force_identity=force_identity):
        if f2_rank(rep_func(x, gd)) == n:
            pool.append(canonicalize(x))
            if max_pool_size is not None and len(pool) >= max_pool_size:
                break
    return pool


# ─────────────────────────────────────────────────────────────────
# Sampler that uses the pool
# ─────────────────────────────────────────────────────────────────


def _normalize_free_weights(free_col_weight, num_free_cols: int) -> list:
    """Accept int or sequence; return a length-``num_free_cols`` list of ints."""
    if isinstance(free_col_weight, int):
        return [free_col_weight] * num_free_cols
    fw = list(free_col_weight)
    if len(fw) != num_free_cols:
        raise ValueError(
            f"free_col_weight has length {len(fw)}, expected {num_free_cols} "
            f"(= na - ma)."
        )
    return [int(w) for w in fw]


def sample_A_from_pool(
    gd: GroupData,
    pool: Sequence,
    shape_a: tuple,
    free_col_weight,
    rng: np.random.Generator,
    *,
    free_col_knobs: Optional[dict] = None,
) -> Optional[list]:
    """Construct a ring matrix ``A`` with a guaranteed full-rank LAST block-col.

    Picks one ring element from ``pool`` and places it at column index
    ``na-1`` (the canonical orbit-pairing position). Samples the
    remaining ``na-1`` block-cols at columns ``0..na-2`` via
    :func:`random_ring_element` with weights given by ``free_col_weight``.

    The anchor (last column) comes directly from ``pool`` — its support
    is whatever the pool builder accepted, with no further filtering.
    The FREE columns are sampled fresh on each call using
    :func:`random_ring_element` defaults unless ``free_col_knobs`` is
    provided.

    Args:
        gd: GroupData.
        pool: non-empty sequence of ring elements (output of
            :func:`build_full_rank_block_pool`).
        shape_a: ``(ma, na)``. Currently ``ma == 1`` required.
        free_col_weight: weight constraint for the ``na - 1`` free
            block-cols. Either a single ``int`` (applied uniformly) or a
            sequence of length ``na - 1``.
        rng: numpy ``Generator``.
        free_col_knobs: optional dict forwarded as keyword arguments to
            :func:`random_ring_element` for each free column. Recognized
            keys: ``include_identity``, ``min_element_order``,
            ``avoid_same_coset``, ``max_tries``. Pass ``None`` (default)
            to use ``random_ring_element`` defaults
            (``include_identity=True``, ``min_element_order=1``,
            ``avoid_same_coset=False``, ``max_tries=1000``).

    Returns:
        A ring matrix (``list[list[tuple]]``) of shape ``(ma, na)``
        suitable for ``build_A_bin``. The LAST block-col entry is from
        ``pool``; the first ``na - 1`` entries are freshly sampled.
        ``None`` if any free-column sampler fails (rejection budget
        exhausted or eligible set too small).

    Raises:
        ValueError: empty pool, invalid shape, wrong-length
            ``free_col_weight``, or unknown ``free_col_knobs`` key.
        NotImplementedError: ``ma != 1``.
    """
    ma, na = shape_a
    if ma != 1:
        raise NotImplementedError(
            f"sample_A_from_pool currently supports ma=1 only; got ma={ma}."
        )
    if na < 1:
        raise ValueError(f"na must be ≥ 1; got na={na}.")
    if len(pool) == 0:
        raise ValueError("Pool is empty; cannot sample.")

    num_free = na - ma   # for ma=1: na - 1
    free_weights = _normalize_free_weights(free_col_weight, num_free)
    n = gd.n
    for w in free_weights:
        if not (1 <= w <= n):
            raise ValueError(
                f"each free-col weight must be in [1, n]; got {w}, n={n}."
            )

    knobs = dict(free_col_knobs) if free_col_knobs else {}
    allowed = {"include_identity", "min_element_order", "avoid_same_coset",
               "max_tries"}
    bad = set(knobs.keys()) - allowed
    if bad:
        raise ValueError(
            f"free_col_knobs has unknown key(s) {sorted(bad)}; "
            f"allowed: {sorted(allowed)}."
        )

    anchor = pool[int(rng.integers(len(pool)))]
    free_entries = []
    for w in free_weights:
        entry = random_ring_element(gd, w, rng=rng, **knobs)
        if entry is None:
            return None
        free_entries.append(canonicalize(entry))
    row = free_entries + [canonicalize(anchor)]
    return [row]
