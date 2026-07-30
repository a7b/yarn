"""Filter configuration + cheapest-first dispatchers.

Two dispatchers, one per stage:

- :func:`apply_classical_filters(A, A_bin, gd, cfg)` — operates on a
  single classical side (one ring matrix ``A`` and its lift ``A_bin``).
  Returns ``True`` iff every enabled filter passes; short-circuits on
  the first rejection.

- :func:`apply_quantum_pairing_filters(meta_A, meta_B, cfg)` — operates
  on metadata of two saved classical codes BEFORE building Hx, Hz. For
  abelian searches, callers pass ``meta_A == meta_B``.

Each filter is enabled by setting a threshold on the corresponding
config field. Disabled = ``None`` (or ``False`` for boolean toggles).

Cheapest-first order is documented inside each function. Programming-
error preconditions (e.g. ``ma == 1`` for ma-restricted filters) raise
``ValueError`` rather than silently passing — preventing accidental
misuse.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.group import GroupData
from search.filters.classical._shared.any_block_col_full_rank import (
    any_block_col_full_rank,
)
from search.filters.classical.abelian.base_girth_bound import base_girth_bound
from search.filters.classical.non_abelian.abelianization_bound import (
    abelianization_bound,
)
from search.filters.classical._shared.entry_order_bound import (
    entry_order_bound,
)
from search.filters.classical._shared.girth_tanner import girth_tanner
from search.filters.quantum_pairing._shared.max_check_weight import (
    max_check_weight,
)
from search.filters.quantum_pairing._shared.min_classical_distance import (
    min_classical_distance,
)
from search.filters.quantum_pairing._shared.min_classical_girth import (
    min_classical_girth,
)
from search.filters.quantum_pairing._shared.same_group import same_group
from search.filters.quantum_pairing._shared.same_shape import same_shape


# ─────────────────────────────────────────────────────────────────
# Classical filter config + dispatcher
# ─────────────────────────────────────────────────────────────────


@dataclass
class ClassicalFilterConfig:
    """Threshold configuration for the classical filter stage.

    Each field defaults to "disabled". Set a threshold to enable.

    Group-agnostic (any G, any shape):
        min_girth_tanner_A_bin: reject if ``girth_tanner(A_bin) is not None``
            AND ``< threshold``.
        require_any_block_col_full_rank: reject if no ``ma``-subset of
            block-cols is invertible (i.e. no structured canonical basis).
        max_canonical_logical_weight: reject if the canonical G-orbit logical
            weight this side produces (the structured ``ker(M_bin)`` codeword
            weight; A-side -> Lz weight, B-side -> Lx weight) EXCEEDS the
            threshold. Prunes classical codes that would force heavy logical
            operators (heavy surgery / extractor gadgets). No-op when the
            threshold is >= the classical code length ``na*n`` (the max possible
            weight) — set it there to disable without removing the key. A side
            with no full-rank block-col has no canonical basis, so the filter
            passes it (defer to ``require_any_block_col_full_rank``).

    ma=1 only (raises ValueError if enabled with ma > 1):
        min_entry_order_bound: reject if the weight-2 bound is finite AND
            below threshold.
        min_abelianization_bound: reject if the abelianization-derived
            upper bound on ``d(A_bin)`` is below threshold. Non-abelian
            only.

    Abelian-only (raises ValueError if enabled with non-abelian G):
        min_base_girth_bound: reject if ``base_girth_bound(W) < threshold``.
            The relator identities behind the bound require commutativity,
            so this filter is abelian-only.
        min_weight_distance_bound: reject if the weight-only (integer
            permanent) bound is below threshold. Cheap first step.
        min_ring_distance_bound: reject if the permanent-based bound is
            below threshold. Tighter for J ≥ 2 but requires ring mult.
    """

    # Any G, any shape
    min_girth_tanner_A_bin: Optional[int] = None
    require_any_block_col_full_rank: bool = False

    # Any G (needs a full-rank block-col)
    max_canonical_logical_weight: Optional[int] = None

    # ma=1 only
    min_entry_order_bound: Optional[int] = None
    min_abelianization_bound: Optional[int] = None

    # Abelian only
    min_base_girth_bound: Optional[int] = None
    min_weight_distance_bound: Optional[int] = None
    min_ring_distance_bound: Optional[int] = None


def apply_classical_filters(
    A: list,
    A_bin,
    gd: GroupData,
    cfg: ClassicalFilterConfig,
) -> bool:
    """Apply enabled classical filters cheapest-first.

    Args:
        A: ring matrix ``(ma, na)``.
        A_bin: binary lift ``build_A_bin(A, gd)``.
        gd: GroupData.
        cfg: ClassicalFilterConfig.

    Returns:
        ``True`` iff every enabled filter passes.

    Raises:
        ValueError if a filter's precondition is violated by the inputs
        (e.g. ma=1 filter enabled with ma > 1, abelian-only filter with
        non-abelian G). These are configuration errors.
    """
    ma, na = len(A), len(A[0])
    n = gd.n

    # 1. base_girth_bound — weight matrix only, no GAP, cheapest.
    # Abelian only: the relator identities behind the bound require
    # commutativity.
    if cfg.min_base_girth_bound is not None:
        if not gd.is_abelian:
            raise ValueError(
                f"min_base_girth_bound requires abelian G; "
                f"got {gd.structure!r}"
            )
        W = [[len(A[ia][ja]) for ja in range(na)] for ia in range(ma)]
        if base_girth_bound(W) < cfg.min_base_girth_bound:
            return False

    # 2. entry_order_bound — group element-order table lookup (ma=1).
    if cfg.min_entry_order_bound is not None:
        if ma != 1:
            raise ValueError(
                f"min_entry_order_bound requires ma=1; got ma={ma}"
            )
        bound = entry_order_bound(A, gd)
        if bound is not None and bound < cfg.min_entry_order_bound:
            return False

    # 3. abelianization_bound — d(A_bin) ≤ total_entries · |[G,G]| (ma=1,
    # non-abelian only).
    if cfg.min_abelianization_bound is not None:
        if ma != 1:
            raise ValueError(
                f"min_abelianization_bound requires ma=1; got ma={ma}"
            )
        if gd.is_abelian:
            raise ValueError(
                "min_abelianization_bound is non-abelian only; use "
                "min_ring_distance_bound for abelian G."
            )
        if abelianization_bound(A, gd) < cfg.min_abelianization_bound:
            return False

    # 4. any_block_col_full_rank — `C(na, ma)` F2 ranks.
    if cfg.require_any_block_col_full_rank:
        if not any_block_col_full_rank(A_bin, n, ma=ma):
            return False

    # 5. girth_tanner on A_bin — BFS on Tanner graph.
    if cfg.min_girth_tanner_A_bin is not None:
        g = girth_tanner(A_bin)
        if g is not None and g < cfg.min_girth_tanner_A_bin:
            return False

    # 5b. canonical_logical_weight — the canonical G-orbit logical weight this side
    # produces (A -> Lz, B -> Lx); an F2 solve over the full-rank block, O(n^3).
    # Skip when the threshold can never bite, i.e. >= na*n = A_bin's column count =
    # the classical code length (also the max possible codeword weight).
    if (cfg.max_canonical_logical_weight is not None
            and cfg.max_canonical_logical_weight < A_bin.shape[1]):
        from search.filters.classical._shared.canonical_logical_weight import (
            canonical_logical_weight,
        )
        w = canonical_logical_weight(A_bin, n)
        if w is not None and w > cfg.max_canonical_logical_weight:
            return False

    # 6a. weight_distance_bound — integer permanent of W_A only. Cheap
    # abelian pre-filter (tight for J=1; loose for J≥2).
    if cfg.min_weight_distance_bound is not None:
        if not gd.is_abelian:
            raise ValueError(
                f"min_weight_distance_bound requires abelian G; "
                f"got {gd.structure!r}"
            )
        from core.classical_code import weight_matrix
        from search.filters.classical.abelian.weight_distance_bound import (
            weight_distance_bound,
        )
        if weight_distance_bound(weight_matrix(A), gd) < cfg.min_weight_distance_bound:
            return False

    # 6b. ring_distance_bound — permanent over F2[G] (abelian only). Most expensive.
    if cfg.min_ring_distance_bound is not None:
        if not gd.is_abelian:
            raise ValueError(
                f"min_ring_distance_bound requires abelian G; "
                f"got {gd.structure!r}"
            )
        from search.filters.classical.abelian.ring_distance_bound import (
            ring_distance_bound,
        )
        if ring_distance_bound(A, gd) < cfg.min_ring_distance_bound:
            return False

    return True


# ─────────────────────────────────────────────────────────────────
# Quantum-pairing filter config + dispatcher
# ─────────────────────────────────────────────────────────────────


@dataclass
class QuantumPairingFilterConfig:
    """Threshold config for the pair-stage quantum filters.

    Operates on metadata of two saved classical codes ``A`` and ``B``
    BEFORE building Hx / Hz. The dispatcher receives per-side metadata
    as dicts; for abelian searches (B = A) the caller passes the SAME
    dict for both sides.

    Each field defaults to "disabled" (None for thresholds, False for
    booleans).

    Identity / shape:
        require_same_group: ``True`` ⇒ reject pairs from different groups
            (compares ``meta["group_tag"]``).
        require_same_shape: ``True`` ⇒ reject pairs with different
            ``(ma, na)`` vs ``(mb, nb)``.

    Classical-side thresholds (scalar metadata):
        min_classical_distance: both sides' ``meta["dist"]`` must be
            ≥ threshold.
        min_classical_girth: both sides' ``meta["girth"]`` must be
            ≥ threshold (``None`` girth = forest → passes).

    Resulting Hx / Hz check-weight caps (uses both weight matrices):
        max_Hx_check_weight: derived ``Hx`` max row weight must be
            ≤ cap.
        max_Hz_check_weight: derived ``Hz`` max row weight must be
            ≤ cap.

    The check-weight caps reach into ``meta["weight_matrix"]`` for both
    sides; if you don't enable them, ``weight_matrix`` need not be set.

    Full-extractor feasibility (group-general; needs ``gd``):
        min_full_extractor_bridge_d: reject the pair unless its full extractor's
            fixed ``[d, 1, d]`` X/Z bridge can be placed — i.e. a length-``d``
            simple path of non-pivot X-only vertices exists in ``G_X`` AND of
            Z-only vertices in ``G_Z`` (the extractor kit's **condition C5**).
            Set the threshold to the target distance ``d``. This filter builds
            the canonical code + orbit logical basis from the pair's ring
            matrices, so it requires ``meta["matrix"]`` and a ``gd`` passed to the
            dispatcher (it raises ``ValueError`` if ``gd`` is missing). Works for
            both abelian (the pair's ``B`` is reconstructed as ``A*``) and
            non-abelian (independent ``A`` / ``B``) LP searches.
    """

    # Cheap identity gates
    require_same_group: bool = False
    require_same_shape: bool = False

    # Scalar classical thresholds
    min_classical_distance: Optional[int] = None
    min_classical_girth: Optional[int] = None

    # Resulting check-weight caps
    max_Hx_check_weight: Optional[int] = None
    max_Hz_check_weight: Optional[int] = None

    # Full-extractor (size-d X/Z bridge) feasibility — condition C5 (non-abelian)
    min_full_extractor_bridge_d: Optional[int] = None


def _required_meta_keys(cfg: QuantumPairingFilterConfig) -> list[tuple[str, str]]:
    """List ``(filter_name, meta_key)`` pairs required by the enabled filters.

    Used by :func:`apply_quantum_pairing_filters` to validate inputs up front
    before any filter logic runs. Returning the list (rather than just the
    set of keys) lets the error message attribute each missing key to the
    filter that needs it.
    """
    required: list[tuple[str, str]] = []
    if cfg.require_same_group:
        required.append(("require_same_group", "group_tag"))
    if cfg.require_same_shape:
        required.append(("require_same_shape", "shape"))
    if cfg.min_classical_girth is not None:
        required.append(("min_classical_girth", "girth"))
    if cfg.min_classical_distance is not None:
        required.append(("min_classical_distance", "dist"))
    if (cfg.max_Hx_check_weight is not None
            or cfg.max_Hz_check_weight is not None):
        required.append(("max_Hx/Hz_check_weight", "weight_matrix"))
    if cfg.min_full_extractor_bridge_d is not None:
        required.append(("min_full_extractor_bridge_d", "matrix"))
    return required


def apply_quantum_pairing_filters(
    meta_A: dict,
    meta_B: dict,
    cfg: QuantumPairingFilterConfig,
    *,
    gd: Optional[GroupData] = None,
) -> bool:
    """Apply enabled pair-stage filters. Cheapest-first short-circuit.

    Args:
        meta_A, meta_B: per-side metadata dicts. Recognized keys:
            ``"group_tag"`` (str), ``"shape"`` (tuple/list of 2 ints),
            ``"dist"`` (int or None), ``"girth"`` (int or None),
            ``"weight_matrix"`` (ndarray or list-of-lists of int),
            ``"matrix"`` (ring matrix, list-of-lists of int tuples — needed by
            ``min_full_extractor_bridge_d``).
            A key only needs to be present if the corresponding filter
            is enabled in ``cfg``. For abelian (B = A) pass the same dict
            for both arguments. Note: present-with-value-None is allowed
            and has filter-specific semantics (girth=None ⇒ forest ⇒ pass;
            dist=None ⇒ unknown ⇒ reject); a missing key is treated as a
            caller bug and rejected up front.
        cfg: QuantumPairingFilterConfig.
        gd: ``GroupData`` for the pair's group. Required (keyword-only) when
            ``cfg.min_full_extractor_bridge_d`` is enabled; unused otherwise.

    Returns:
        ``True`` iff every enabled filter passes.

    Raises:
        ValueError if any enabled filter needs a meta key that is absent
            from ``meta_A`` or ``meta_B`` (all missing keys reported together),
            or if ``min_full_extractor_bridge_d`` is enabled without a
            non-abelian ``gd``.
    """
    # Precondition: every key needed by an enabled filter must be present in
    # both sides. Collect all violations and report them in one error so the
    # caller can fix everything in one shot.
    missing: list[str] = []
    for filter_name, key in _required_meta_keys(cfg):
        if key not in meta_A:
            missing.append(f"meta_A[{key!r}] (needed by {filter_name})")
        if key not in meta_B:
            missing.append(f"meta_B[{key!r}] (needed by {filter_name})")
    if missing:
        raise ValueError(
            "apply_quantum_pairing_filters: missing required metadata keys: "
            + "; ".join(missing)
        )

    # 1. Same-group — string equality, O(1).
    if cfg.require_same_group:
        if not same_group(meta_A["group_tag"], meta_B["group_tag"]):
            return False

    # 2. Same-shape — tuple equality, O(1).
    if cfg.require_same_shape:
        if not same_shape(meta_A["shape"], meta_B["shape"]):
            return False

    # 3. Classical girth — value-None (forest) passes; key-missing already
    # rejected by the precondition above.
    if cfg.min_classical_girth is not None:
        if not min_classical_girth(
            meta_A["girth"], meta_B["girth"],
            cfg.min_classical_girth,
        ):
            return False

    # 4. Classical distance — value-None rejected (pessimistic); key-missing
    # already rejected by the precondition above.
    if cfg.min_classical_distance is not None:
        if not min_classical_distance(
            meta_A["dist"], meta_B["dist"],
            cfg.min_classical_distance,
        ):
            return False

    # 5. Quantum check-weight caps — derived from weight matrices.
    if (cfg.max_Hx_check_weight is not None
            or cfg.max_Hz_check_weight is not None):
        if not max_check_weight(
            meta_A["weight_matrix"], meta_B["weight_matrix"],
            max_Hx=cfg.max_Hx_check_weight,
            max_Hz=cfg.max_Hz_check_weight,
        ):
            return False

    # 6. Full-extractor (size-d X/Z bridge) feasibility — condition C5.
    #    Most expensive (builds the canonical code + orbit logical basis), so last.
    #    Group-general (abelian + non-abelian LP). The A/B rings are reconstructed
    #    exactly as the pairing phase builds the code: for abelian the convention is
    #    B = A* (elementwise dagger) with the same side passed twice, so we derive B
    #    from A; for non-abelian A and B are independent (meta_A / meta_B).
    if cfg.min_full_extractor_bridge_d is not None:
        if gd is None:
            raise ValueError(
                "min_full_extractor_bridge_d requires a GroupData `gd` argument."
            )
        from search.filters.quantum_pairing._shared.full_extractor_bridge import (
            full_extractor_bridge_from_rings,
        )

        def _ring(meta):
            return [[tuple(int(g) for g in entry) for entry in row]
                    for row in meta["matrix"]]

        A_ring = _ring(meta_A)
        if gd.is_abelian:
            from core.group import dagger
            B_ring = [[tuple(sorted(dagger(e, gd))) for e in row] for row in A_ring]
        else:
            B_ring = _ring(meta_B)

        if not full_extractor_bridge_from_rings(
            A_ring, B_ring, gd, cfg.min_full_extractor_bridge_d,
        ):
            return False

    return True
