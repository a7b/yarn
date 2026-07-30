"""YAML → SearchConfig loader.

Maps a nested dict (from ``yaml.safe_load``) onto the dataclasses in
:mod:`search.configs.config`. Unknown top-level keys raise ``ValueError``.

Group tag derivation
--------------------
If ``group.tag`` is not provided, derive a short tag from ``gap_expr``:
- ``SymmetricGroup(n)`` → ``Sn``
- ``CyclicGroup(n)`` → ``Cn``
- ``DihedralGroup(n)`` → ``Dn``
- ``AlternatingGroup(n)`` → ``An``
- ``DirectProduct(A, B, ...)`` → ``Atag_x_Btag_x_...`` (recursive)
- Other expressions → sanitized GAP string.
"""

import dataclasses
import re
from pathlib import Path
from typing import Union

import yaml

from .config import (
    BPOSDConfig,
    CanonicalConfig,
    ClassicalDistanceConfig,
    ClassicalFiltersConfig,
    ClassicalStageConfig,
    SqetchVerifyConfig,
    GroupConfig,
    PairingFiltersConfig,
    PairingPoolConfig,
    PairingStageConfig,
    PoolConfig,
    ReportStageConfig,
    SamplingConfig,
    SearchConfig,
    WeightPatternConfig,
)


def load_config(path: Union[str, Path]):
    """Parse a YAML file into a config object.

    The top-level ``mode`` key selects the schema: ``pipeline`` (default,
    may be omitted) → :class:`SearchConfig`; ``canonical`` →
    :class:`CanonicalConfig`. Also records ``cfg.source_path`` so saved
    JSONs can back-link to the config that wrote them.
    """
    resolved = Path(path).resolve()
    raw = yaml.safe_load(resolved.read_text())
    if raw is None:
        raise ValueError(f"YAML file is empty: {path}")
    mode = raw.get("mode", "pipeline")
    if mode == "pipeline":
        cfg = from_dict(raw)
    elif mode == "canonical":
        cfg = canonical_from_dict(raw)
    else:
        raise ValueError(
            f"unknown mode {mode!r}; valid modes: 'pipeline' (default), "
            f"'canonical'"
        )
    cfg.source_path = resolved
    return cfg


def from_dict(raw: dict) -> SearchConfig:
    """Build a :class:`SearchConfig` from a parsed YAML dict."""
    _check_keys(raw, {
        "mode", "shape", "group", "run_stages", "classical", "pairing",
        "report", "results_dir", "verbose",
    }, where="top-level")
    if raw.get("mode", "pipeline") != "pipeline":
        raise ValueError(
            f"from_dict builds pipeline configs; got mode={raw['mode']!r} "
            f"(use load_config / canonical_from_dict)"
        )

    if "shape" not in raw:
        raise ValueError("top-level config missing required key 'shape'")
    if "group" not in raw:
        raise ValueError("top-level config missing required key 'group'")
    if "run_stages" not in raw:
        raise ValueError("top-level config missing required key 'run_stages'")
    if "classical" not in raw:
        raise ValueError("top-level config missing required key 'classical'")

    shape = tuple(raw["shape"])
    if len(shape) != 2:
        raise ValueError(f"shape must have length 2; got {shape}")

    run_stages = list(raw["run_stages"])
    valid_stages = {"classical", "pairing", "report"}
    bad_stages = [s for s in run_stages if s not in valid_stages]
    if bad_stages:
        raise ValueError(
            f"run_stages contains unknown stage(s) {bad_stages}; "
            f"valid: {sorted(valid_stages)}"
        )

    group = _group_from_dict(raw["group"])
    if group.tag is None:
        group.tag = auto_group_tag(group.gap_expr)

    classical = _classical_from_dict(raw["classical"])
    ma, na = shape
    for name, W in (("weight_A", classical.weight_A),
                    ("weight_B", classical.weight_B)):
        if W is not None and (
                len(W) != ma or any(len(row) != na for row in W)):
            raise ValueError(
                f"classical.{name} must be a {ma}x{na} matrix matching the "
                f"top-level shape {list(shape)}; got "
                f"{len(W)}x{len(W[0]) if W else 0}"
            )

    return SearchConfig(
        shape=shape,
        group=group,
        run_stages=run_stages,
        classical=classical,
        pairing=(_pairing_from_dict(raw["pairing"])
                 if "pairing" in raw and raw["pairing"] is not None
                 else None),
        report=(_report_from_dict(raw["report"])
                if "report" in raw and raw["report"] is not None
                else ReportStageConfig()),
        results_dir=Path(raw.get("results_dir", "search_results")),
        verbose=bool(raw.get("verbose", False)),
    )


def auto_group_tag(gap_expr: str) -> str:
    """Short, filename-safe tag for a GAP group expression.

    Examples:
        ``SymmetricGroup(3)`` → ``"S3"``
        ``CyclicGroup(4)`` → ``"C4"``
        ``DirectProduct(SymmetricGroup(3), CyclicGroup(5))`` → ``"S3_x_C5"``
    """
    expr = gap_expr.strip()
    simple = {
        "SymmetricGroup": "S",
        "CyclicGroup": "C",
        "DihedralGroup": "D",
        "AlternatingGroup": "A",
    }
    for fn, prefix in simple.items():
        m = re.fullmatch(rf"{fn}\((\d+)\)", expr)
        if m:
            return f"{prefix}{m.group(1)}"

    if expr.startswith("DirectProduct(") and expr.endswith(")"):
        inner = expr[len("DirectProduct("):-1]
        parts = _split_top_level_args(inner)
        return "_x_".join(auto_group_tag(p) for p in parts)

    # Fallback: sanitize.
    return (expr.replace(" ", "")
                .replace(",", "_")
                .replace("(", "_")
                .replace(")", "")
                .replace("__", "_")
                .strip("_"))


def _split_top_level_args(s: str) -> list:
    """Split ``s`` on top-level commas, respecting parenthesis depth."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _group_from_dict(d: dict) -> GroupConfig:
    _check_keys(d, {"gap_expr", "tag"}, where="group")
    if "gap_expr" not in d:
        raise ValueError("group config missing 'gap_expr'")
    return GroupConfig(gap_expr=d["gap_expr"], tag=d.get("tag"))


def _classical_from_dict(d: dict) -> ClassicalStageConfig:
    _check_keys(
        d,
        {"weight_A", "weight_B", "weight_pattern", "sampling", "filters",
         "distance", "pool"},
        where="classical",
    )
    if "sampling" not in d:
        raise ValueError("classical config missing 'sampling'")
    if "distance" not in d:
        raise ValueError("classical config missing 'distance'")
    # Exactly one of (weight_A) or (weight_pattern) must be present —
    # non-abelian uses fixed weight_A/weight_B, abelian uses weight_pattern.
    has_fixed = "weight_A" in d
    has_pattern = "weight_pattern" in d
    if has_fixed and has_pattern:
        raise ValueError(
            "classical config has BOTH 'weight_A' and 'weight_pattern' — "
            "pick one. Use 'weight_A'/'weight_B' for non-abelian searches "
            "and 'weight_pattern' for abelian."
        )
    if not has_fixed and not has_pattern:
        raise ValueError(
            "classical config missing both 'weight_A' (non-abelian) and "
            "'weight_pattern' (abelian) — supply exactly one."
        )
    return ClassicalStageConfig(
        weight_A=d.get("weight_A"),
        weight_B=d.get("weight_B"),
        weight_pattern=(_weight_pattern_from_dict(d["weight_pattern"])
                        if has_pattern else None),
        sampling=_sampling_from_dict(d["sampling"]),
        filters=(_filters_from_dict(d["filters"])
                 if "filters" in d and d["filters"] is not None
                 else ClassicalFiltersConfig()),
        distance=_distance_from_dict(d["distance"]),
        pool=(_pool_from_dict(d["pool"])
              if "pool" in d and d["pool"] is not None
              else PoolConfig()),
    )


def _weight_pattern_from_dict(d: dict) -> WeightPatternConfig:
    _check_keys(
        d,
        {"entry_max", "entry_min", "num_weight_samples", "ring_samples_per_weight",
         "max_row_weight", "max_col_weight",
         "min_base_girth_bound", "min_weight_distance_bound"},
        where="classical.weight_pattern",
    )
    for k in ("entry_max", "num_weight_samples", "ring_samples_per_weight"):
        if k not in d:
            raise ValueError(f"classical.weight_pattern missing required '{k}'")
    return WeightPatternConfig(
        entry_max=int(d["entry_max"]),
        entry_min=int(d.get("entry_min", 0)),
        num_weight_samples=int(d["num_weight_samples"]),
        ring_samples_per_weight=int(d["ring_samples_per_weight"]),
        max_row_weight=d.get("max_row_weight"),
        max_col_weight=d.get("max_col_weight"),
        min_base_girth_bound=d.get("min_base_girth_bound"),
        min_weight_distance_bound=d.get("min_weight_distance_bound"),
    )


def _sampling_from_dict(d: dict) -> SamplingConfig:
    _check_keys(
        d, {"total_samples", "seed", "include_identity", "min_element_order",
            "avoid_same_coset", "max_tries", "canonicalize"},
        where="classical.sampling",
    )
    if "total_samples" not in d:
        raise ValueError("classical.sampling missing 'total_samples'")
    return SamplingConfig(
        total_samples=int(d["total_samples"]),
        seed=d.get("seed"),
        include_identity=bool(d.get("include_identity", True)),
        min_element_order=int(d.get("min_element_order", 1)),
        avoid_same_coset=bool(d.get("avoid_same_coset", False)),
        max_tries=int(d.get("max_tries", 1000)),
        canonicalize=bool(d.get("canonicalize", True)),
    )


def _filters_from_dict(d: dict) -> ClassicalFiltersConfig:
    allowed = {
        "min_base_girth_bound", "min_girth_tanner_A_bin",
        "require_any_block_col_full_rank", "max_canonical_logical_weight",
        "min_entry_order_bound", "min_abelianization_bound",
        "min_weight_distance_bound", "min_ring_distance_bound",
    }
    _check_keys(d, allowed, where="classical.filters")
    return ClassicalFiltersConfig(**{k: d.get(k) for k in allowed
                                     if k in d})


def _distance_from_dict(d: dict) -> ClassicalDistanceConfig:
    _check_keys(
        d, {"d_target", "num_trials", "n_workers", "osd_order"},
        where="classical.distance",
    )
    if "d_target" not in d:
        raise ValueError("classical.distance missing 'd_target'")
    if "num_trials" not in d:
        raise ValueError("classical.distance missing 'num_trials'")
    return ClassicalDistanceConfig(
        d_target=int(d["d_target"]),
        num_trials=int(d["num_trials"]),
        n_workers=int(d.get("n_workers", 1)),
        osd_order=int(d.get("osd_order", 5)),
    )


def _pool_from_dict(d: dict) -> PoolConfig:
    _check_keys(
        d, {"min_pool_size", "max_saved", "force_new"},
        where="classical.pool",
    )
    return PoolConfig(
        min_pool_size=int(d.get("min_pool_size", 0)),
        max_saved=int(d.get("max_saved", 0)),
        force_new=bool(d.get("force_new", False)),
    )


def _pairing_filters_from_dict(d: dict) -> PairingFiltersConfig:
    allowed = {
        "require_same_group", "require_same_shape",
        "min_classical_distance", "min_classical_girth",
        "max_Hx_check_weight", "max_Hz_check_weight",
        "min_full_extractor_bridge_d",
    }
    _check_keys(d, allowed, where="pairing.filters")
    return PairingFiltersConfig(**{k: d.get(k) for k in allowed if k in d})


def _pairing_from_dict(d: dict) -> PairingStageConfig:
    _check_keys(
        d, {"bposd", "sqetch_verify", "filters", "pool"}, where="pairing",
    )
    if "bposd" not in d:
        raise ValueError("pairing config missing 'bposd'")
    bp_in = d["bposd"] or {}
    _check_keys(bp_in, {"d_target", "num_trials", "n_workers", "osd_order"},
                where="pairing.bposd")
    for req in ("d_target", "num_trials"):
        if req not in bp_in:
            raise ValueError(f"pairing.bposd missing required key '{req}'")
    sv_in = d.get("sqetch_verify") or {}
    _check_keys(sv_in, {"enabled", "d_target", "num_trials", "devices",
                        "strategy", "k_sub", "batch_size"},
                where="pairing.sqetch_verify")
    return PairingStageConfig(
        bposd=BPOSDConfig(
            d_target=int(bp_in["d_target"]),
            num_trials=int(bp_in["num_trials"]),
            n_workers=int(bp_in.get("n_workers", 1)),
            osd_order=int(bp_in.get("osd_order", 0)),
        ),
        sqetch_verify=SqetchVerifyConfig(
            enabled=bool(sv_in.get("enabled", False)),
            d_target=sv_in.get("d_target"),
            num_trials=int(sv_in.get("num_trials", 100_000)),
            devices=sv_in.get("devices"),
            strategy=str(sv_in.get("strategy", "auto")),
            k_sub=int(sv_in.get("k_sub", 64)),
            batch_size=int(sv_in.get("batch_size", 50_000)),
        ),
        filters=(_pairing_filters_from_dict(d["filters"])
                 if "filters" in d and d["filters"] is not None
                 else PairingFiltersConfig()),
        pool=(_pairing_pool_from_dict(d["pool"])
              if "pool" in d and d["pool"] is not None
              else PairingPoolConfig()),
    )


def _pairing_pool_from_dict(d: dict) -> PairingPoolConfig:
    _check_keys(d, {"pair_mode", "max_pairs", "min_quantum_pool_size"},
                where="pairing.pool")
    return PairingPoolConfig(
        pair_mode=str(d.get("pair_mode", "full_pool")),
        max_pairs=d.get("max_pairs"),
        min_quantum_pool_size=int(d.get("min_quantum_pool_size", 0)),
    )


def _report_from_dict(d: dict) -> ReportStageConfig:
    _check_keys(d, {"auto_report"}, where="report")
    return ReportStageConfig(auto_report=bool(d.get("auto_report", False)))


def _check_keys(d: dict, allowed: set, where: str) -> None:
    bad = set(d.keys()) - allowed
    if bad:
        raise ValueError(
            f"Unknown key(s) in {where} config: {sorted(bad)}; "
            f"allowed: {sorted(allowed)}."
        )


# ─────────────────────────────────────────────────────────────────
# mode: canonical
# ─────────────────────────────────────────────────────────────────


def canonical_from_dict(raw: dict) -> CanonicalConfig:
    """Build a :class:`CanonicalConfig` from a parsed YAML dict.

    ``group`` must be ONE explicit GAP group expression — one config = one
    group (no lists, no enumeration; to search several groups, write
    several configs).

    The ``params`` section is validated strictly against — and constructed
    as — :class:`search.canonical.sample_search.SweepParams`, the native
    knob dataclass of the randomized funnel, so every knob it documents is
    a YAML knob, with no drift.

    Imports the canonical modules lazily: they pull in the GAP-backed
    stack, which pipeline-mode users should never pay for.
    """
    _check_keys(raw, {
        "mode", "group", "params", "out_dir", "verbose",
    }, where="top-level (canonical)")
    for req in ("group", "params"):
        if req not in raw:
            raise ValueError(f"canonical config missing required key '{req}'")

    group_in = raw["group"]
    if not isinstance(group_in, str) or not group_in.strip():
        raise ValueError(
            f"group must be ONE GAP group expression string, e.g. "
            f"group: SmallGroup(84,11) — one config searches one group; "
            f"got {group_in!r}"
        )

    from search.canonical.sample_search import SweepParams as params_cls

    params_in = raw["params"] or {}
    allowed = {f.name for f in dataclasses.fields(params_cls)}
    _check_keys(params_in, allowed, where="params (canonical)")
    if "target" not in params_in:
        raise ValueError("canonical params missing required key 'target'")
    params = params_cls(**params_in)

    return CanonicalConfig(
        group=group_in.strip(),
        params=params,
        out_dir=Path(raw.get("out_dir", "canonical_results")),
        verbose=bool(raw.get("verbose", False)),
    )
