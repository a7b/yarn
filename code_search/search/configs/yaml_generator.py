"""Auto-generate a YAML config skeleton tailored to a specific group + shape.

The generated YAML lists ONLY the filters / fields that apply for the
chosen ``(group, shape)`` — abelian-only filters are skipped on
non-abelian groups, ``ma=1``-only filters are skipped on ``ma > 1``,
``avoid_same_coset`` only shows up for non-abelian groups, etc. Defaults
are conservative (most filters disabled = ``null``) so the user can
selectively enable / tweak / delete lines by hand.

Entry points::

    generate_search_yaml(gap_expr, shape, *, group_tag=None,
                         include_pairing=True) -> str
    write_search_yaml(path, gap_expr, shape, *, ...) -> Path

CLI::

    python -m search.runners.new_config <gap_expr> <ma> <na> [--output OUT.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from core.group import GroupData
from search.configs.loader import auto_group_tag


# Sentinel kept on the YAML disabled lines.
_DISABLED = "null"


def generate_search_yaml(
    gap_expr: str,
    shape: tuple,
    *,
    group_tag: Optional[str] = None,
    include_pairing: bool = True,
) -> str:
    """Render a YAML config skeleton string.

    Args:
        gap_expr: GAP expression for the group (e.g. ``"SymmetricGroup(3)"``).
        shape: ``(ma, na)``.
        group_tag: optional short tag for filenames. Defaults to the result
            of :func:`auto_group_tag`.
        include_pairing: include the pairing-stage section. Set ``False``
            to emit a classical-only YAML.

    Returns:
        A YAML string (UTF-8 text).
    """
    ma, na = shape
    if ma < 1 or na < 1:
        raise ValueError(f"shape must be positive; got {shape}")

    gd = GroupData(gap_expr)
    is_abelian = gd.is_abelian
    tag = group_tag or auto_group_tag(gap_expr)

    lines: list = []
    lines += _top_lines(gap_expr, tag, shape, is_abelian, include_pairing)
    lines += [""]
    lines += _classical_lines(gap_expr, shape, is_abelian, gd.n)
    if include_pairing:
        lines += [""]
        lines += _pairing_lines()
    lines += [""]
    lines += _report_lines()
    lines += [""]
    lines += [
        "results_dir: search_results",
        "verbose: false",
    ]
    return "\n".join(lines) + "\n"


def write_search_yaml(
    path: Union[str, Path],
    gap_expr: str,
    shape: tuple,
    *,
    group_tag: Optional[str] = None,
    include_pairing: bool = True,
) -> Path:
    """Render and write a YAML config. Returns the resolved path."""
    text = generate_search_yaml(
        gap_expr, shape, group_tag=group_tag,
        include_pairing=include_pairing,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


# ─────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────


def _top_lines(gap_expr, tag, shape, is_abelian, include_pairing):
    ma, na = shape
    flavor = "abelian" if is_abelian else "non-abelian"
    stages = ("[classical, pairing, report]" if include_pairing
              else "[classical]")
    return [
        f"# auto-generated search config — {flavor} group {gap_expr}, shape {ma}x{na}",
        "# Edit fields by hand; set a filter to a number to enable, or delete its line.",
        "",
        f"shape: [{ma}, {na}]",
        "group:",
        f"  gap_expr: {gap_expr}",
        f"  tag: {tag}",
        f"run_stages: {stages}",
    ]


def _classical_lines(gap_expr, shape, is_abelian, n):
    ma, na = shape
    out: list = ["classical:"]

    if is_abelian:
        out += [
            "  # Abelian: two-stage weight-pattern enumeration.",
            "  weight_pattern:",
            "    entry_min: 0                        # min integer per entry (set 1 to forbid 0)",
            "    entry_max: 3                        # max integer per entry",
            "    num_weight_samples: 100             # weight matrices to try",
            "    ring_samples_per_weight: 10         # ring placements per surviving W",
            "    max_row_weight: null",
            "    max_col_weight: null",
            "    min_base_girth_bound: null          # pattern-level (abelian)",
        ]
        if ma == 1:
            out += [
                "    min_weight_distance_bound: null   # pattern-level (ma=1, abelian)",
            ]
    else:
        out += [
            "  # Non-abelian: fixed per-entry weight matrices for A and B.",
            f"  weight_A: {_default_weight_matrix(ma, na)}   # REQUIRED: tune per-entry weights",
            f"  weight_B: {_default_weight_matrix(ma, na)}",
        ]

    out += [
        "",
        "  sampling:",
        "    total_samples: 1000",
        "    seed: 42",
        "    include_identity: true",
        "    min_element_order: 1",
    ]
    if not is_abelian:
        out += [
            "    avoid_same_coset: false             # non-abelian only",
        ]
    out += [
        "    max_tries: 1000",
        "    canonicalize: true",
    ]

    # Filters: emit only those that apply.
    out += [
        "",
        "  filters:",
        f"    min_girth_tanner_A_bin: {_DISABLED}",
        f"    require_any_block_col_full_rank: false",
        f"    max_canonical_logical_weight: {na * n}   "
        f"# = na*n (no-op default = classical code length; lower to prune heavy logicals)",
    ]
    if ma == 1:
        out += [
            f"    min_entry_order_bound: {_DISABLED}       # ma=1 only",
        ]
        if not is_abelian:
            out += [
                f"    min_abelianization_bound: {_DISABLED}    # ma=1, non-abelian only",
            ]
    if is_abelian:
        out += [
            f"    min_base_girth_bound: {_DISABLED}        # abelian only",
            f"    min_weight_distance_bound: {_DISABLED}   # abelian only",
            f"    min_ring_distance_bound: {_DISABLED}     # abelian only",
        ]

    out += [
        "",
        "  distance:",
        "    d_target: 6",
        "    num_trials: 2000",
        "    n_workers: 1                        # 1 = inline; >1 pays off for large n",
        "    osd_order: 5",
        "",
        "  pool:",
        "    min_pool_size: 0",
        "    max_saved: 0",
        "    force_new: false",
    ]
    return out


def _pairing_lines():
    return [
        "pairing:",
        "  bposd:",
        "    d_target: 6",
        "    num_trials: 2000",
        "    n_workers: 1",
        "    osd_order: 0",
        "",
        "  sqetch_verify:",
        "    enabled: false",
        "    d_target: null",
        "    num_trials: 100000",
        "    devices: null                          # null = all visible CUDA devices",
        "    strategy: auto",
        "    k_sub: 64",
        "    batch_size: 50000",
        "",
        "  filters:",
        "    require_same_group: true",
        "    require_same_shape: true",
        "    min_classical_distance: null",
        "    min_classical_girth: null",
        "    max_Hx_check_weight: null",
        "    max_Hz_check_weight: null",
        "",
        "  pool:",
        "    pair_mode: full_pool                    # ('new_only' is not implemented yet)",
        "    max_pairs: 2000                         # bounded by default; null = exhaustive cross-product",
        "    min_quantum_pool_size: 10               # stop once the pool holds this many (0 = never)",
    ]


def _report_lines():
    return [
        "report:",
        "  auto_report: false",
    ]


def _default_weight_matrix(ma, na) -> str:
    """``[[3, 3, ...], ...]`` rendered as YAML inline list.

    Weight-3 entries are a sane starting point for LP searches; the user
    is expected to tune them (0 disables an entry).
    """
    row = "[" + ", ".join("3" for _ in range(na)) + "]"
    rows = ", ".join(row for _ in range(ma))
    return f"[{rows}]"


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────


def _cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m search.runners.new_config",
        description="Auto-generate a search YAML config tailored to a group + shape.",
    )
    parser.add_argument("gap_expr", help="GAP expression for the group, "
                        "e.g. 'SymmetricGroup(3)' or "
                        "'DirectProduct(CyclicGroup(5), SymmetricGroup(3))'.")
    parser.add_argument("ma", type=int)
    parser.add_argument("na", type=int)
    parser.add_argument("--output", "-o", help="Output path. Default: stdout.")
    parser.add_argument("--tag", help="Override the auto-derived group tag.")
    parser.add_argument("--no-pairing", action="store_true",
                        help="Skip the pairing-stage section.")
    args = parser.parse_args(argv)

    text = generate_search_yaml(
        args.gap_expr, (args.ma, args.na),
        group_tag=args.tag,
        include_pairing=not args.no_pairing,
    )
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    _cli()
