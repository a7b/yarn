"""CLI runner: ``python -m search.runners.search <config.yaml> [--stages ...]``.

Loads a YAML config and dispatches on its ``mode``:

- ``pipeline`` (default): run each requested stage (classical → pairing →
  report) via the ``search.phases.run_*`` entry points. ``--stages``
  overrides ``cfg.run_stages``.
- ``canonical``: run the order-band campaign via
  :func:`search.canonical.campaign.run_campaign` (``--stages`` does not
  apply and is rejected).

``--quiet`` suppresses progress prints in both modes (errors still raise).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from search.configs.config import CanonicalConfig
from search.configs.loader import load_config
from search.phases.classical import run_classical
from search.phases.pairing import run_pairing
from search.phases.report import run_report


_STAGE_DISPATCH = {
    "classical": ("run_classical", run_classical),
    "pairing":   ("run_pairing", run_pairing),
    "report":    ("run_report", run_report),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: optional argument list (uses ``sys.argv[1:]`` if ``None``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="python -m search.runners.search",
        description="Run a QLDPC search pipeline from a YAML config.",
    )
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument(
        "--stages", nargs="+", default=None,
        choices=tuple(_STAGE_DISPATCH.keys()),
        help="Override the stages listed in cfg.run_stages.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-stage summary prints.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config_path)

    if isinstance(cfg, CanonicalConfig):
        if args.stages:
            print("--stages applies to pipeline mode only; a canonical "
                  "campaign has no stages to select.", file=sys.stderr)
            return 2
        return _run_canonical(cfg, args)

    stages = args.stages if args.stages else cfg.run_stages

    if not args.quiet:
        print(f"Loaded config: {args.config_path}")
        print(f"  Group: {cfg.group.gap_expr} (tag={cfg.group.tag})")
        print(f"  Shape: {cfg.shape}")
        print(f"  Stages: {stages}")
        print()

    for stage in stages:
        if stage not in _STAGE_DISPATCH:
            print(f"Unknown stage: {stage!r}", file=sys.stderr)
            return 1
        name, func = _STAGE_DISPATCH[stage]
        if not args.quiet:
            print(f"=== {name}({stage}) ===")
        try:
            result = func(cfg)
        except Exception as e:
            print(f"[{stage}] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise
        if not args.quiet:
            _print_summary(stage, result)
            print()

        if (stage == "pairing" and cfg.report.auto_report
                and "report" not in stages):
            if not args.quiet:
                print("=== run_report(auto_report) ===")
            report_result = run_report(cfg)
            if not args.quiet:
                _print_summary("report", report_result)
                print()

    return 0


def _run_canonical(cfg: CanonicalConfig, args) -> int:
    """Run a ``mode: canonical`` search (lazy import: needs GAP)."""
    from search.canonical.driver import run_canonical

    if not args.quiet:
        print(f"Loaded canonical search: {args.config_path}")
        print(f"  Group: {cfg.group}  target: {cfg.params.target}")
        print(f"  Out dir: {cfg.out_dir}")
        print()

    result = run_canonical(cfg, log=None if args.quiet else print)

    if not args.quiet:
        print()
        print(f"Done: {result['group']} — {result['n_passed']} codes passed "
              f"(verdict: {result['verdict']}).")
        print(f"Summary: {Path(cfg.out_dir) / 'canonical_summary.json'}")
    return 0


def _print_summary(stage: str, result) -> None:
    """Compact one-line recap of a stage's return value."""
    if stage == "classical":
        n_a = len(result.get("new_A", []))
        n_b = len(result.get("new_B", []))
        print(f"  Saved {n_a} new A-side, {n_b} new B-side classical codes.")
    elif stage == "pairing":
        n_new = len(result.get("new_quantum", []))
        print(
            f"  Tried {result.get('n_pairs_tried', 0)} pairs, "
            f"{result.get('n_pairs_passed', 0)} passed, "
            f"{n_new} quantum codes saved."
        )
    elif stage == "report":
        # run_report returns the markdown string.
        n_lines = result.count("\n") if isinstance(result, str) else 0
        print(f"  Report written ({n_lines} lines).")
    else:
        print(f"  result = {result!r}")


if __name__ == "__main__":
    raise SystemExit(main())
