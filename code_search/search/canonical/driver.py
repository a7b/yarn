"""Driver for ``mode: canonical`` YAML configs — one config, one group.

Consumes a :class:`search.configs.config.CanonicalConfig` (see
``search/configs/examples/canonical_sample.yaml``): builds the group named
by the YAML and runs the randomized, budgeted funnel
(:func:`search.canonical.sample_search.run_group_sweep`) once. There is no
group enumeration here — helpers for scanning group families
programmatically live in :mod:`search.canonical.groups`, and the
exhaustive-pool funnel (:func:`search.canonical.run_group.run_group`)
remains available as a Python API.

Artifacts: the funnel writes its own stats + passed-code dirs into
``out_dir``; the driver additionally writes ``out_dir/canonical_summary.json``
with the run verdict and provenance.

GAP is required (to construct the group); the funnel needs a CUDA GPU
(sqetch).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from search.canonical.groups import GroupSpec
from search.canonical.sample_search import run_group_sweep
from search.configs.config import CanonicalConfig
from search.configs.loader import auto_group_tag
from search.configs.provenance import _git_commit

_SMALL_GROUP_RE = re.compile(r"^SmallGroup\(\s*(\d+)\s*,\s*(\d+)\s*\)$")


@dataclass(frozen=True)
class FixedGroupSpec:
    """A group given directly by a GAP expression (duck-types GroupSpec)."""

    gap_expr: str
    tag: str


def spec_for(gap_expr: str):
    """Group spec for the YAML's group expression.

    ``SmallGroup(o,i)`` becomes a :class:`GroupSpec` (keeping the
    ``Sg<o>_<i>`` tag convention of the funnels); any other GAP expression
    gets a filename-safe auto tag.
    """
    m = _SMALL_GROUP_RE.match(gap_expr)
    if m:
        return GroupSpec(order=int(m.group(1)), small_group_id=int(m.group(2)))
    return FixedGroupSpec(gap_expr=gap_expr, tag=auto_group_tag(gap_expr))


def run_canonical(cfg: CanonicalConfig, *,
                  log: Optional[Callable] = None) -> dict:
    """Run the funnel for the configured group; return (and persist) a summary."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log = log if log is not None else (lambda *a, **k: None)

    provenance = {
        "mode": "canonical",
        "entry_point": "search.canonical.driver.run_canonical",
        "config_path": (str(cfg.source_path)
                        if cfg.source_path is not None else None),
        "git_commit": _git_commit(),
    }

    spec = spec_for(cfg.group)
    _log(f"{spec.tag}: running canonical funnel")
    summary = run_group_sweep(spec, cfg.params, out_dir,
                              provenance=provenance, log=log)
    n_passed = len(summary.get("passed", []))
    _log(f"{spec.tag}: {n_passed} passed")

    result = {
        "target": cfg.params.target,
        "group": spec.tag,
        "gap_expr": spec.gap_expr,
        "n_passed": n_passed,
        "verdict": summary.get("verdict"),
        "provenance": provenance,
    }
    (out_dir / "canonical_summary.json").write_text(
        json.dumps(result, indent=2))
    return result
