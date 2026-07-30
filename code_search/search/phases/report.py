"""Report phase: render a Markdown summary of the saved quantum pool.

Reads every ``k*_dx*_dz*_*.json`` under ``quantum/``, sorts by
``(k descending, min(dx, dz) descending)`` with ``None`` (no codeword
found = best optimistic case) ranked as ``+infinity``, and writes a
Markdown table to ``group_dir / report.md``.

The function is pure: it returns the rendered Markdown string AND writes
the file. ``run_report(cfg, output=None)`` to skip the write.
"""

import json
import time
from pathlib import Path
from typing import Optional

from search.configs.config import SearchConfig
from search.configs.paths import (
    group_dir,
    quantum_dir,
    report_path,
)


def run_report(cfg: SearchConfig, *,
               output: Optional[Path] = None,
               top_n: Optional[int] = None) -> str:
    """Render and save a Markdown summary of the saved quantum codes.

    Args:
        cfg: parsed :class:`SearchConfig`.
        output: where to write the report. Defaults to
            :func:`report_path(cfg)`. Pass ``None`` to also skip the
            write by setting ``output=False``... actually ``None`` writes
            to the default path. Pass ``"-"`` (string) to **skip** writing
            and return the Markdown only.
        top_n: limit the table to this many rows. ``None`` shows all.

    Returns:
        The rendered Markdown string.
    """
    codes = _load_quantum_pool(quantum_dir(cfg))
    md = _render(cfg, codes, top_n=top_n)
    if output != "-":
        target = Path(output) if output is not None else report_path(cfg)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(md)
    return md


def _load_quantum_pool(qdir: Path) -> list:
    """Load every JSON under ``qdir`` (recursive) that looks like a quantum code."""
    if not qdir.exists():
        return []
    out: list = []
    for path in sorted(qdir.rglob("k*_dx*_dz*_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_path"] = str(path)
        out.append(data)
    return out


def _sort_key(code: dict):
    """Sort key: (k desc, min(dx, dz) desc). ``None`` distances rank as ∞."""
    k = code.get("k", 0)
    dx = code.get("dx")
    dz = code.get("dz")
    big = float("inf")
    dvals = [big if d is None else d for d in (dx, dz)]
    min_d = min(dvals) if dvals else 0
    return (-k, -min_d)


def _fmt(v) -> str:
    if v is None:
        return "—"
    return str(v)


def _render(cfg: SearchConfig, codes: list, *,
            top_n: Optional[int]) -> str:
    """Produce the Markdown body."""
    codes = sorted(codes, key=_sort_key)
    if top_n is not None:
        codes = codes[:top_n]

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    ma, na = cfg.shape

    lines: list = []
    lines += [
        f"# Search Report — {cfg.group.tag} / {ma}×{na}",
        "",
        f"Generated: {timestamp}",
        f"Search root: `{group_dir(cfg)}`",
        f"GAP expression: `{cfg.group.gap_expr}`",
        "",
        "## Summary",
        "",
        f"- Total quantum codes saved: **{len(codes)}**",
    ]
    # Estimator breakdown.
    from collections import Counter
    est_counts = Counter(c.get("estimator", "—") for c in codes)
    if est_counts:
        lines.append("- Estimator breakdown:")
        for est, count in est_counts.most_common():
            lines.append(f"    - `{est}`: {count}")

    lines += [
        "",
        "## Codes",
        "",
        "Sorted by `k` (descending), then by `min(dx, dz)` (descending). "
        "`None` distances (no codeword found in trials) are ranked highest "
        "as optimistic-PASS per the search-pipeline convention.",
        "",
        "| # | k | dx | dz | wA | wB | estimator | bposd_trials | sqetch_trials | timestamp |",
        "|---|---|----|----|----|----|-----------|--------------|---------------|-----------|",
    ]
    for i, c in enumerate(codes, 1):
        wA = _weight_tag_from_matrix(c.get("weight_A"))
        wB = _weight_tag_from_matrix(c.get("weight_B"))
        ts = c.get("timestamp", "—")
        lines.append(
            f"| {i} | {_fmt(c.get('k'))} | {_fmt(c.get('dx'))} | "
            f"{_fmt(c.get('dz'))} | {wA} | {wB} | "
            f"`{_fmt(c.get('estimator'))}` | "
            f"{_fmt(c.get('bposd_num_trials'))} | "
            f"{_fmt(c.get('sqetch_num_trials'))} | {ts} |"
        )

    return "\n".join(lines) + "\n"


def _weight_tag_from_matrix(W) -> str:
    """Render a weight matrix (list of lists) as the canonical
    ``r1c1.r1c2-r2c1.r2c2`` tag. ``None`` → ``'—'``."""
    if W is None:
        return "—"
    rows = [".".join(str(int(x)) for x in row) for row in W]
    return "-".join(rows)
