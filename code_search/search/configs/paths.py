"""Path helpers for the search pipeline.

Layout::

    {results_dir}/
        {group_tag}/
            {ma}x{na}/
                classical_A/
                    w{weight_tag}/
                        d{dist}_w{wtag}_t{trials}_{ts}.json
                classical_B/                     ← non-abelian only
                    w{weight_tag}/...
                quantum/
                    k{k}_dx{dx}_dz{dz}_..._{ts}.json
                manifest.json

``weight_tag`` formats a weight matrix as a filename-safe string:
entries joined by ``.``, rows joined by ``-``.
"""

from pathlib import Path


def weight_tag(W) -> str:
    """Format a weight matrix (numpy or list-of-lists of ints) as ``"r1c1.r1c2-r2c1.r2c2..."``.

    Examples:
        ``[[2, 2]]``          → ``"2.2"``
        ``[[2, 3], [3, 2]]``  → ``"2.3-3.2"``
        ``[[10, 2]]``         → ``"10.2"``
    """
    if hasattr(W, "tolist"):
        W = W.tolist()
    rows = [".".join(str(int(x)) for x in row) for row in W]
    return "-".join(rows)


def group_dir(cfg) -> Path:
    """``results_dir / group_tag / {ma}x{na}/``"""
    ma, na = cfg.shape
    return Path(cfg.results_dir) / cfg.group.tag / f"{ma}x{na}"


def classical_A_dir(cfg) -> Path:
    return group_dir(cfg) / "classical_A"


def classical_B_dir(cfg) -> Path:
    return group_dir(cfg) / "classical_B"


def quantum_dir(cfg) -> Path:
    return group_dir(cfg) / "quantum"


def manifest_path(cfg) -> Path:
    return group_dir(cfg) / "manifest.json"


def classical_filename(dist: int, wtag: str, num_trials: int,
                       timestamp: int) -> str:
    """``d{dist}_w{wtag}_t{num_trials}_{ts}.json``"""
    return f"d{dist}_w{wtag}_t{num_trials}_{timestamp}.json"


def quantum_filename(
    k: int, dx, dz, wA_tag: str, wB_tag: str,
    bposd_trials: int, timestamp: int,
    sqetch_trials: int = 0,
) -> str:
    """``k{k}_dx{dx}_dz{dz}_wA{wA}_wB{wB}_bposd{n}[_sqetch{n}]_{ts}.json``

    ``dx``/``dz`` may be ``None`` (= no codeword found in the budget); they
    are rendered as ``"X"`` in that case so filenames stay sortable.
    """
    def _d(x):
        return "X" if x is None else str(int(x))
    parts = [
        f"k{k}",
        f"dx{_d(dx)}",
        f"dz{_d(dz)}",
        f"wA{wA_tag}",
        f"wB{wB_tag}",
        f"bposd{bposd_trials}",
    ]
    if sqetch_trials > 0:
        parts.append(f"sqetch{sqetch_trials}")
    parts.append(str(timestamp))
    return "_".join(parts) + ".json"


def tried_pairs_path(cfg) -> Path:
    """``quantum_dir / tried_pairs.json`` — JSON array of ``[a_path, b_path]`` pairs already evaluated."""
    return quantum_dir(cfg) / "tried_pairs.json"


def report_path(cfg) -> Path:
    """``group_dir / report.md`` — Markdown report of the saved quantum pool."""
    return group_dir(cfg) / "report.md"
