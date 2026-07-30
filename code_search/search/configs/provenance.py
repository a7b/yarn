"""Provenance dict embedded in every saved classical / quantum JSON.

Each saved code carries a back-pointer to the config it came from plus
metadata about the Python entry point that wrote it::

    "provenance": {
        "phase": "classical",
        "phase_module": "search.phases.classical",
        "phase_function": "run_classical",
        "config_path": "<absolute path of the loaded YAML>" | null,
        "python_executable": sys.executable,
        "git_commit": "<sha>" | null
    }

This lets you trace any saved code back to the YAML config and Python
module that generated it without scanning the manifest. ``config_path``
is ``None`` if the :class:`SearchConfig` was built programmatically
rather than via :func:`load_config`; ``git_commit`` is ``None`` outside
a git checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


_GIT_COMMIT_CACHE: Optional[str] = None
_GIT_COMMIT_FAILED: bool = False


def build_provenance(cfg, *, phase: str, module: str,
                     function: str) -> dict:
    """Return a provenance dict ready to embed in a saved JSON."""
    return {
        "phase": phase,
        "phase_module": module,
        "phase_function": function,
        "config_path": _config_path(cfg),
        "python_executable": sys.executable,
        "git_commit": _git_commit(),
    }


def _config_path(cfg) -> Optional[str]:
    path = getattr(cfg, "source_path", None)
    return str(path) if path else None


def _git_commit() -> Optional[str]:
    """Current git commit SHA from the CWD's repo, ``None`` if unavailable.

    Cached per process — we only shell out once.
    """
    global _GIT_COMMIT_CACHE, _GIT_COMMIT_FAILED
    if _GIT_COMMIT_CACHE is not None:
        return _GIT_COMMIT_CACHE
    if _GIT_COMMIT_FAILED:
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        _GIT_COMMIT_CACHE = out.stdout.strip()
        return _GIT_COMMIT_CACHE
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        _GIT_COMMIT_FAILED = True
        return None
