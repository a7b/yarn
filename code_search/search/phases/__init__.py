"""Search pipeline phases.

Each phase consumes a :class:`search.configs.SearchConfig` and either
produces saved JSON artifacts (``classical``, ``pairing``) or a
markdown report (``report``).

Available in Chunk 1: classical (non-abelian only).
"""

from .classical import run_classical
from .pairing import run_pairing
from .report import run_report

__all__ = ["run_classical", "run_pairing", "run_report"]
