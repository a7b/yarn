"""Family base interface.

A *family* is a structurally constrained LP-code construction — the
ring-matrix entries follow a specific recipe (Hankel strip, self-dagger
constraints, etc.) instead of being independently sampled. Each family
exposes:

- A short ``name``.
- A ``shape() -> (ma, na)`` — the base-matrix shape implied by the family.
- A ``required_group_type()`` — one of ``"abelian"``, ``"non_abelian"``,
  or ``"any"`` — which the phase code can check before invoking the family.
- A ``sample(gd, rng, **kwargs)`` method returning ``(A, B)`` ring
  matrices, or ``None`` on rejection.

Family-specific *filters* live alongside the family module (as plain
functions); they're applied inside the sampler's rejection loop.

The classical phase will branch on ``cfg.family`` (added by config
integration in a follow-up) — when set, the phase calls ``family.sample``
instead of the generic ``random_ring_matrix``. Generic ring-level
filters from ``ClassicalFiltersConfig`` still apply on top of the
family-specific ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from core.group import GroupData


class Family(ABC):
    """Base class for structurally-constrained LP-code families."""

    #: Short string used in YAML configs to select this family.
    name: str = "<unset>"

    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Base-matrix shape implied by this family (``(ma, na)``)."""

    @abstractmethod
    def required_group_type(self) -> str:
        """One of ``"abelian"``, ``"non_abelian"``, ``"any"``."""

    @abstractmethod
    def sample(
        self,
        gd: GroupData,
        rng: np.random.Generator,
        **kwargs,
    ) -> Optional[Tuple[list, list]]:
        """Sample one ``(A, B)`` ring-matrix pair under this family's structure.

        Returns ``None`` if a single-attempt rejection budget is
        exhausted or the family is infeasible for the given group.

        The pair (A, B) is returned in whatever canonical form the family
        chooses; the phase code may further canonicalize via
        :func:`core.classical_code.canonical_form_A`/``B``.
        """

    def validate_group(self, gd: GroupData) -> None:
        """Raise ``ValueError`` if ``gd`` doesn't match
        :meth:`required_group_type`."""
        req = self.required_group_type()
        if req == "abelian" and not gd.is_abelian:
            raise ValueError(
                f"Family {self.name!r} requires an abelian group; "
                f"got {gd.structure!r}."
            )
        if req == "non_abelian" and gd.is_abelian:
            raise ValueError(
                f"Family {self.name!r} requires a non-abelian group; "
                f"got {gd.structure!r}."
            )
