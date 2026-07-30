"""Group-agnostic quantum-pairing filters.

Each is a small predicate over (already-computed) classical-code metadata
of two sides ``A`` and ``B``. The dispatcher in
``search/filters/config.py`` composes them cheapest-first BEFORE building
``Hx`` / ``Hz``.

For abelian search, callers pass the same scalar/array twice (B = A).
"""

from .full_extractor_bridge import (
    bridge_subgroup_order,
    full_extractor_bridge_from_rings,
    full_extractor_size_d_xz_bridge,
)
from .max_check_weight import max_check_weight
from .min_classical_distance import min_classical_distance
from .min_classical_girth import min_classical_girth
from .same_group import same_group
from .same_shape import same_shape

# Note: ``full_extractor_*`` is group-general (abelian + non-abelian) but heavier
# than the metadata predicates above — it builds the canonical code + orbit basis
# and runs a graph check, so the dispatcher passes it ``gd`` and applies it last.

__all__ = [
    "max_check_weight",
    "min_classical_distance",
    "min_classical_girth",
    "same_group",
    "same_shape",
    "full_extractor_size_d_xz_bridge",
    "full_extractor_bridge_from_rings",
    "bridge_subgroup_order",
]
