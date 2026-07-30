"""Sampling primitives applicable to abelian groups.

Re-exports the group-agnostic primitives from ``_shared``. Abelian-only
samplers (if any) land here as additional modules later.
"""

from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool,
    sample_A_from_pool,
)
from search.sampling._shared.random_ring_element import random_ring_element
from search.sampling._shared.random_ring_matrix import random_ring_matrix
from search.sampling._shared.weight_matrix import (
    all_weight_patterns,
    random_weight_patterns,
)

__all__ = [
    "all_weight_patterns",
    "build_full_rank_block_pool",
    "random_ring_element",
    "random_ring_matrix",
    "random_weight_patterns",
    "sample_A_from_pool",
]
