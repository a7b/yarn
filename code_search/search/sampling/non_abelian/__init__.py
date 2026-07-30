"""Sampling primitives applicable to non-abelian groups.

Re-exports the group-agnostic primitives from ``_shared``. Non-abelian
specific samplers (e.g. ``avoid_same_coset`` rejection) will land here
as additional modules later.
"""

from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool,
    sample_A_from_pool,
)
from search.sampling._shared.random_ring_element import random_ring_element
from search.sampling._shared.random_ring_matrix import random_ring_matrix

__all__ = [
    "build_full_rank_block_pool",
    "random_ring_element",
    "random_ring_matrix",
    "sample_A_from_pool",
]
