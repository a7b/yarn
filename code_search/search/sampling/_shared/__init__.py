"""Group-agnostic sampling primitives.

- ``random_ring_element(gd, weight, *, rng, knobs...)`` — per-entry
  sampler (handles any weight; monomial/polynomial dispatched by ``weight``).
- ``random_ring_matrix(gd, weight_matrix, *, rng, knobs..., overrides)``
  — per-matrix sampler with two-level knob resolution.
- ``full_rank_block_pool`` — specialized sampler that builds a pool of
  full-rank anchor entries (units of F₂[G]) for structured canonical
  basis construction.
"""

from .full_rank_block_pool import (
    build_full_rank_block_pool,
    sample_A_from_pool,
)
from .random_ring_element import random_ring_element
from .random_ring_matrix import random_ring_matrix
from .weight_matrix import all_weight_patterns, random_weight_patterns

__all__ = [
    "all_weight_patterns",
    "build_full_rank_block_pool",
    "random_ring_element",
    "random_ring_matrix",
    "random_weight_patterns",
    "sample_A_from_pool",
]
