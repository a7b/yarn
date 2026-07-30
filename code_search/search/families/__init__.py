"""Structurally-constrained LP-code families.

A family encodes a specific construction rule and exposes a sampler (or a
constraint on the sampling space) plus family-specific filters, through the
abstract :class:`search.families.base.Family` interface. No concrete family
ships currently; plug new ones in through the same interface.
"""

from .base import Family

__all__ = ["Family"]
