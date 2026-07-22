"""sqetch -- GPU random information-set decoder for CSS quantum codes.

Public surface:

* :func:`estimate_distance` -- run randomized ISD trials on the GPU and
  return the smallest logical-coset weight found.
* :class:`DistanceResult` -- frozen dataclass returned by
  :func:`estimate_distance`.
* :func:`estimate_distance_multi` -- multi-GPU dispatch (trial_split).
* :func:`recover_codeword` -- like estimate_distance but returns the bit vector.
* :func:`detect_gpus` -- enumerate visible CUDA devices.
"""

from .api import DistanceResult, estimate_distance
from .multi import detect_gpus, estimate_distance_multi
from .recover import recover_codeword

__version__ = "0.1.0"

__all__ = [
    "estimate_distance",
    "estimate_distance_multi",
    "recover_codeword",
    "detect_gpus",
    "DistanceResult",
    "__version__",
]
