"""Distance estimators for classical and CSS quantum codes.

Each backend is a separate module (Level 0 — no unified dispatcher) so the
caller picks the engine explicitly. All quantum-CSS backends share the
two-direction signature::

    estimate_quantum_distances_*(Hx, Hz, Lx, Lz, *, num_trials, ..., d_target=None)
        -> tuple[int | None, int | None]            (default)
        -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]
            (sqetch only, with return_logical=True)

Conventions shared by every backend:
- **Strict-`<` early stop**: a codeword of weight strictly less than
  `d_target` halts the search; weight `== d_target` is the boundary PASS
  case and does NOT halt.
- **`int | None` per direction**: ``None`` means "no codeword found in the
  trial budget" — callers should treat as PASS.
- **Cross-type augmentation**: dx uses `(Hz, Lz)`; dz uses `(Hx, Lx)`.
- **Lx, Lz are optional inputs.** When omitted, the quantum estimators
  derive a consistent paired RREF basis internally
  (``logical_basis.find_logical_noncanonical_RREF``); pass explicit bases
  when a specific one matters.
"""

from .classical import (
    estimate_classical_distance,
    estimate_classical_distance_bposd,
    estimate_classical_distance_sqetch,
)
from .quantum_bposd import estimate_quantum_distances_bposd
from .quantum_sqetch import (
    estimate_quantum_distances_sqetch,
    sample_low_weight_logicals_sqetch,
)

__all__ = [
    "estimate_classical_distance",
    "estimate_classical_distance_bposd",
    "estimate_classical_distance_sqetch",
    "estimate_quantum_distances_bposd",
    "estimate_quantum_distances_sqetch",
    "sample_low_weight_logicals_sqetch",
]
