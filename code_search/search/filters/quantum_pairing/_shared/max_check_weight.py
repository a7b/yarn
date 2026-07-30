"""Pair filter: cap the resulting Hx / Hz quantum check weights.

Given ``W_A = weight_matrix(A)`` and ``W_B = weight_matrix(B)``, the
maximum row weight of ``Hx`` / ``Hz`` is computed by
:func:`core.quantum_code.quantum_check_weights` without instantiating
the binary matrices. This filter rejects pairs whose resulting check
weights exceed configured caps.
"""

from typing import Optional

import numpy as np

from core.quantum_code import quantum_check_weights


def max_check_weight(
    W_A: np.ndarray,
    W_B: np.ndarray,
    max_Hx: Optional[int] = None,
    max_Hz: Optional[int] = None,
) -> bool:
    """``True`` iff resulting check weights are within both caps.

    Args:
        W_A, W_B: integer weight matrices (``weight_matrix(A)``,
            ``weight_matrix(B)``).
        max_Hx: maximum allowed Hx row weight. ``None`` = no cap.
        max_Hz: maximum allowed Hz row weight. ``None`` = no cap.

    Returns:
        ``True`` iff (cap is ``None`` OR derived check weight ≤ cap) for
        both directions.
    """
    cw = quantum_check_weights(W_A, W_B)
    if max_Hx is not None and cw["Hx_check_weight"] > max_Hx:
        return False
    if max_Hz is not None and cw["Hz_check_weight"] > max_Hz:
        return False
    return True
