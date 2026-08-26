"""GPU belief-propagation kernels for the S1 and S2 stages.

Each module contains CUDA C source in string literals. CuPy compiles the source
when a decoder is constructed. The modules do not share an implementation, so
changes to common BP logic may need to be applied to more than one file.

    s1_layered_bp.py       S1LayeredBP     fixed-iteration BP (S1)
    s2_relay_bp.py         S2RelayBP       relay-BP + coset quorum (S2)
    s2_relay_bp_gari.py    S2RelayBPGari   the same, for the GARI system (S2)

Importing this subpackage imports CuPy. The top-level package therefore loads
it lazily when a GPU stage is constructed.
"""
from .s1_layered_bp import S1LayeredBP
from .s2_relay_bp import DEFAULT_S2_PHASE, S2RelayBP, S2RelayPhase
from .s2_relay_bp_gari import (
    DEFAULT_S2_PHASE_GARI, S2RelayBPGari, S2RelayPhaseGari,
)

__all__ = [
    "S1LayeredBP",
    "S2RelayBP", "S2RelayPhase", "DEFAULT_S2_PHASE",
    "S2RelayBPGari", "S2RelayPhaseGari", "DEFAULT_S2_PHASE_GARI",
]
