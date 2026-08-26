"""Staged decoding for circuit-level quantum error correction.

Stages: S1 (GPU flooding BP) -> S2 (GPU relay-BP, coset quorum) ->
S3-A/B/C (CPU relay stages) -> S4 (Gurobi IP certification, license-gated).
"""
from .config import (S1Config, S2Config, S3Config, S4Config,
                     TelescopeConfig)
from .decoder import DecodeResult, Stage, TelescopingDecoder
from .system import DecodingSystem, derive_init_det_system
from .dem_utils import dem_to_sparse_matrices
from .gari import GariModel, gari_transform, load_gari_npz, save_gari_npz, \
    xz_detector_type_mask
from .stim_mask import circuit_to_dem, circuit_xz_detector_mask

__version__ = "0.1.0"

__all__ = [
    "TelescopingDecoder", "DecodeResult", "Stage",
    "TelescopeConfig", "S1Config", "S2Config", "S3Config", "S4Config",
    "DecodingSystem", "derive_init_det_system",
    "dem_to_sparse_matrices",
    "GariModel", "gari_transform", "load_gari_npz", "save_gari_npz",
    "xz_detector_type_mask",
    "circuit_xz_detector_mask", "circuit_to_dem",
]
