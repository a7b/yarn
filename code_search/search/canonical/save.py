"""Persist a passed quantum code: ``.npy`` matrices + a rich self-describing JSON.

One directory per code:

    <out_dir>/<tag>__<hash>/
        Hx.npy  Hz.npy  A_bin.npy  B_bin.npy  Lx.npy  Lz.npy
        code.json

``code.json`` records everything needed to reconstruct and trust the code:
identities, the ring matrices (with the identity-forced anchor flagged),
canonical / single-orbit status, the canonical logical basis + its row
weights, every distance run with ``k_sub`` paired to ``num_trials`` (the hard
rule), and check weights.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.classical_code import weight_matrix
from core.f2 import f2_rank
from core.group import GroupData


def _block_col_ranks(M_bin: np.ndarray, n: int) -> List[int]:
    """F2 ranks of each width-``n`` block-col of an ``m × (k·n)`` matrix."""
    n_blocks = M_bin.shape[1] // n
    return [int(f2_rank(M_bin[:, j * n:(j + 1) * n])) for j in range(n_blocks)]


def _ring_to_json(M) -> list:
    return [[list(entry) for entry in row] for row in M]


def code_hash(code: dict) -> str:
    """Short stable hash of the canonical ring matrices (for dir naming)."""
    payload = json.dumps(
        {"A": _ring_to_json(code["A_canonical"]),
         "B": _ring_to_json(code["B_canonical"])},
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def save_passed_code(
    out_dir: Path,
    *,
    gd: GroupData,
    gap_expr: str,
    tag: str,
    code: dict,
    basis: dict,
    dx: Optional[int],
    dz: Optional[int],
    classical_dA: Optional[int],
    classical_dB: Optional[int],
    distance_runs: List[dict],
    provenance: Optional[dict] = None,
) -> Path:
    """Write one passed-code directory; returns its path.

    ``distance_runs`` is a list of per-run dicts; each MUST carry ``k_sub``
    alongside ``num_trials`` for any sqetch stage (uninterpretable otherwise).
    """
    n = gd.n
    A_can, B_can = code["A_canonical"], code["B_canonical"]
    na = len(A_can[0])
    nb = len(B_can[0])
    k = basis["k"]
    n_phys = code["n_phys"]
    d = min(x for x in (dx, dz) if x is not None) if (dx or dz) else None
    label = f"[[{n_phys},{k}" + (f",{d}" if d is not None else "") + "]]"

    h = code_hash(code)
    code_dir = out_dir / f"{tag}__{h}"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Matrices
    np.save(code_dir / "Hx.npy", code["Hx"])
    np.save(code_dir / "Hz.npy", code["Hz"])
    np.save(code_dir / "A_bin.npy", code["A_bin"])
    np.save(code_dir / "B_bin.npy", code["B_bin"])
    np.save(code_dir / "Lx.npy", basis["Lx"])
    np.save(code_dir / "Lz.npy", basis["Lz"])

    anchor_A = A_can[0][na - 1]
    anchor_B = B_can[0][nb - 1]
    W_A, W_B = weight_matrix(A_can), weight_matrix(B_can)

    data = {
        "code": {"n": n_phys, "k": k, "dx": dx, "dz": dz, "label": label},
        "group": {"gap_expr": gap_expr, "tag": tag, "order": n,
                  "abelian": bool(gd.is_abelian)},
        "shape": [1, 2],
        "ring": {
            "weight_A": W_A.tolist(), "weight_B": W_B.tolist(),
            "A": _ring_to_json(A_can), "B": _ring_to_json(B_can),
            "anchor_col": na - 1,
            "identity_in_anchor_A": bool(gd.identity in set(anchor_A)),
            "identity_in_anchor_B": bool(gd.identity in set(anchor_B)),
        },
        "canonical": {
            "by_construction": True,
            "single_orbit": True,
            "block_col_ranks_A": _block_col_ranks(code["A_bin"], n),
            "block_col_ranks_B": _block_col_ranks(code["B_bin"], n),
            "has_full_rank_a": bool(code["has_full_rank_a"]),
            "has_full_rank_b": bool(code["has_full_rank_b"]),
            "is_css": bool(code["is_css"]),
        },
        "logical_basis": {
            "k": k,
            "n_groups": basis["n_groups"],
            "Lx_weight": basis["Lx_weight"],
            "Lz_weight": basis["Lz_weight"],
            "Lx_dot_LzT_is_I": basis["Lx_dot_LzT_is_I"],
            "files": {"Lx": "Lx.npy", "Lz": "Lz.npy"},
        },
        "distance": {
            "dx": dx, "dz": dz,
            "classical_dA": classical_dA, "classical_dB": classical_dB,
            "verdict": "pass",
            "runs": distance_runs,
        },
        "check_weight": {"max_Hx": code["max_Hx_check_weight"],
                         "max_Hz": code["max_Hz_check_weight"]},
        "files": {"Hx": "Hx.npy", "Hz": "Hz.npy",
                  "A_bin": "A_bin.npy", "B_bin": "B_bin.npy",
                  "Lx": "Lx.npy", "Lz": "Lz.npy"},
        "provenance": provenance or {},
        "timestamp": int(time.time()),
    }
    (code_dir / "code.json").write_text(json.dumps(data, indent=2))
    return code_dir
