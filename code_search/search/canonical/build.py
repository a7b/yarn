"""Build a canonical shape-(1,2) LP code from four ring entries, and its
G-orbit canonical logical basis.

A = [[a0, a1]], B = [[b0, b1]]  (both 1×2 → n_phys = 5|G|, k = |G| if single
orbit). ``a1``/``b1`` are the unit anchors (identity in support) sitting at
the LAST block-col, so codes are **canonical by construction**.

The classical sides are screened independently: ``d(ker A_bin)`` and
``d(ker B_bin)`` upper-bound the quantum distance, so neither side need be
paired to compute it.
"""

from typing import Iterable, Optional

import numpy as np

from core.classical_code import build_A_bin, build_B_bin, weight_matrix
from core.group import GroupData, canonicalize
from core.quantum_code import (
    build_quantum_code,
    check_css,
    compute_k,
    quantum_check_weights,
)
from logical_basis.logical_basis import find_logical_basis

SHAPE_A = (1, 2)
SHAPE_B = (1, 2)


class NotSingleOrbit(ValueError):
    """Raised when a code has no clean single-orbit canonical basis (k != |G|)."""


# ─────────────────────────────────────────────────────────────────
# Classical-side lifts (for the sqetch distance screen)
# ─────────────────────────────────────────────────────────────────


def a_bin_for(a0: Iterable[int], a1: Iterable[int], gd: GroupData) -> np.ndarray:
    """``A_bin`` (|G|×2|G|) for the classical A-code A = [[a0, a1]]."""
    A = [[canonicalize(a0), canonicalize(a1)]]
    return build_A_bin(A, gd)


def b_bin_for(b0: Iterable[int], b1: Iterable[int], gd: GroupData) -> np.ndarray:
    """``B_bin`` (|G|×2|G|) for the classical B-code B = [[b0, b1]]."""
    B = [[canonicalize(b0), canonicalize(b1)]]
    return build_B_bin(B, gd)


# ─────────────────────────────────────────────────────────────────
# Quantum-code construction (canonical by construction)
# ─────────────────────────────────────────────────────────────────


def build_canonical_code(a0, a1, b0, b1, gd: GroupData) -> dict:
    """Build the CSS code for A=[[a0,a1]], B=[[b0,b1]].

    Returns a dict with the ring matrices, ``qcode`` (the
    :func:`build_quantum_code` output: Hx, Hz, canonical A/B, perms,
    full-rank flags), the canonical ``A_bin``/``B_bin``, ``k``, the
    CSS-orthogonality flag, and the max check weights.
    """
    A = [[canonicalize(a0), canonicalize(a1)]]
    B = [[canonicalize(b0), canonicalize(b1)]]
    qcode = build_quantum_code(A, B, gd)
    A_can, B_can = qcode["A_canonical"], qcode["B_canonical"]
    A_bin = build_A_bin(A_can, gd)
    B_bin = build_B_bin(B_can, gd)
    Hx, Hz = qcode["Hx"], qcode["Hz"]
    W_A, W_B = weight_matrix(A_can), weight_matrix(B_can)
    cw = quantum_check_weights(W_A, W_B)
    return {
        "A": A, "B": B,
        "A_canonical": A_can, "B_canonical": B_can,
        "qcode": qcode,
        "Hx": Hx, "Hz": Hz,
        "A_bin": A_bin, "B_bin": B_bin,
        "k": compute_k(Hx, Hz),
        "n_phys": int(Hx.shape[1]),
        "is_css": check_css(Hx, Hz),
        "max_Hx_check_weight": int(cw["Hx_check_weight"]),
        "max_Hz_check_weight": int(cw["Hz_check_weight"]),
        "has_full_rank_a": qcode["has_full_rank_a"],
        "has_full_rank_b": qcode["has_full_rank_b"],
    }


# ─────────────────────────────────────────────────────────────────
# Canonical logical basis (G-orbit)
# ─────────────────────────────────────────────────────────────────


def _row_weights(L: np.ndarray) -> dict:
    """Row-weight summary of a logical basis block ``(k, n_phys)``."""
    if L.shape[0] == 0:
        return {"uniform": None, "min": None, "max": None, "values": []}
    w = L.sum(axis=1).astype(int)
    uniq = sorted(set(int(x) for x in w))
    return {
        "uniform": (uniq[0] if len(uniq) == 1 else None),
        "min": int(w.min()),
        "max": int(w.max()),
        "values": uniq,
    }


def canonical_logical_basis(code: dict, gd: GroupData) -> dict:
    """Construct the G-orbit canonical logical basis for a built code.

    Returns a dict with ``Lx``, ``Lz`` (uint8 arrays), ``k``, ``n_groups``,
    per-direction row-weight summaries ``Lx_weight``/``Lz_weight``, and
    ``Lx_dot_LzT_is_I``.

    Raises:
        NotSingleOrbit: the code has no structured canonical basis
            (``k % |G| != 0`` — not a clean orbit code).
    """
    try:
        res = find_logical_basis(
            code["Hx"], code["Hz"], code["A_bin"], code["B_bin"],
            gd.n, shape_a=SHAPE_A, shape_b=SHAPE_B,
        )
    except ValueError as e:
        raise NotSingleOrbit(str(e)) from e

    Lx, Lz = res["Lx"], res["Lz"]
    k, n = res["k"], gd.n
    if k != n:
        # k is a multiple of n (orbit code) but not a *single* orbit — the
        # user wants the clean single-orbit canonical basis (k == |G|).
        raise NotSingleOrbit(f"k={k} != |G|={n} (multi-orbit); want single orbit.")

    lx_lzt = (Lx.astype(np.int64) @ Lz.astype(np.int64).T) % 2
    is_I = bool(np.array_equal(lx_lzt, np.eye(k, dtype=np.int64)))
    return {
        "Lx": Lx, "Lz": Lz,
        "k": int(k), "n_groups": int(res["n_groups"]),
        "Lx_weight": _row_weights(Lx),
        "Lz_weight": _row_weights(Lz),
        "Lx_dot_LzT_is_I": is_I,
    }
