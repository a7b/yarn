"""T1: canonical-search build + logical basis + JSON save (CPU, no GPU).

Marked ``gap`` (needs GroupData). Distance estimation (sqetch/bposd) is NOT
exercised here — that's the GPU path, covered by the live driver runs.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from core.group import GroupData
from search.canonical.build import (
    NotSingleOrbit,
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.save import save_passed_code

pytestmark = pytest.mark.gap


def _single_orbit_s3():
    """Find a single-orbit (k=|G|) S3 code with identity-monomial anchors."""
    gd = GroupData("SymmetricGroup(3)")
    n = gd.n
    for a0 in itertools.combinations(range(n), 3):
        for b0 in itertools.combinations(range(n), 3):
            code = build_canonical_code(a0, (0,), b0, (0,), gd)
            if code["is_css"] and code["k"] == n:
                try:
                    basis = canonical_logical_basis(code, gd)
                except NotSingleOrbit:
                    continue
                return gd, a0, b0, code, basis
    raise AssertionError("no single-orbit S3 code found")


def test_build_shapes_and_css():
    gd, a0, b0, code, basis = _single_orbit_s3()
    n = gd.n
    assert code["n_phys"] == 5 * n          # shape (1,2) → 5|G|
    assert code["k"] == n                   # single orbit
    assert code["is_css"]
    assert code["Hx"].shape[1] == 5 * n and code["Hz"].shape[1] == 5 * n
    assert a_bin_for(a0, (0,), gd).shape == (n, 2 * n)
    assert b_bin_for(b0, (0,), gd).shape == (n, 2 * n)


def test_logical_basis_invariants():
    gd, a0, b0, code, basis = _single_orbit_s3()
    Hx, Hz, Lx, Lz = code["Hx"], code["Hz"], basis["Lx"], basis["Lz"]
    assert np.all((Hx @ Lz.T) % 2 == 0)     # Hx Lz^T = 0
    assert np.all((Hz @ Lx.T) % 2 == 0)     # Hz Lx^T = 0
    assert basis["Lx_dot_LzT_is_I"]         # Lx Lz^T = I_k
    assert basis["k"] == gd.n
    # single-orbit ⇒ uniform per-direction row weight
    assert basis["Lx_weight"]["uniform"] is not None
    assert basis["Lz_weight"]["uniform"] is not None


def test_not_single_orbit_raises():
    """A code with k % |G| != 0 (no clean orbit basis) raises NotSingleOrbit."""
    gd = GroupData("SymmetricGroup(3)")
    # degenerate: a0 = a1 = identity makes A_bin rank-deficient → k > |G|.
    code = build_canonical_code((0,), (0,), (0, 1, 2), (0,), gd)
    if code["k"] != gd.n:
        with pytest.raises(NotSingleOrbit):
            canonical_logical_basis(code, gd)


def test_save_passed_code_writes_everything(tmp_path):
    gd, a0, b0, code, basis = _single_orbit_s3()
    runs = [
        {"stage": "classical_A", "backend": "sqetch", "num_trials": 5_000_000,
         "k_sub": gd.n, "batch_size": 50000, "d": 4},
        {"stage": "quantum_screen", "backend": "sqetch", "num_trials": 200000,
         "k_sub": 192, "dx": 4, "dz": 4},
    ]
    d = save_passed_code(
        tmp_path, gd=gd, gap_expr="SymmetricGroup(3)", tag="S3",
        code=code, basis=basis, dx=4, dz=4, classical_dA=4, classical_dB=4,
        distance_runs=runs, provenance={"test": True},
    )
    # all matrix files present
    for fn in ("Hx.npy", "Hz.npy", "A_bin.npy", "B_bin.npy", "Lx.npy",
               "Lz.npy", "code.json"):
        assert (d / fn).exists(), fn
    meta = json.loads((d / "code.json").read_text())
    assert meta["code"]["n"] == 5 * gd.n and meta["code"]["k"] == gd.n
    assert meta["canonical"]["single_orbit"] is True
    assert meta["logical_basis"]["Lx_dot_LzT_is_I"] is True
    # k_sub recorded on every sqetch run (the hard rule)
    for r in meta["distance"]["runs"]:
        if r["backend"] == "sqetch":
            assert "k_sub" in r and "num_trials" in r
    # block-col ranks recorded; last col is the full-rank anchor
    assert meta["canonical"]["block_col_ranks_A"][-1] == gd.n
