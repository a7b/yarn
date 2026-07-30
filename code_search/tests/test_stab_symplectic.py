# -*- coding: utf-8 -*-
"""T1 tests for core/dist/stab_symplectic.py (non-CSS stabilizer distance
sampler). All fast (pure numpy, no GPU/GAP)."""

import numpy as np
import pytest

from core.dist.stab_symplectic import (
    centralizer_basis,
    estimate_coset_min_weight,
    estimate_stab_distance,
    stab_k,
    symplectic_weight,
)

pytestmark = pytest.mark.fast


def _five_qubit():
    """[[5,1,3]] perfect code: cyclic XZZXI, 4 generators, genuinely non-CSS."""
    s = "XZZXI"
    gens = [s[-i:] + s[:-i] for i in range(4)]
    HX = np.array([[1 if c == "X" else 0 for c in g] for g in gens], np.uint8)
    HZ = np.array([[1 if c == "Z" else 0 for c in g] for g in gens], np.uint8)
    return HX, HZ


def _steane():
    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]], np.uint8)
    HX = np.vstack([H, np.zeros_like(H)])
    HZ = np.vstack([np.zeros_like(H), H])
    return HX, HZ


def test_stab_k():
    HX, HZ = _five_qubit()
    assert stab_k(HX, HZ) == 1
    HX, HZ = _steane()
    assert stab_k(HX, HZ) == 1


def test_centralizer_basis_orthogonal_and_full():
    HX, HZ = _five_qubit()
    CX, CZ = centralizer_basis(HX, HZ)
    # dim = W + k = 6
    assert CX.shape == (6, 5) and CZ.shape == (6, 5)
    comm = (HX.astype(int) @ CZ.T + HZ.astype(int) @ CX.T) % 2
    assert not comm.any()


def test_five_qubit_distance():
    HX, HZ = _five_qubit()
    r = estimate_stab_distance(HX, HZ, num_trials=300, seed=1)
    assert r["d_est"] == 3
    # witness is a genuine logical: commutes, nonzero, weight 3
    wx, wz = r["witness_x"], r["witness_z"]
    comm = (HX.astype(int) @ wz + HZ.astype(int) @ wx) % 2
    assert not comm.any()
    assert symplectic_weight(wx, wz) == 3
    assert r["trials_run"] == 300 and not r["early_stopped"]


def test_steane_distance():
    HX, HZ = _steane()
    r = estimate_stab_distance(HX, HZ, num_trials=300, seed=2)
    assert r["d_est"] == 3


def test_422_distance_and_early_stop_semantics():
    HX = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], np.uint8)
    HZ = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], np.uint8)
    r = estimate_stab_distance(HX, HZ, num_trials=100, seed=3)
    assert r["d_est"] == 2
    # strict <: d_target=2 must NOT early-stop on a weight-2 logical
    r = estimate_stab_distance(HX, HZ, num_trials=60, seed=4, d_target=2)
    assert not r["early_stopped"] and r["d_est"] == 2
    # d_target=3 stops early on the weight-2 witness
    r = estimate_stab_distance(HX, HZ, num_trials=100, seed=5, d_target=3)
    assert r["early_stopped"] and r["d_est"] == 2 and r["trials_run"] < 100


def test_k0_raises():
    HX = np.array([[1, 0], [0, 0]], np.uint8)
    HZ = np.array([[0, 0], [1, 1]], np.uint8)
    # k = 2 - 2 = 0
    with pytest.raises(ValueError):
        estimate_stab_distance(HX, HZ, num_trials=10, seed=0)


def test_coset_min_weight_steane():
    HX, HZ = _steane()
    vx = np.array([1, 1, 1, 0, 0, 0, 0], np.uint8)  # weight-3 Xbar rep
    vz = np.zeros(7, np.uint8)
    r = estimate_coset_min_weight(HX, HZ, vx, vz, num_trials=150, seed=6)
    assert r["w_est"] == 3
    # witness lies in the coset: witness ^ v in rowspan
    dx = (r["witness_x"] ^ vx, r["witness_z"] ^ vz)
    aug = np.hstack([HX, HZ])
    from core.f2 import f2_rank
    v = np.concatenate(dx).reshape(1, -1)
    assert f2_rank(aug) == f2_rank(np.vstack([aug, v]))


def test_coset_in_span_raises():
    HX, HZ = _steane()
    with pytest.raises(ValueError):
        estimate_coset_min_weight(HX, HZ, HX[0], np.zeros(7, np.uint8),
                                  num_trials=10, seed=7)
