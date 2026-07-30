"""Fresh-eye contract tests for the canonical campaign that need GAP.

The GAP oracle is queried directly inside the tests (NumberSmallGroups /
IsAbelian / DerivedSubgroup), so groups.py is checked against GAP itself and
against hardcoded textbook facts — never against its own output. The
build-level tests run on real ``GroupData`` groups and verify true distances
by exhaustive kernel enumeration (tests/_fresh_helpers.py).
"""

import itertools
import json
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

import _fresh_helpers as fh
from core.group import GroupData
from search.canonical import run_group as rg
from search.canonical.build import (
    NotSingleOrbit,
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.groups import (
    GroupSpec,
    commutator_ratio,
    enumerate_band,
    enumerate_orders,
    nonabelian_ids,
    orders_in_band,
    select_groups_by_ratio,
    sweep_orders,
)
from search.canonical.run_group import ScreenParams, run_group
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
)

pytestmark = pytest.mark.gap


# Textbook counts of NON-abelian groups per order (independent of GAP).
TEXTBOOK_NONABELIAN_COUNTS = {
    6: 1,    # S3
    8: 2,    # D4, Q8
    10: 1,   # D5
    12: 3,   # Dic3, A4, D6
    14: 1,   # D7
    15: 0,   # C15 only
    16: 9,   # 14 groups, 5 abelian
    18: 3,   # D9, C3xS3, (C3xC3):C2
    20: 3,   # Dic5, F20, D10
    21: 1,   # C7:C3
    24: 12,  # 15 groups, 3 abelian
    27: 2,   # Heisenberg + C9:C3
}


def _gap_oracle_nonabelian_ids(order):
    """Independent oracle: ask GAP directly which SmallGroup ids are
    non-abelian (NumberSmallGroups + IsAbelian, nothing from groups.py)."""
    from gappy import gap

    total = int(gap.eval(f"NumberSmallGroups({order})"))
    return [i for i in range(1, total + 1)
            if not bool(gap.eval(f"IsAbelian(SmallGroup({order},{i}))"))]


# ─────────────────────────────────────────────────────────────────
# groups.py against the GAP oracle
# ─────────────────────────────────────────────────────────────────


def test_nonabelian_ids_matches_gap_oracle_and_textbook():
    for order, expected_count in TEXTBOOK_NONABELIAN_COUNTS.items():
        oracle = _gap_oracle_nonabelian_ids(order)
        got = nonabelian_ids(order)
        assert got == oracle, f"order {order}"
        assert len(got) == expected_count, f"order {order}"
        assert got == sorted(got)                       # ascending
        assert all(1 <= i for i in got)


def test_enumerate_orders_custom_defer_semantics():
    orders = [12, 6, 10, 8]                             # deliberately unsorted
    defer = (8, 10)
    first_pass = enumerate_orders(orders, defer=defer)
    # deferred orders contribute NOTHING on the first pass
    assert {s.order for s in first_pass} == {6, 12}
    assert [(s.order, s.small_group_id) for s in first_pass] == \
        [(6, i) for i in _gap_oracle_nonabelian_ids(6)] + \
        [(12, i) for i in _gap_oracle_nonabelian_ids(12)]
    # second pass folds them back IN NATURAL ASCENDING POSITION
    second_pass = enumerate_orders(orders, defer=defer, include_deferred=True)
    seq = [(s.order, s.small_group_id) for s in second_pass]
    assert seq == sorted(seq)
    expected = []
    for o in (6, 8, 10, 12):
        expected += [(o, i) for i in _gap_oracle_nonabelian_ids(o)]
    assert seq == expected


def test_enumerate_orders_default_defer_excludes_giants():
    # 95 (5·19) and 97 (prime) have no non-abelian groups; 96 is a deferred
    # giant — the default first pass over [95, 96, 97] must be EMPTY.
    assert enumerate_orders([95, 96, 97]) == []


def test_orders_in_band_consistent_with_band_enumeration():
    specs = enumerate_band("d16")
    expected_orders = sorted({s.order for s in specs})
    got = orders_in_band("d16")
    assert got == expected_orders
    assert 96 not in got                                # deferred giant
    assert all(84 <= o <= 100 for o in got)
    # spot-check two orders against the oracle
    assert 84 in got and _gap_oracle_nonabelian_ids(84) != []
    assert 85 not in got and _gap_oracle_nonabelian_ids(85) == []


def test_commutator_ratio_textbook_values():
    from gappy import gap

    # verify the id → group mapping first, then the ratio
    assert str(gap.eval("StructureDescription(SmallGroup(6,1))")) == "S3"
    assert commutator_ratio("SmallGroup(6,1)") == (6, 3, 0.5)
    assert str(gap.eval("StructureDescription(SmallGroup(8,4))")) == "Q8"
    assert commutator_ratio("SmallGroup(8,4)") == (8, 2, 0.25)
    assert str(gap.eval("StructureDescription(SmallGroup(12,3))")) == "A4"
    n, d, r = commutator_ratio("SmallGroup(12,3)")
    assert (n, d) == (12, 4) and abs(r - 1 / 3) < 1e-12


def test_select_groups_by_ratio_small_pool_returns_all_ascending():
    # order 12 has 3 non-abelian groups < n_low+n_high=5 → all, ascending by
    # ratio: Dic3 (3/12) and D12 (3/12) first in some order, A4 (4/12) LAST.
    chosen = select_groups_by_ratio(12)
    assert [s.order for s in chosen] == [12, 12, 12]
    assert {s.small_group_id for s in chosen} == {1, 3, 4}
    assert chosen[-1].small_group_id == 3               # A4 = unique max ratio
    assert {chosen[0].small_group_id, chosen[1].small_group_id} == {1, 4}
    # cyclic-only order → empty
    assert select_groups_by_ratio(15) == []


def test_sweep_orders_textbook_window():
    # 7, 9, 11 have no non-abelian groups; 6, 8, 10, 12 all do.
    assert sweep_orders(6, 12) == [6, 8, 10, 12]


# ─────────────────────────────────────────────────────────────────
# Duck-typing: a Cayley-table-only object must be a full GroupData substitute
# ─────────────────────────────────────────────────────────────────


def test_run_group_duck_typed_group_matches_real_groupdata(tmp_path):
    """README: run_group "accepts a real GroupData or any GAP-free duck-typed
    equivalent". Build the duck object from the real group's Cayley table and
    demand IDENTICAL funnel output (same survivors, same saved code dirs,
    same JSON payloads up to the timestamp)."""
    gd = GroupData("SmallGroup(8,3)")

    class Duck:
        pass

    duck = Duck()
    duck.n = gd.n
    duck.mult = [list(row) for row in gd.mult]
    duck.inv = list(gd.inv)
    duck.identity = gd.identity
    duck.is_abelian = gd.is_abelian
    fh.check_group_axioms(duck)                        # sanity on the fixture

    def cl(M, **kw):
        return 4

    def q(Hx, Hz, **kw):
        return (None, None)

    p = ScreenParams(target=2, max_A_pool=2, max_B_pool=2, anchor_limit=1,
                     free_limit=3, max_quantum_pass=2, certify_seeds=[1])
    with mock.patch.object(rg, "estimate_classical_distance_sqetch", cl), \
         mock.patch.object(rg, "estimate_quantum_distances_sqetch", q), \
         mock.patch.object(rg, "estimate_quantum_distances_bposd", q):
        r_real = run_group(GroupSpec(8, 3), p, tmp_path / "real", gd=gd)
        r_duck = run_group(GroupSpec(8, 3), p, tmp_path / "duck", gd=duck)

    for key in ("group", "order", "pool_size", "target", "n_A_survivors",
                "n_B_survivors", "A_d_classical", "B_d_classical", "n_pairs",
                "n_single_orbit", "best_dx_seen", "best_dz_seen"):
        assert r_real[key] == r_duck[key], key
    real_names = [Path(x).name for x in r_real["passed"]]
    duck_names = [Path(x).name for x in r_duck["passed"]]
    assert real_names == duck_names and len(real_names) == 2
    for name in real_names:
        m_real = json.loads((tmp_path / "real" / name / "code.json").read_text())
        m_duck = json.loads((tmp_path / "duck" / name / "code.json").read_text())
        m_real.pop("timestamp"), m_duck.pop("timestamp")
        assert m_real == m_duck
        for fn in ("Hx.npy", "Hz.npy", "A_bin.npy", "B_bin.npy",
                   "Lx.npy", "Lz.npy"):
            assert np.array_equal(np.load(tmp_path / "real" / name / fn),
                                  np.load(tmp_path / "duck" / name / fn)), fn


# ─────────────────────────────────────────────────────────────────
# Real-group build contracts + exact (brute-force) distances
# ─────────────────────────────────────────────────────────────────


def test_brute_anchor_pool_matches_independent_oracle():
    gd = GroupData("SmallGroup(8,3)")
    shim = fh.GroupShimFresh(np.array(gd.mult))
    pool = build_full_rank_block_pool_brute(gd, 3, force_identity=True)
    assert [tuple(x) for x in pool] == fh.identity_unit_supports(shim, 3)
    assert len(pool) == 21
    # S3 has no weight-3 units at all (documented gotcha)
    gd6 = GroupData("SmallGroup(6,1)")
    assert build_full_rank_block_pool_brute(gd6, 3, force_identity=True) == []
    assert fh.identity_unit_supports(fh.GroupShimFresh(np.array(gd6.mult)), 3) == []


def test_s3_code_true_distances_and_css_battery():
    """Exhaustively-verified ground truth on a real S3 code: the [[30,6]]
    code from a0=(0,1,2), b0=(0,2,4) with identity-monomial anchors has TRUE
    dx=2 and dz=4 (kernel enumeration), and the canonical basis satisfies
    every CSS invariant. Also checks the README claim that the classical
    side distances upper-bound the quantum distance."""
    gd = GroupData("SmallGroup(6,1)")
    n = gd.n
    code = build_canonical_code((0, 1, 2), (0,), (0, 2, 4), (0,), gd)
    assert code["n_phys"] == 5 * n and code["is_css"]
    assert code["k"] == n == 5 * n - fh.rank2(code["Hx"]) - fh.rank2(code["Hz"])
    basis = canonical_logical_basis(code, gd)
    Hx, Hz, Lx, Lz = code["Hx"], code["Hz"], basis["Lx"], basis["Lz"]
    assert np.all((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2 == 0)
    assert np.all((Hz.astype(np.int64) @ Lx.T.astype(np.int64)) % 2 == 0)
    assert np.all((Hx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2 == 0)
    assert np.array_equal(
        (Lx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2, np.eye(n, dtype=np.int64))
    # exact distances by kernel enumeration (asymmetric on purpose: a swap
    # of the dx / dz conventions anywhere would show up as (4, 2))
    dx = fh.true_quantum_distance_one_direction(Hx, Hz)
    dz = fh.true_quantum_distance_one_direction(Hz, Hx)
    assert (dx, dz) == (2, 4)
    # classical screen distances upper-bound the quantum distance
    dA = fh.true_classical_distance(a_bin_for((0, 1, 2), (0,), gd))
    dB = fh.true_classical_distance(b_bin_for((0, 2, 4), (0,), gd))
    assert (dA, dB) == (4, 2)
    assert min(dx, dz) <= min(dA, dB)
    # logical rows really are non-stabilizer (weight-dx logical exists in Lx
    # span + stabilizers, so distance ≤ row weights)
    assert basis["Lx_weight"]["min"] >= dx
    assert basis["Lz_weight"]["min"] >= dz


def test_not_single_orbit_on_real_groups():
    gd = GroupData("SmallGroup(6,1)")
    # k = 8 (not a multiple of 6) must raise
    code = build_canonical_code((0, 1, 2), (0, 1, 2), (0, 1, 2), (0,), gd)
    assert code["k"] == 8
    with pytest.raises(NotSingleOrbit):
        canonical_logical_basis(code, gd)
    # k = 2|G| (multi-orbit) on C3 must raise as well
    gd3 = GroupData("SmallGroup(3,1)")
    code3 = build_canonical_code((0, 1), (0, 1), (0, 1, 2), (0, 1, 2), gd3)
    assert code3["k"] == 2 * gd3.n
    with pytest.raises(NotSingleOrbit):
        canonical_logical_basis(code3, gd3)
    # boundary: k == |G| exactly must NOT raise, even for the degenerate
    # a0 = a1 = identity ring
    dcode = build_canonical_code((0,), (0,), (0, 1, 2), (0,), gd)
    if dcode["k"] == gd.n:
        basis = canonical_logical_basis(dcode, gd)
        assert basis["k"] == gd.n


def test_every_pool_anchor_gives_canonical_code_by_construction():
    """Spot-check the canonical-by-construction promise on a real group:
    for pool anchors, the built code needs no block-col permutation and the
    last block-col of both lifts is invertible."""
    gd = GroupData("SmallGroup(8,3)")
    n = gd.n
    pool = build_full_rank_block_pool_brute(gd, 3, force_identity=True)
    for anchor in [tuple(pool[0]), tuple(pool[-1])]:
        code = build_canonical_code((0, 1, 2), anchor, (0, 1, 4), anchor, gd)
        assert code["qcode"]["perm_a"] == [0, 1]
        assert code["qcode"]["perm_b"] == [0, 1]
        assert code["has_full_rank_a"] and code["has_full_rank_b"]
        assert fh.rank2(code["A_bin"][:, n:]) == n
        assert fh.rank2(code["B_bin"][:, n:]) == n
        assert 0 in code["A_canonical"][0][1] and 0 in code["B_canonical"][0][1]
