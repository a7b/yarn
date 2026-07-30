"""Regression: a classical screen that stores BOTH ``None`` ("nothing found",
a legitimate PASS) and integer distances must not crash the end-of-group
summary. ``sorted({None, 16})`` used to raise ``TypeError`` in
``run_group``/``sample_search`` — after every estimator call had already been
spent and before ``_save_stats`` ran, losing the group's results. Integers now
sort first, ``None`` last.
"""

import json

import pytest

import _fresh_helpers as fh
from search.canonical import run_group as rg
from search.canonical.groups import GroupSpec
from search.canonical.run_group import ScreenParams, run_group

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def d8():
    g = fh.dihedral_shim(4)
    fh.check_group_axioms(g)
    return g


def _alternating_none_16():
    """None on odd calls, 16 on even — both PASS any target ≤ 16."""
    state = {"n": 0}

    def est(M, **kw):
        state["n"] += 1
        return None if state["n"] % 2 else 16

    return est


def test_mixed_none_and_int_survivor_distances_do_not_crash(
        d8, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch",
                        _alternating_none_16())
    # Refute every pairing at the quantum screen: the funnel still has to
    # reach the summary (where the mixed d-set is sorted) and save stats.
    monkeypatch.setattr(rg, "estimate_quantum_distances_sqetch",
                        lambda *a, **kw: (1, 1))
    monkeypatch.setattr(rg, "estimate_quantum_distances_bposd",
                        lambda *a, **kw: (1, 1))

    params = ScreenParams(target=2, max_A_pool=2, max_B_pool=2,
                          anchor_limit=1, free_limit=2)
    res = run_group(GroupSpec(8, 3), params, tmp_path, gd=d8)

    # Each side screened exactly 2 candidates → d-set {None, 16} on both.
    assert res["A_d_classical"] == [16, None]
    assert res["B_d_classical"] == [16, None]
    assert res["passed"] == []

    # The summary must have been persisted (this is what the crash destroyed).
    stats = json.loads((tmp_path / "_stats_Sg8_3.json").read_text())
    assert stats["A_d_classical"] == [16, None]
    assert stats["B_d_classical"] == [16, None]
