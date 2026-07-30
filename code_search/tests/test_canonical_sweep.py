"""T1 tests for ``search/canonical/sample_search`` (random-sampling sweep).

All ``fast`` — no GAP, no GPU. The group is a GAP-free dihedral shim (D8);
the sqetch estimators are injected stubs; ``max_k_sub_for`` is monkeypatched
(it would otherwise touch the GPU). Every D8 weight-3 unit pair is
single-orbit (k=|G|), so the pairing counters are deterministic.
"""

import json

import numpy as np
import pytest

from search.canonical import sample_search as ss
from search.canonical.groups import GroupSpec, _pick_low_high
from search.canonical.sample_search import (
    SweepParams,
    pair_sampled,
    run_group_sweep,
    run_group_sweep_streaming,
    sample_classical_side,
)
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
)

pytestmark = pytest.mark.fast


# ─────────────────────────────────────────────────────────────────
# GAP-free fixtures
# ─────────────────────────────────────────────────────────────────


class _Shim:
    """Minimal GAP-free GroupData stand-in (Cayley table only)."""

    def __init__(self, mult, identity=0):
        self.mult = np.asarray(mult, dtype=np.int64)
        self.n = int(self.mult.shape[0])
        self.identity = int(identity)
        inv = np.zeros(self.n, np.int64)
        for g in range(self.n):
            for h in range(self.n):
                if self.mult[g, h] == identity:
                    inv[g] = h
        self.inv = inv
        self.is_abelian = bool((self.mult == self.mult.T).all())
        self.structure = "test-shim"
        self.gap_expr = "test-shim"
        # NOTE: deliberately NO `coset_id` — the production GroupShim lacks it,
        # so the default avoid_same_coset=False path must not read it.


def dihedral_shim(m: int) -> _Shim:
    """Dihedral group of order 2m: index = k + m*f for rotation^k * flip^f."""
    n = 2 * m
    mult = np.zeros((n, n), np.int64)
    for k1 in range(m):
        for f1 in range(2):
            for k2 in range(m):
                for f2 in range(2):
                    k = (k1 + (k2 if f1 == 0 else -k2)) % m
                    mult[k1 + m * f1, k2 + m * f2] = k + m * (f1 ^ f2)
    return _Shim(mult)


@pytest.fixture(scope="module")
def d8():
    return dihedral_shim(4)


@pytest.fixture(scope="module")
def pool(d8):
    p = build_full_rank_block_pool_brute(d8, 3, force_identity=True)
    assert len(p) > 0
    return p


@pytest.fixture(autouse=True)
def _no_gpu_ksub(monkeypatch):
    """max_k_sub_for would touch the GPU; pin it for CPU-only tests."""
    monkeypatch.setattr(ss, "max_k_sub_for", lambda *a, **k: 8)


# Estimator stubs ---------------------------------------------------
def _cl_fail(M, **kw):       # classical d below target → no survivor
    return 2


def _cl_pass(M, **kw):       # classical "nothing < target found" → survivor
    return None


def _q_fail(Hx, Hz, **kw):   # quantum logical of weight 1 found → not a pass
    return (1, 1)


def _q_pass(Hx, Hz, **kw):   # nothing < target → pass in both directions
    return (None, None)


# ─────────────────────────────────────────────────────────────────
# Pure selection logic
# ─────────────────────────────────────────────────────────────────


def test_pick_low_high_basic():
    ranked = [(0.6, "f"), (0.1, "a"), (0.3, "c"),
              (0.2, "b"), (0.5, "e"), (0.4, "d")]
    chosen = _pick_low_high(ranked, n_low=3, n_high=2)
    # 3 lowest (a,b,c) then 2 highest (f,e -> ascending e,f)
    assert chosen[:3] == ["a", "b", "c"]
    assert set(chosen[3:]) == {"e", "f"}
    assert len(chosen) == 5


def test_pick_low_high_small_pool_returns_all_no_dup():
    ranked = [(0.2, "b"), (0.1, "a"), (0.3, "c")]   # 3 ≤ n_low+n_high
    chosen = _pick_low_high(ranked, n_low=3, n_high=2)
    assert chosen == ["a", "b", "c"]                 # ascending, no duplicates


def test_pick_low_high_exact_boundary_no_overlap():
    ranked = [(i / 10, f"g{i}") for i in range(5)]   # exactly n_low+n_high
    chosen = _pick_low_high(ranked, n_low=3, n_high=2)
    assert sorted(chosen) == ["g0", "g1", "g2", "g3", "g4"]
    assert len(chosen) == len(set(chosen))


# ─────────────────────────────────────────────────────────────────
# Classical screen: cap / budget / barren-skip
# ─────────────────────────────────────────────────────────────────


def test_classical_barren_skip(d8, pool):
    rng = np.random.default_rng(0)
    p = SweepParams(target=16, cl_barren_skip=40, cl_sample_budget=500)
    surv, status = sample_classical_side(d8, pool, p, "A", rng=rng,
                                         classical_estimator=_cl_fail)
    assert status == "barren"
    assert surv == []


def test_classical_survivor_cap(d8, pool):
    rng = np.random.default_rng(0)
    p = SweepParams(target=16, surv_cap=10, cl_sample_budget=10_000,
                    cl_barren_skip=10_000)
    surv, status = sample_classical_side(d8, pool, p, "A", rng=rng,
                                         classical_estimator=_cl_pass)
    assert status == "ok"
    assert len(surv) == 10
    s = surv[0]
    assert set(s) == {"free", "anchor", "d", "k_sub"}
    assert s["d"] is None and s["k_sub"] >= 1


def test_classical_budget_caps_draws(d8, pool):
    rng = np.random.default_rng(0)
    # pass-always with a huge cap → the 30-draw budget is the binding limit.
    p = SweepParams(target=16, surv_cap=10**9, cl_sample_budget=30,
                    cl_barren_skip=10**9)
    surv, status = sample_classical_side(d8, pool, p, "B", rng=rng,
                                         classical_estimator=_cl_pass)
    assert status == "ok"
    assert len(surv) == 30


# ─────────────────────────────────────────────────────────────────
# Pairing: pass-stop / fail-cutoff / budget
# ─────────────────────────────────────────────────────────────────


def _survivors(d8, pool, n, rng, side="A"):
    p = SweepParams(target=16, surv_cap=n, cl_sample_budget=10_000,
                    cl_barren_skip=10_000)
    surv, _ = sample_classical_side(d8, pool, p, side, rng=rng,
                                    classical_estimator=_cl_pass)
    return surv


def test_pair_pass_stop_and_save(d8, pool, tmp_path):
    rng = np.random.default_rng(1)
    A = _survivors(d8, pool, 6, rng, "A")
    B = _survivors(d8, pool, 6, rng, "B")
    p = SweepParams(target=16, max_pass=3, pair_budget=10_000, pair_fail=10**9)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=rng,
                       gap_expr="D8", tag="d8", quantum_estimator=_q_pass)
    assert res["verdict"] == "pass"
    assert len(res["passed"]) == 3
    # save path produced real canonical artifacts
    cj = json.loads((__import__("pathlib").Path(res["passed"][0]) / "code.json").read_text())
    assert cj["canonical"]["single_orbit"] is True
    assert cj["logical_basis"]["Lx_dot_LzT_is_I"] is True
    assert cj["code"]["n"] == 5 * d8.n


def test_pair_fail_cutoff(d8, pool, tmp_path):
    rng = np.random.default_rng(2)
    A = _survivors(d8, pool, 6, rng, "A")
    B = _survivors(d8, pool, 6, rng, "B")
    p = SweepParams(target=16, max_pass=6, pair_budget=10_000, pair_fail=5)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=rng,
                       gap_expr="D8", tag="d8", quantum_estimator=_q_fail)
    assert res["verdict"] == "fail"
    assert res["passed"] == []
    assert res["n_pairs"] == 5            # every D8 pair is single-orbit → exact


def test_pair_budget_stop(d8, pool, tmp_path):
    rng = np.random.default_rng(3)
    A = _survivors(d8, pool, 6, rng, "A")
    B = _survivors(d8, pool, 6, rng, "B")
    p = SweepParams(target=16, max_pass=6, pair_budget=7, pair_fail=10**9)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=rng,
                       gap_expr="D8", tag="d8", quantum_estimator=_q_fail)
    assert res["verdict"] == "budget"
    assert res["n_pairs"] == 7
    assert res["passed"] == []


# ─────────────────────────────────────────────────────────────────
# End-to-end per-group verdicts
# ─────────────────────────────────────────────────────────────────


def test_run_group_sweep_skip_on_barren(d8, tmp_path):
    g = GroupSpec(order=8, small_group_id=3)
    p = SweepParams(target=16, cl_barren_skip=40, cl_sample_budget=200)
    res = run_group_sweep(g, p, tmp_path, gd=d8, classical_estimator=_cl_fail)
    assert res["verdict"] == "skip"
    assert res["passed"] == []
    assert (tmp_path / f"_sweep_{g.tag}.json").exists()


def test_run_group_sweep_pass(d8, tmp_path):
    g = GroupSpec(order=8, small_group_id=3)
    p = SweepParams(target=16, surv_cap=6, cl_sample_budget=10_000,
                    cl_barren_skip=10_000, max_pass=2,
                    pair_budget=10_000, pair_fail=10**9)
    res = run_group_sweep(g, p, tmp_path, gd=d8,
                          classical_estimator=_cl_pass, quantum_estimator=_q_pass)
    assert res["verdict"] == "pass"
    assert len(res["passed"]) == 2
    assert res["n_A_survivors"] == 6 and res["n_B_survivors"] == 6


def test_run_group_sweep_no_units(tmp_path):
    # D6 ≅ S3 (order 6) has ZERO weight-3 units → empty anchor pool → no_units
    # (the search never reaches the screen). No estimator should be called.
    s3 = dihedral_shim(3)
    g = GroupSpec(order=6, small_group_id=1)

    def _boom(*a, **k):                       # must not be called
        raise AssertionError("estimator called despite empty anchor pool")

    p = SweepParams(target=16)
    res = run_group_sweep(g, p, tmp_path, gd=s3, classical_estimator=_boom)
    assert res["verdict"] == "no_units"
    assert res["pool_size"] == 0
    assert (tmp_path / f"_sweep_{g.tag}.json").exists()


# ─────────────────────────────────────────────────────────────────
# Streaming variant: pair-as-found
# ─────────────────────────────────────────────────────────────────


def test_streaming_pass_saves_and_stops(d8, tmp_path):
    g = GroupSpec(order=8, small_group_id=3)
    p = SweepParams(target=16, surv_cap=6, cl_sample_budget=10_000,
                    cl_barren_skip=10_000, max_pass=2,
                    pair_budget=10_000, pair_fail=10**9)
    res = run_group_sweep_streaming(g, p, tmp_path, gd=d8,
                                    classical_estimator=_cl_pass,
                                    quantum_estimator=_q_pass)
    assert res["verdict"] == "pass"
    assert len(res["passed"]) == 2            # stops at max_pass
    cj = json.loads((__import__("pathlib").Path(res["passed"][0]) / "code.json").read_text())
    assert cj["canonical"]["single_orbit"] is True
    assert cj["logical_basis"]["Lx_dot_LzT_is_I"] is True
    assert cj["code"]["n"] == 5 * d8.n


def test_streaming_skip_on_barren_short_circuits(d8, tmp_path):
    # A side barren ⇒ verdict skip; B should never be screened (estimator
    # called only for A's barren budget — assert via small barren budget).
    g = GroupSpec(order=8, small_group_id=3)
    p = SweepParams(target=16, cl_barren_skip=30, cl_sample_budget=500,
                    surv_cap=6, pair_fail=10**9)
    res = run_group_sweep_streaming(g, p, tmp_path, gd=d8,
                                    classical_estimator=_cl_fail,
                                    quantum_estimator=_q_pass)
    assert res["verdict"] == "skip"
    assert res["passed"] == []
    assert res["n_A_survivors"] == 0
    assert res["n_pairs"] == 0


def test_streaming_fail_cutoff(d8, tmp_path):
    g = GroupSpec(order=8, small_group_id=3)
    p = SweepParams(target=16, surv_cap=6, cl_sample_budget=10_000,
                    cl_barren_skip=10_000, max_pass=6,
                    pair_budget=10_000, pair_fail=5)
    res = run_group_sweep_streaming(g, p, tmp_path, gd=d8,
                                    classical_estimator=_cl_pass,
                                    quantum_estimator=_q_fail)
    assert res["verdict"] == "fail"
    assert res["passed"] == []
    assert res["n_pairs"] >= 5
