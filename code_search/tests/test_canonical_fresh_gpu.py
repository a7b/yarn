"""Fresh-eye GPU tests: the canonical funnel against exhaustive ground truth.

These run the REAL sqetch (and, where marked, bposd) backends on deliberately
tiny groups, so the funnel's claimed distances can be checked against exact
brute-force kernel enumeration (tests/_fresh_helpers.py). The S3 reference
code is asymmetric (true dx=2, dz=4), so any dx/dz swap anywhere in the
plumbing fails loudly.

First sqetch call JIT-compiles the CUDA kernel (~1-2 min on a cold cache);
everything after that is milliseconds per call.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import _fresh_helpers as fh
from core.group import GroupData
from search.canonical.build import (
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.groups import GroupSpec
from search.canonical.run_group import (
    ScreenParams,
    certify_quantum,
    pair_and_verify,
    run_group,
    screen_classical_side,
)
from search.canonical.sample_search import SweepParams, run_group_sweep
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gd8():
    return GroupData("SmallGroup(8,3)")            # dihedral, order 8


@pytest.fixture(scope="module")
def s3_code_and_truth():
    """The asymmetric S3 reference code plus its EXACT distances."""
    gd = GroupData("SmallGroup(6,1)")
    code = build_canonical_code((0, 1, 2), (0,), (0, 2, 4), (0,), gd)
    basis = canonical_logical_basis(code, gd)
    dx_true = fh.true_quantum_distance_one_direction(code["Hx"], code["Hz"])
    dz_true = fh.true_quantum_distance_one_direction(code["Hz"], code["Hx"])
    assert (dx_true, dz_true) == (2, 4)            # independent ground truth
    return gd, code, basis, dx_true, dz_true


# ─────────────────────────────────────────────────────────────────
# Stage oracles: sqetch-reported distances must equal brute-force truth
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
def test_classical_screen_real_sqetch_matches_bruteforce(gd8):
    """Both classical sides, real sqetch: every survivor's claimed d must
    equal the exact distance of the correct (left vs right) lift."""
    shim = fh.GroupShimFresh(np.array(gd8.mult))
    pool = build_full_rank_block_pool_brute(gd8, 3, force_identity=True)
    p = ScreenParams(target=2, cl_num_trials=20_000, cl_k_sub=0,
                     cl_batch_size=5000, anchor_limit=1, free_limit=4,
                     cl_early_stop=False, devices=[0])
    for side, my_lift in (("A", fh.my_A_bin), ("B", fh.my_B_bin)):
        surv = screen_classical_side(gd8, pool, p, side)
        # weight-3 blocks have no zero column → true d ≥ 2 always, so ALL
        # 4 scanned candidates must survive a target-2 screen.
        assert len(surv) == 4, side
        for s in surv:
            truth = fh.true_classical_distance(
                my_lift([s["free"], s["anchor"]], shim))
            assert s["d"] == truth, (side, s["free"], s["anchor"])
            assert s["d"] >= 2
            assert s["k_sub"] == gd8.n             # 0 ⇒ ker dim = |G|
        # ranked descending by claimed distance
        ds = [s["d"] for s in surv]
        assert ds == sorted(ds, reverse=True)


@pytest.mark.gap
def test_certify_quantum_real_sqetch_exact(s3_code_and_truth):
    gd, code, basis, dx_true, dz_true = s3_code_and_truth
    p = ScreenParams(target=2, certify_num_trials=50_000,
                     certify_batch_size=5000, certify_seeds=[1, 2], devices=[0])
    cdx, cdz, runs = certify_quantum(code, basis, p, 0, k_sub=18)
    assert (cdx, cdz) == (dx_true, dz_true)        # exact, and NOT swapped
    assert [r["seed"] for r in runs] == [1, 2]
    for r in runs:
        assert r["stage"] == "certify" and r["backend"] == "sqetch"
        assert r["k_sub"] == 18 and r["num_trials"] == 50_000
        assert r["dx"] == dx_true and r["dz"] == dz_true


@pytest.mark.gap
@pytest.mark.bposd
def test_pair_and_verify_real_funnel_saves_true_distances(
        s3_code_and_truth, tmp_path):
    """screen → bposd confirm → certify → save, all real, on the asymmetric
    S3 pair. The saved dx/dz must be the exact true (2, 4)."""
    gd, code, basis, dx_true, dz_true = s3_code_and_truth
    dA_true = fh.true_classical_distance(a_bin_for((0, 1, 2), (0,), gd))
    dB_true = fh.true_classical_distance(b_bin_for((0, 2, 4), (0,), gd))
    A = [{"free": (0, 1, 2), "anchor": (0,), "d": dA_true, "k_sub": gd.n}]
    B = [{"free": (0, 2, 4), "anchor": (0,), "d": dB_true, "k_sub": gd.n}]
    p = ScreenParams(target=2,
                     q_num_trials=50_000, q_k_sub=64, q_batch_size=5000,
                     bposd_num_trials=200, bposd_n_workers=2, bposd_osd_order=1,
                     certify_num_trials=50_000, certify_k_sub=64,
                     certify_batch_size=5000, certify_seeds=[1, 2], devices=[0])
    res = pair_and_verify(gd, A, B, p, tmp_path, gap_expr="SmallGroup(6,1)",
                          tag="Sg6_1", provenance={"fresh": "gpu"})
    assert len(res["passed"]) == 1
    assert res["best_dx_seen"] == dx_true and res["best_dz_seen"] == dz_true
    d = Path(res["passed"][0])
    meta = json.loads((d / "code.json").read_text())
    assert meta["distance"]["dx"] == dx_true
    assert meta["distance"]["dz"] == dz_true
    assert meta["code"]["label"] == f"[[{5*gd.n},{gd.n},{dx_true}]]"
    assert meta["distance"]["classical_dA"] == dA_true
    assert meta["distance"]["classical_dB"] == dB_true
    runs = {r["stage"]: r for r in meta["distance"]["runs"]
            if r["stage"] != "certify"}
    # the sqetch screen found the exact values
    assert runs["quantum_screen"]["dx"] == dx_true
    assert runs["quantum_screen"]["dz"] == dz_true
    # bposd (real) may only report weights of REAL logicals: ≥ truth
    bp = runs["bposd_confirm"]
    assert bp.get("dx") is None or bp["dx"] >= dx_true
    assert bp.get("dz") is None or bp["dz"] >= dz_true
    # k_sub hard rule on the real artifacts
    for r in meta["distance"]["runs"]:
        assert "num_trials" in r
        if r["backend"] == "sqetch":
            assert isinstance(r["k_sub"], int) and r["k_sub"] >= 1
    # saved npys reproduce the exact matrices we built
    assert np.array_equal(np.load(d / "Hx.npy"), code["Hx"])
    assert np.array_equal(np.load(d / "Lz.npy"), basis["Lz"])


@pytest.mark.gap
@pytest.mark.bposd
def test_run_group_real_full_funnel_d8(gd8, tmp_path):
    """The complete per-group unit of work with every backend real, on
    SmallGroup(8,3) at target 2. All screened pairs must pass (no LP code in
    this box has a weight-1 logical: no parity-check column is zero), and
    every saved artifact must satisfy the full independently-recomputed
    contract battery."""
    shim = fh.GroupShimFresh(np.array(gd8.mult))
    n = gd8.n
    p = ScreenParams(
        target=2,
        cl_num_trials=20_000, cl_k_sub=0, cl_batch_size=5000,
        max_A_pool=2, max_B_pool=2, anchor_limit=1, free_limit=4,
        q_num_trials=20_000, q_k_sub=64, q_batch_size=5000,
        max_quantum_pass=2,
        bposd_num_trials=200, bposd_n_workers=2, bposd_osd_order=1,
        certify_enabled=True, certify_num_trials=20_000, certify_k_sub=64,
        certify_batch_size=5000, certify_seeds=[1, 2],
        devices=[0])
    res = run_group(GroupSpec(8, 3), p, tmp_path, gd=gd8,
                    provenance={"suite": "fresh-gpu"})

    assert res["group"] == "Sg8_3" and res["order"] == n
    assert res["pool_size"] == len(fh.identity_unit_supports(shim, 3)) == 21
    assert res["n_A_survivors"] == 2 and res["n_B_survivors"] == 2
    # survivors are the first two frees (0,1,2), (0,1,3); their TRUE
    # classical distances (computed exactly here) must be what was recorded
    truth_ds = sorted({
        fh.true_classical_distance(fh.my_A_bin([f, tuple(shim_pool)], shim))
        for f in [(0, 1, 2), (0, 1, 3)]
        for shim_pool in [fh.identity_unit_supports(shim, 3)[0]]})
    assert res["A_d_classical"] == truth_ds
    assert res["n_pairs"] == 2 and res["n_single_orbit"] == 2
    assert len(res["passed"]) == 2                 # every pair passes target 2

    stats = json.loads((tmp_path / "_stats_Sg8_3.json").read_text())
    assert stats["n_passed"] == 2

    for cd in res["passed"]:
        d = Path(cd)
        meta = json.loads((d / "code.json").read_text())
        Hx = np.load(d / "Hx.npy")
        Hz = np.load(d / "Hz.npy")
        Lx = np.load(d / "Lx.npy")
        Lz = np.load(d / "Lz.npy")
        A_bin = np.load(d / "A_bin.npy")
        B_bin = np.load(d / "B_bin.npy")
        # independent CSS battery on the saved artifacts
        assert Hx.shape == (2 * n, 5 * n) and Hz.shape == (2 * n, 5 * n)
        assert np.all((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2 == 0)
        k = 5 * n - fh.rank2(Hx) - fh.rank2(Hz)
        assert k == n == meta["code"]["k"]
        assert np.all((Hz.astype(np.int64) @ Lx.T.astype(np.int64)) % 2 == 0)
        assert np.all((Hx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2 == 0)
        assert np.array_equal(
            (Lx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2,
            np.eye(n, dtype=np.int64))
        assert fh.rank2(np.vstack([Hx, Lx])) == fh.rank2(Hx) + n
        # saved ring must regenerate the saved lifts (independent rebuild)
        ringA = [tuple(e) for e in meta["ring"]["A"][0]]
        ringB = [tuple(e) for e in meta["ring"]["B"][0]]
        assert np.array_equal(A_bin, fh.my_A_bin(ringA, shim))
        assert np.array_equal(B_bin, fh.my_B_bin(ringB, shim))
        # recorded classical distances == exact truth for the saved rings
        assert meta["distance"]["classical_dA"] == \
            fh.true_classical_distance(A_bin)
        assert meta["distance"]["classical_dB"] == \
            fh.true_classical_distance(B_bin)
        # quantum pass floor + min-over-runs self-consistency
        dx, dz = meta["distance"]["dx"], meta["distance"]["dz"]
        assert dx is None or dx >= 2
        assert dz is None or dz >= 2
        rec_dx = [r["dx"] for r in meta["distance"]["runs"] if "dx" in r]
        rec_dz = [r["dz"] for r in meta["distance"]["runs"] if "dz" in r]
        assert dx == min(rec_dx) and dz == min(rec_dz)
        # stage roster: classical A/B + screen + confirm + 2 certify seeds
        stages = [r["stage"] for r in meta["distance"]["runs"]]
        assert stages == ["classical_A", "classical_B", "quantum_screen",
                          "bposd_confirm", "certify", "certify"]
        assert [r["seed"] for r in meta["distance"]["runs"][-2:]] == [1, 2]
        for r in meta["distance"]["runs"]:
            if r["backend"] == "sqetch":
                assert "k_sub" in r and "num_trials" in r
        assert meta["provenance"] == {"suite": "fresh-gpu"}


def test_sweep_default_estimators_gap_free_real_gpu(tmp_path):
    """sample_search with its DEFAULT (real sqetch) estimators on a purely
    Cayley-table group object — the documented GAP-free worker path. With
    seed=0 the draw sequence is deterministic and sqetch is exact on this
    tiny [[40,8]] code, so the outcome is pinned."""
    d8 = fh.dihedral_shim(4)
    fh.check_group_axioms(d8)
    p = SweepParams(target=2,
                    cl_num_trials=20_000, cl_k_sub=0, cl_batch_size=5000,
                    surv_cap=2, cl_sample_budget=10, cl_barren_skip=10,
                    q_num_trials=20_000, q_k_sub=0, q_batch_size=5000,
                    pair_budget=3, pair_fail=100, max_pass=1,
                    devices=[0], seed=0)
    res = run_group_sweep(GroupSpec(8, 3), p, tmp_path, gd=d8)
    assert res["verdict"] == "pass"
    assert res["pool_size"] == 21
    assert res["n_A_survivors"] == 2 and res["n_B_survivors"] == 2
    assert res["n_pairs"] == 1 and res["n_gate_skip"] == 0
    assert len(res["passed"]) == 1
    d = Path(res["passed"][0])
    meta = json.loads((d / "code.json").read_text())
    # single quantum sqetch at the max k_sub; the shmem ceiling is far above
    # the 5|G| = 40 physical qubits, so k_sub must exceed it
    qs = [r for r in meta["distance"]["runs"] if r["stage"] == "quantum_screen"]
    assert len(qs) == 1
    assert qs[0]["k_sub"] >= 5 * d8.n
    assert qs[0]["num_trials"] == 20_000
    # sweep saves are screen-only: no bposd / certify stages
    assert [r["stage"] for r in meta["distance"]["runs"]] == \
        ["classical_A", "classical_B", "quantum_screen"]
    # pass floor honored on the saved values
    dx, dz = meta["distance"]["dx"], meta["distance"]["dz"]
    assert (dx is None or dx >= 2) and (dz is None or dz >= 2)
    # saved artifacts pass the independent CSS battery
    Hx, Hz = np.load(d / "Hx.npy"), np.load(d / "Hz.npy")
    Lx, Lz = np.load(d / "Lx.npy"), np.load(d / "Lz.npy")
    assert np.all((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2 == 0)
    assert 5 * d8.n - fh.rank2(Hx) - fh.rank2(Hz) == d8.n
    assert np.array_equal(
        (Lx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2,
        np.eye(d8.n, dtype=np.int64))
    # stats persisted
    assert json.loads((tmp_path / "_sweep_Sg8_3.json").read_text())["n_passed"] == 1
