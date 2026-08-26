"""Numerical correctness tests for the Gurobi-backed S4 stage."""
from itertools import product
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import stim

from conftest import DEM_FILE, MATRICES_NPZ
from telescoping_decoder.config import TelescopeConfig
from telescoping_decoder.decoder import Stage, TelescopingDecoder
from telescoping_decoder.ip_solver import (
    DEMData,
    _gurobi_env,
    _solve_ip_gurobi,
    build_sub_dem_indices,
)
from telescoping_decoder.system import DecodingSystem
import telescoping_decoder.s4_ip as s4


H = sp.csr_matrix(
    [
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1],
    ],
    dtype=np.uint8,
)
L = sp.csr_matrix(
    [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)
PROBS = np.array([0.01, 0.025, 0.06, 0.11, 0.18, 0.27])
WEIGHTS = np.log((1.0 - PROBS) / PROBS)


@pytest.fixture(scope="module", autouse=True)
def working_gurobi():
    """Skip cleanly when the dependency or a usable license is absent."""
    pytest.importorskip("gurobipy")
    try:
        _gurobi_env()
    except Exception as exc:
        pytest.skip(f"Gurobi is installed but unavailable: {exc}")


def _exhaustive(H_matrix, weights, syndrome, L_matrix=None):
    """All minimum-cost binary corrections for one tiny syndrome."""
    H_dense = np.asarray(H_matrix.toarray(), dtype=np.uint8)
    L_dense = (None if L_matrix is None
               else np.asarray(L_matrix.toarray(), dtype=np.uint8))
    best_cost = np.inf
    best = []
    for bits in product((0, 1), repeat=H_dense.shape[1]):
        x = np.asarray(bits, dtype=np.uint8)
        if not np.array_equal((H_dense @ x) & 1, syndrome):
            continue
        cost = float(weights @ x)
        if cost < best_cost - 1e-12:
            best_cost = cost
            best = [x]
        elif abs(cost - best_cost) <= 1e-12:
            best.append(x)
    observations = ({tuple(((L_dense @ x) & 1).tolist()) for x in best}
                    if L_dense is not None else set())
    return best_cost, best, observations


@pytest.mark.parametrize(
    "syndrome",
    [np.asarray(s, dtype=np.uint8) for s in product((0, 1), repeat=3)],
)
def test_gurobi_ip_matches_exhaustive_ml_decode(syndrome):
    """The integer formulation must find an exact ML correction."""
    expected_cost, _best, expected_obs = _exhaustive(
        H, WEIGHTS, syndrome, L,
    )
    result = _solve_ip_gurobi(
        H, WEIGHTS, syndrome, L, time_limit=10.0, mip_focus=2,
    )

    assert result["status_name"] == "OPTIMAL"
    assert result["has_solution"]
    assert np.array_equal(np.asarray(H @ result["x"]).ravel() & 1,
                          syndrome)
    assert result["final_obj"] == pytest.approx(expected_cost, abs=1e-9)
    assert tuple(result["pred_obs"].tolist()) in expected_obs


def test_gurobi_ip_reports_an_infeasible_parity_system():
    """An impossible syndrome must not produce an incumbent correction."""
    impossible_h = sp.csr_matrix((1, 1), dtype=np.uint8)
    result = _solve_ip_gurobi(
        impossible_h,
        np.array([1.0]),
        np.array([1], dtype=np.uint8),
        sp.csr_matrix((1, 1), dtype=np.uint8),
        time_limit=10.0,
    )
    assert result["status_name"] == "INFEASIBLE"
    assert not result["has_solution"]
    assert result["x"] is None


def _system(tmp_path):
    return DecodingSystem.from_matrices(
        H, L, PROBS,
        is_x_detector=np.ones(H.shape[0], dtype=bool),
        init_basis="X",
        workdir=tmp_path,
    )


def _flat_s4_config(*, init_dets_only=False):
    config = TelescopeConfig()
    config.s4.init_dets_only = init_dets_only
    config.s4.time_limit_subdem_s = 10.0
    config.s4.full_time_limit_s = 10.0
    config.s4.gap_threshold = 0.0
    config.s4.full_gap_threshold = 0.0
    return config._flatten()


def test_s4_init_dets_only_matches_exhaustive_full_solve(tmp_path):
    """The one-shot exact S4 route must preserve cost and observables."""
    system = _system(tmp_path)
    cfg = _flat_s4_config(init_dets_only=True)
    s4._ip_pool_init(cfg, system.npz_path_for("init_dets"), None)

    for gid, syndrome in enumerate((
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([1, 1, 1], dtype=np.uint8),
    )):
        expected_cost, _best, expected_obs = _exhaustive(
            H, WEIGHTS, syndrome, L,
        )
        # These weights give a unique observable class at every tested
        # optimum, so it is safe to use that class as ground truth here.
        assert len(expected_obs) == 1
        true_obs = np.asarray(next(iter(expected_obs)), dtype=np.uint8)
        result = s4._solve_one_ip_shot((syndrome, true_obs, gid))

        assert result["ok"]
        assert result["status"] in {"TRIVIAL", "OPTIMAL"}
        assert result["stage"] == ("trivial" if not syndrome.any()
                                   else "full")
        assert result["cost"] == pytest.approx(expected_cost, abs=1e-9)
        assert np.array_equal(result["obs_pred"], true_obs)
        assert result["le"] is False


def test_s4_sub_dem_matches_exhaustive_restricted_problem(tmp_path):
    """An accepted one-hop result must be optimal on its selected sub-DEM."""
    system = _system(tmp_path)
    cfg = _flat_s4_config(init_dets_only=False)
    s4._ip_pool_init(cfg, system.npz_path_for("original"), None)
    syndrome = np.array([1, 1, 0], dtype=np.uint8)

    dem = DEMData(H, L, PROBS)
    S_idx, D_idx = build_sub_dem_indices(dem, syndrome, hops=1)
    expected_cost, _best, expected_obs = _exhaustive(
        H[D_idx, :][:, S_idx], WEIGHTS[S_idx], syndrome[D_idx], L[:, S_idx],
    )
    assert len(expected_obs) == 1
    true_obs = np.asarray(next(iter(expected_obs)), dtype=np.uint8)

    result = s4._solve_one_ip_shot((syndrome, true_obs, 17))
    assert result["ok"]
    assert result["stage"] == "sub"
    assert result["status"] == "OPTIMAL"
    assert result["cost"] == pytest.approx(expected_cost, abs=1e-9)
    assert np.array_equal(result["obs_pred"], true_obs)
    assert result["le"] is False


def test_s4_escalates_a_rejected_sub_solve_to_exact_full_dem(
        tmp_path, monkeypatch):
    """A rejected sub-DEM incumbent must be replaced by the full solve."""
    system = _system(tmp_path)
    cfg = _flat_s4_config(init_dets_only=False)
    s4._ip_pool_init(cfg, system.npz_path_for("original"), None)
    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    expected_cost, _best, expected_obs = _exhaustive(
        H, WEIGHTS, syndrome, L,
    )
    assert len(expected_obs) == 1
    true_obs = np.asarray(next(iter(expected_obs)), dtype=np.uint8)

    real_solve = s4._solve_ip_gurobi
    calls = []

    def reject_sub_then_solve_full(*args, **kwargs):
        calls.append(args[0].shape)
        if len(calls) == 1:
            return {
                "status_name": "TIME_LIMIT",
                "has_solution": False,
                "mip_gap": None,
            }
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(s4, "_solve_ip_gurobi", reject_sub_then_solve_full)
    result = s4._solve_one_ip_shot((syndrome, true_obs, 23))

    assert len(calls) == 2
    assert calls[1] == H.shape
    assert result["ok"]
    assert result["stage"] == "full"
    assert result["sub_status"] == "TIME_LIMIT"
    assert result["status"] == "OPTIMAL"
    assert result["cost"] == pytest.approx(expected_cost, abs=1e-9)
    assert np.array_equal(result["obs_pred"], true_obs)
    assert result["le"] is False


def test_public_decoder_runs_s4_in_worker_process(tmp_path):
    """Exercise S4 through the public facade, worker pool, and NPZ loader."""
    syndromes = np.asarray(
        list(product((0, 1), repeat=H.shape[0])), dtype=np.uint8,
    )
    true_obs = []
    for syndrome in syndromes:
        _cost, _best, observations = _exhaustive(H, WEIGHTS, syndrome, L)
        assert len(observations) == 1
        true_obs.append(next(iter(observations)))
    true_obs = np.asarray(true_obs, dtype=np.uint8)

    config = TelescopeConfig()
    config.s1.enabled = False
    config.s2.enabled = False
    config.s3.enabled = False
    config.s4.enabled = True
    config.s4.n_procs = 1
    config.s4.init_dets_only = True
    config.s4.full_time_limit_s = 10.0
    config.s4.full_gap_threshold = 0.0
    # When the caller selected a license explicitly, make sure the worker
    # receives that exact path. Otherwise Gurobi performs its normal lookup.
    config.s4.license_file = os.environ.get("GRB_LICENSE_FILE")

    with TelescopingDecoder.from_matrices(
        H, L, PROBS,
        is_x_detector=np.ones(H.shape[0], dtype=bool),
        init_basis="X",
        config=config,
        workdir=tmp_path,
    ) as decoder:
        result = decoder.decode(
            syndromes,
            true_obs=true_obs,
            shot_ids=np.arange(len(syndromes), dtype=np.uint64),
        )

    assert np.all(result.stage == Stage.S4)
    assert result.converged.all()
    assert np.array_equal(result.obs_pred, true_obs)
    assert not result.le.any()
    assert result.diagnostics["s4"]["n_ok"] == len(syndromes)
    assert result.diagnostics["n_nc"] == 0


def test_five_sampled_150_code_shots_solve_in_production_s4():
    """Run five deterministic bundled-code shots through S4 and nothing else.

    The bundled model exceeds Gurobi's restricted-license size cap. Machines
    without an explicitly configured full license skip this integration test;
    a supplied but unusable license is a test failure.
    """
    root_license = Path(MATRICES_NPZ).parent.parent.parent / "gurobi.lic"
    configured_license = os.environ.get("GRB_LICENSE_FILE")
    if configured_license is None and root_license.is_file():
        configured_license = str(root_license)

    dem = stim.DetectorErrorModel.from_file(DEM_FILE)
    detectors, true_obs, _ = dem.compile_sampler(seed=20260820).sample(
        shots=5, bit_packed=False,
    )
    detectors = detectors.astype(np.uint8)
    true_obs = true_obs.astype(np.uint8)

    config = TelescopeConfig()
    config.s1.enabled = False
    config.s2.enabled = False
    config.s3.enabled = False
    config.s4.enabled = True
    config.s4.n_procs = 5
    config.s4.license_file = configured_license

    with TelescopingDecoder.from_npz(
        matrices_npz=MATRICES_NPZ,
        config=config,
    ) as decoder:
        result = decoder.decode(
            detectors,
            true_obs=true_obs,
            shot_ids=np.arange(5, dtype=np.uint64),
        )

    s4_diag = result.diagnostics.get("s4", {})
    if s4_diag.get("n_no_license"):
        if configured_license is not None:
            pytest.fail(
                f"configured Gurobi license was unusable: "
                f"{s4_diag.get('status_hist')}"
            )
        pytest.skip("production S4 model needs a full Gurobi license")

    assert np.all(result.stage == Stage.S4)
    assert result.label.tolist() == ["IP_sub_OPTIMAL"] * 5
    assert result.converged.all()
    assert np.array_equal(result.obs_pred, true_obs)
    assert not result.le.any()
    assert s4_diag["n_ok"] == 5
    assert s4_diag["n_giveup_uncertified"] == 0
    assert result.diagnostics["n_nc"] == 0
