"""S4: Gurobi integer-programming certification.

Solves the original correlated XYZ DEM, never the GARI system: a 1-hop
sub-DEM MIP first, escalating to the full DEM when the gap rule rejects.
With ``init_dets_only`` the system is instead the init-basis-detectors-only
DEM and a single exact solve replaces the sub/full ladder.

Per-shot statuses: ``stage`` in {"trivial", "sub", "full", "fail"}; a "fail"
means the solver exhausted its budget without an accepted incumbent, so the
shot remains deferred.

``MIPFocus`` is configurable and defaults to 2, which asks Gurobi to emphasize
bound improvement and optimality proofs.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
import scipy.sparse as sp

# ip_solver imports gurobipy only inside its solve functions. Keeping the
# import lazy lets users disable S4 even if their core installation is broken.
from .ip_solver import DEMData, build_sub_dem_indices, _solve_ip_gurobi

_G: dict = {}   # per-worker state, populated by _ip_pool_init


def _ip_le_class(obs_pred, true_obs, cfg):
    """2-bit measured/unmeasured LE class from a predicted-obs vector, using
    an explicit cfg — S4 pool workers carry cfg locally."""
    n_meas = int(getattr(cfg, "n_measured_observables", 0))
    if n_meas <= 0:
        return 0
    diff = np.asarray(obs_pred, np.uint8) != np.asarray(true_obs, np.uint8)
    return (int(bool(diff[:n_meas].any()))
            | (int(bool(diff[n_meas:].any())) << 1))


def _ip_pool_init(cfg_obj, ip_npz_path: str, license_path) -> None:
    """Pool initializer — runs once per worker. Loads the IP system and
    stashes it in `_G` so subsequent `_solve_one_ip_shot` calls in this
    worker can read it. Also sets `GRB_LICENSE_FILE` so gurobipy imports
    pick up the license.

    When cfg.s4_init_dets_only, `ip_npz_path` must be the materialized
    init-dets npz (it carries `init_idx`, recording which original detector
    rows it keeps) so _solve_one_ip_shot can restrict each full-width
    syndrome to them. Otherwise it is the original matrices npz and
    `ip_init_idx` is None."""
    _G['cfg'] = cfg_obj
    if license_path:
        os.environ["GRB_LICENSE_FILE"] = str(license_path)

    mats = np.load(ip_npz_path)
    H = sp.csr_matrix(
        (mats["h_data"], mats["h_indices"], mats["h_indptr"]),
        shape=tuple(mats["h_shape"]),
    )
    L = sp.csr_matrix(
        (mats["l_data"], mats["l_indices"], mats["l_indptr"]),
        shape=tuple(mats["l_shape"]),
    )
    probs = mats["probs"].astype(np.float64)
    if cfg_obj.s4_init_dets_only:
        assert "init_idx" in mats.files, (
            "s4_init_dets_only needs the materialized init-dets npz "
            "(it carries init_idx); got a plain matrices npz")
        _G['ip_init_idx'] = mats["init_idx"].astype(np.int64)
    else:
        _G['ip_init_idx'] = None
    p_clip = np.clip(probs, 1e-300, 0.5 - 1e-12)
    channel_llr = np.log((1.0 - p_clip) / p_clip)
    dem = DEMData(H, L, probs)

    _G['ip_H'] = H
    _G['ip_L'] = L
    _G['ip_channel_llr'] = channel_llr
    _G['ip_dem'] = dem


# Gurobi reports licensing problems as ordinary GurobiErrors, which would
# otherwise land in the generic handler and be reported as
# ``IP_giveup_uncertified`` — indistinguishable from a solver that ran out of
# budget on a genuinely hard shot. These map the common cases to their own
# status so ``diagnostics["s4"]["status_hist"]`` names the real problem.
_LICENSE_STATUSES = (
    ("size-limited", "LICENSE_TOO_SMALL"),   # restricted pip license, >2000 vars
    ("has expired", "LICENSE_EXPIRED"),
    ("expired", "LICENSE_EXPIRED"),
    ("no license", "LICENSE_MISSING"),
    ("failed to obtain a valid license", "LICENSE_MISSING"),
    ("license file", "LICENSE_MISSING"),
)


def _license_status(exc: BaseException):
    """The LICENSE_* status for a Gurobi licensing error, else None."""
    if type(exc).__name__ != "GurobiError":
        return None
    message = str(exc).lower()
    for needle, status in _LICENSE_STATUSES:
        if needle in message:
            return status
    return None


def _solve_one_ip_shot(args):
    """Solve one shot's sub-DEM IP, escalating to full-DEM on gap-reject.

    Args:
        args: (syndrome_u8 (n_det,), true_obs_u8 (n_obs,) or None, gid).

    Returns a per-shot summary dict: {global_idx, ok, le, le_class, cost,
    wall_seconds, stage, status, obs_pred, ...}. le is None when true_obs
    is None.
    """
    syndrome_u8, true_obs_u8, gid = args
    try:
        cfg = _G['cfg']
        mip_focus = int(getattr(cfg, "s4_mip_focus", 2))
        syndrome = np.ascontiguousarray(
            np.asarray(syndrome_u8, dtype=np.uint8))
        true_obs = (np.asarray(true_obs_u8, dtype=np.uint8)
                    if true_obs_u8 is not None else None)
        n_obs = int(_G['ip_L'].shape[0])

        def _le_of(obs_pred):
            if true_obs is None:
                return None, 0
            le = bool(np.any(obs_pred != true_obs))
            return le, _ip_le_class(obs_pred, true_obs, cfg)

        # init-dets-only: syndromes arrive full-width (original detector
        # order); restrict to the init-family rows the IP system was built on.
        init_idx = _G.get('ip_init_idx')
        if init_idx is not None:
            syndrome = np.ascontiguousarray(syndrome[init_idx])

        H = _G['ip_H']
        L = _G['ip_L']
        channel_llr = _G['ip_channel_llr']
        dem = _G['ip_dem']

        # ---- init-dets-only: ONE full-(init)-DEM solve, no sub-DEM --------
        # The init-basis DEM is small enough (~10^4 cols) that Gurobi solves
        # it to proven optimality in seconds, so the 1-hop sub-DEM heuristic
        # is both unnecessary and harmful here: it leaves shots stuck at the
        # time limit and accepts sub-optimal (possibly wrong-coset)
        # incumbents under the looser sub-DEM gap rule. Measured over the
        # shots that reached this stage on a distance-10 code: the exact
        # solve proved optimality on all of them versus 95% via the sub-DEM
        # route, in roughly a tenth of the total wall time, and the sub-DEM
        # route returned a sub-optimal correction on a third of them. Accept
        # OPTIMAL, or a timeout incumbent
        # with gap < s4_full_gap_threshold; otherwise it is an
        # uncertified give-up (stage="fail").
        if init_idx is not None:
            t0 = time.perf_counter()
            if not syndrome.any():
                wall = time.perf_counter() - t0
                obs_pred = np.zeros(n_obs, dtype=np.uint8)
                le, le_class = _le_of(obs_pred)
                return {
                    "global_idx": gid, "ok": True, "le": le,
                    "le_class": le_class, "cost": 0.0,
                    "wall_seconds": wall, "stage": "trivial",
                    "status": "TRIVIAL", "obs_pred": obs_pred,
                }

            res = _solve_ip_gurobi(
                dem.h_csr, dem.weights, syndrome, dem.l_csr,
                time_limit=cfg.s4_full_time_limit_s,
                mip_focus=mip_focus,
            )
            wall = time.perf_counter() - t0
            status_name = res["status_name"]
            has_solution = bool(res["has_solution"])
            mip_gap = res.get("mip_gap")
            accept = has_solution and (
                status_name == "OPTIMAL"
                or (mip_gap is not None
                    and mip_gap < cfg.s4_full_gap_threshold)
            )

            if accept:
                x_vec = res["x"].astype(np.uint8)
                obs_pred = np.asarray(
                    L @ x_vec.astype(np.int32) % 2
                ).ravel().astype(np.uint8)
                le, le_class = _le_of(obs_pred)
                cost = float(np.sum(np.abs(channel_llr) * x_vec))
                return {
                    "global_idx": gid, "ok": True, "le": le,
                    "le_class": le_class, "cost": cost,
                    "wall_seconds": wall, "stage": "full",
                    "status": status_name,
                    "mip_gap": mip_gap, "obs_pred": obs_pred,
                }

            # Reject (no incumbent, or timeout with a wide gap) ->
            # uncertified give-up.
            return {
                "global_idx": gid, "ok": False, "le": None, "le_class": 0,
                "status": status_name, "mip_gap": mip_gap,
                "wall_seconds": wall, "stage": "fail", "obs_pred": None,
            }
        # ---- end init-dets-only -------------------------------------------

        t0 = time.perf_counter()
        S_idx, D_idx = build_sub_dem_indices(
            dem, syndrome, hops=cfg.s4_hops
        )

        if S_idx.size == 0 or D_idx.size == 0:
            wall = time.perf_counter() - t0
            # Trivial: zero correction -> obs_pred all-zero, so the LE class
            # is read straight off true_obs.
            obs_pred = np.zeros(n_obs, dtype=np.uint8)
            le, le_class = _le_of(obs_pred)
            return {
                "global_idx": gid, "ok": True, "le": le,
                "le_class": le_class, "cost": 0.0,
                "wall_seconds": wall, "stage": "trivial",
                "status": "TRIVIAL", "obs_pred": obs_pred,
            }

        h_sub = dem.h_csr[D_idx, :][:, S_idx]
        l_sub = dem.l_csr[:, S_idx]
        weights_sub = dem.weights[S_idx]
        synd_sub = syndrome[D_idx]

        res = _solve_ip_gurobi(
            h_sub, weights_sub, synd_sub, l_sub,
            time_limit=cfg.s4_time_limit_subdem_s,
            mip_focus=mip_focus,
        )
        wall = time.perf_counter() - t0

        status_name = res["status_name"]
        has_solution = bool(res["has_solution"])
        mip_gap = res.get("mip_gap")
        accept = has_solution and (
            status_name == "OPTIMAL"
            or (mip_gap is not None
                and mip_gap < cfg.s4_gap_threshold)
        )

        if accept:
            x_sub = res["x"]
            x_full = np.zeros(int(H.shape[1]), dtype=np.uint8)
            x_full[S_idx] = x_sub.astype(np.uint8)
            obs_pred = np.asarray(
                L @ x_full.astype(np.int32) % 2
            ).ravel().astype(np.uint8)
            le, le_class = _le_of(obs_pred)
            cost = float(np.sum(np.abs(channel_llr) * x_full))
            return {
                "global_idx": gid, "ok": True, "le": le,
                "le_class": le_class, "cost": cost,
                "wall_seconds": wall, "stage": "sub",
                "status": status_name,
                "mip_gap": mip_gap, "obs_pred": obs_pred,
            }

        # Escalate to full DEM
        sub_status = status_name
        sub_mip_gap = mip_gap
        t1 = time.perf_counter()
        res_full = _solve_ip_gurobi(
            dem.h_csr, dem.weights, syndrome, dem.l_csr,
            time_limit=cfg.s4_full_time_limit_s,
            mip_focus=mip_focus,
        )
        wall_full = time.perf_counter() - t1
        wall_total = time.perf_counter() - t0

        status_full = res_full["status_name"]
        has_solution_full = bool(res_full["has_solution"])
        mip_gap_full = res_full.get("mip_gap")
        # Full-DEM stage: accept the incumbent only if the solver proved it
        # near-optimal — status=OPTIMAL, or it hit the TimeLimit with an
        # incumbent whose MIPGap < s4_full_gap_threshold. A wide-gap
        # time-limited incumbent is not trustworthy (its coset may be
        # arbitrarily wrong), so it is rejected and the shot is marked an
        # uncertified give-up rather than committing the bad solution.
        # Callers can re-decode rejected shots with a larger
        # full_time_limit_s or more threads.
        accept_full = has_solution_full and (
            status_full == "OPTIMAL"
            or (mip_gap_full is not None
                and mip_gap_full < cfg.s4_full_gap_threshold)
        )

        if accept_full:
            x_full_vec = res_full["x"].astype(np.uint8)
            obs_pred = np.asarray(
                L @ x_full_vec.astype(np.int32) % 2
            ).ravel().astype(np.uint8)
            le, le_class = _le_of(obs_pred)
            cost = float(np.sum(np.abs(channel_llr) * x_full_vec))
            return {
                "global_idx": gid, "ok": True, "le": le,
                "le_class": le_class, "cost": cost,
                "wall_seconds": wall_total, "stage": "full",
                "status": status_full, "sub_status": sub_status,
                "mip_gap": mip_gap_full, "sub_mip_gap": sub_mip_gap,
                "sub_wall_seconds": wall, "full_wall_seconds": wall_full,
                "obs_pred": obs_pred,
            }

        # Both stages rejected
        return {
            "global_idx": gid, "ok": False, "le": None, "le_class": 0,
            "status": status_full, "sub_status": sub_status,
            "mip_gap": mip_gap_full, "sub_mip_gap": sub_mip_gap,
            "wall_seconds": wall_total, "stage": "fail", "obs_pred": None,
        }
    except Exception as e:
        status = _license_status(e)
        if status is None:
            status = "EXCEPTION"
            traceback.print_exc()   # a real solver/model bug: keep the trace
        return {
            "global_idx": int(gid),
            "ok": False, "le": None, "le_class": 0, "stage": "fail",
            "status": status, "obs_pred": None,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


__all__ = ["_ip_pool_init", "_solve_one_ip_shot", "_ip_le_class"]
