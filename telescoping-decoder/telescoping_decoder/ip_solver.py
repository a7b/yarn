"""Integer-programming helpers for sub-DEM and full-DEM decoding.

Uses Gurobi; the license comes from ``S4Config.license_file`` or Gurobi's
standard lookup. The MIP formulation:

    minimize  sum_i w_i x_i
    s.t.      H x - 2 y = s    (over Z)
              x_i ∈ {0,1},  y_d ∈ Z_{≥0}, y_d ≤ ⌈|row d|/2⌉

`w_i = log((1 - p_i) / p_i)` for ML decoding.
"""
import os
import time

import numpy as np
import scipy.sparse as sp



_GRB_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
    13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


class DEMData:
    def __init__(self, h, l, probs):
        self.h_csr = h.tocsr()
        self.h_csc = h.tocsc()
        self.l_csr = l.tocsr()
        self.probs = probs
        p_clip = np.clip(probs, 1e-300, 0.5 - 1e-12)
        self.weights = np.log((1.0 - p_clip) / p_clip)
        self.n_dets, self.n_errors = h.shape
        self.n_obs = l.shape[0]


def load_dem(path):
    d = np.load(path)
    h = sp.csr_matrix(
        (d["h_data"], d["h_indices"], d["h_indptr"]),
        shape=tuple(d["h_shape"]),
    )
    l = sp.csr_matrix(
        (d["l_data"], d["l_indices"], d["l_indptr"]),
        shape=tuple(d["l_shape"]),
    )
    return DEMData(h, l, d["probs"])


def build_sub_dem_indices(dem, syndrome, hops):
    """k-hop expansion in the bipartite (detector, error) graph from active syndrome."""
    h_csr_indices = dem.h_csr.indices
    h_csr_indptr = dem.h_csr.indptr
    h_csc_indices = dem.h_csc.indices
    h_csc_indptr = dem.h_csc.indptr

    A = np.flatnonzero(syndrome)
    D_set = set(A.tolist())
    S_set = set()

    for _ in range(hops):
        S_set = set()
        for d in D_set:
            S_set.update(h_csr_indices[h_csr_indptr[d]:h_csr_indptr[d + 1]].tolist())
        D_set = set()
        for i in S_set:
            D_set.update(h_csc_indices[h_csc_indptr[i]:h_csc_indptr[i + 1]].tolist())

    S_idx = np.array(sorted(S_set), dtype=np.int64)
    D_idx = np.array(sorted(D_set), dtype=np.int64)
    return S_idx, D_idx


def _empty_result(n_obs, build_time=0.0):
    return {
        "wall_time": build_time,
        "tff": 0.0, "tbi": 0.0,
        "first_obj": 0.0,
        "final_obj": 0.0,
        "best_bound": 0.0,
        "mip_gap": 0.0,
        "node_count": 0,
        "status_name": "OPTIMAL",
        "n_vars": 0, "n_constraints": 0, "n_x_vars": 0,
        "has_solution": True,
        "x": np.zeros(0, dtype=np.uint8),
        "pred_obs": np.zeros(n_obs, dtype=np.uint8),
        "sub_build_time": build_time,
    }


_GRB_ENV = None


def _gurobi_env():
    """Lazily create a single Gurobi env per process."""
    global _GRB_ENV
    if _GRB_ENV is None:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _GRB_ENV = env
    return _GRB_ENV


def _solve_ip_gurobi(h_sub_csr, weights_sub, syndrome_sub, l_sub, time_limit,
                     mip_focus=1, heuristics=0.5, presolve=2, symmetry=2,
                     threads=1, mip_gap=None):
    import gurobipy as gp
    from gurobipy import GRB

    if not sp.isspmatrix_csr(h_sub_csr):
        h_sub_csr = h_sub_csr.tocsr()
    n_dets, n_x = h_sub_csr.shape

    env = _gurobi_env()
    m = gp.Model("ip", env=env)
    m.Params.TimeLimit = float(time_limit)
    m.Params.Threads = int(threads)
    m.Params.MIPFocus = int(mip_focus)
    m.Params.Heuristics = float(heuristics)
    m.Params.Presolve = int(presolve)
    m.Params.Symmetry = int(symmetry)
    m.Params.OutputFlag = 0
    # When set, Gurobi terminates (reporting OPTIMAL) as soon as the relative
    # MIP gap drops below this tolerance — i.e. "solve only until gap < X"
    # rather than pushing all the way to the default 1e-4 proof of optimality.
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    e = m.addMVar(n_x, vtype=GRB.BINARY, name="e")

    row_supp = np.diff(h_sub_csr.indptr)
    k_ub = np.ceil(row_supp / 2.0).astype(np.float64)
    k = m.addMVar(n_dets, vtype=GRB.INTEGER, lb=0.0, ub=k_ub, name="k")

    # Constraint A x_full = s with x_full = [e; k], A = [H | -2 I]
    neg2I = sp.diags([-2.0], offsets=[0], shape=(n_dets, n_dets), format="csr")
    A = sp.hstack([h_sub_csr.astype(np.float64), neg2I], format="csr")
    x_full = gp.MVar.fromlist(e.tolist() + k.tolist())
    m.addMConstr(A, x_full, GRB.EQUAL, syndrome_sub.astype(np.float64))

    m.setObjective(weights_sub.astype(np.float64) @ e, GRB.MINIMIZE)

    state = {"tff": None, "tbi": None, "first_obj": None}
    t0 = time.time()

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            t = time.time() - t0
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if state["tff"] is None:
                state["tff"] = t
                state["first_obj"] = obj
            state["tbi"] = t

    m.optimize(callback)
    wall = time.time() - t0

    status = int(m.Status)
    has_solution = m.SolCount > 0

    out = {
        "wall_time": wall,
        "tff": state["tff"],
        "tbi": state["tbi"],
        "first_obj": state["first_obj"],
        "final_obj": float(m.ObjVal) if has_solution else None,
        "best_bound": float(m.ObjBound) if status != GRB.INFEASIBLE else None,
        "mip_gap": float(m.MIPGap) if has_solution else None,
        "node_count": int(m.NodeCount),
        "status_name": _GRB_STATUS_NAMES.get(status, f"UNK_{status}"),
        "n_vars": int(n_x + n_dets),
        "n_x_vars": int(n_x),
        "n_constraints": int(n_dets),
        "has_solution": bool(has_solution),
    }

    if has_solution:
        e_vals = (np.rint(np.asarray(e.X)) > 0.5).astype(np.uint8)
        out["x"] = e_vals
        if l_sub is not None:
            pred = np.asarray(l_sub @ e_vals).ravel() % 2
            out["pred_obs"] = pred.astype(np.uint8)
        else:
            out["pred_obs"] = None
    else:
        out["x"] = None
        out["pred_obs"] = None

    m.dispose()
    return out


def run_variant(dem, syndrome, variant, time_limit):
    """variant in {'1hop', '2hop', 'full'}."""
    if variant in ("1hop", "2hop"):
        hops = 1 if variant == "1hop" else 2
        t_build = time.time()
        S_idx, D_idx = build_sub_dem_indices(dem, syndrome, hops=hops)
        build_time = time.time() - t_build

        if S_idx.size == 0 or D_idx.size == 0:
            return _empty_result(dem.n_obs, build_time=build_time)

        h_sub = dem.h_csr[D_idx, :][:, S_idx]
        l_sub = dem.l_csr[:, S_idx]
        weights_sub = dem.weights[S_idx]
        synd_sub = syndrome[D_idx]
    else:
        build_time = 0.0
        h_sub = dem.h_csr
        l_sub = dem.l_csr
        weights_sub = dem.weights
        synd_sub = syndrome

    res = _solve_ip_gurobi(h_sub, weights_sub, synd_sub, l_sub, time_limit)
    res["sub_build_time"] = build_time
    return res
