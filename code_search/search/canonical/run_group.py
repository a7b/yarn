"""Per-group canonical brute-force search driver (the unit of work).

GPU-backend-agnostic: the same code runs on a local GPU (validation) and on
remote GPU workers (scale). The funnel, per group:

    brute full-rank pool (CPU)
      → classical sqetch screen of A=[[a0,a1]] and B=[[b0,b1]] independently
        (d(ker A_bin), d(ker B_bin) upper-bound the quantum distance), keep
        top-N each by classical distance ≥ target
      → pair survivors, build canonical code, keep single-orbit (k=|G|)
      → quantum sqetch screen  (SCREEN — default, first)
      → BP+OSD confirm         (CONFIRM — only on sqetch-passers)
      → save passer (canonical logical basis + rich JSON)

Every sqetch run records ``k_sub`` alongside ``num_trials`` (always).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.classical_code import weight_matrix
from core.dist.classical import estimate_classical_distance_sqetch
from core.dist.quantum_bposd import estimate_quantum_distances_bposd
from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
from core.f2_fast import screen_basis
from core.group import GroupData, canonicalize
from core.quantum_code import build_Hx, build_Hz, quantum_check_weights
from search.canonical.build import (
    NotSingleOrbit,
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.groups import GroupSpec
from search.canonical.save import save_passed_code
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
    enumerate_weight_elements,
)

try:
    from circuit.spacetime.distance import max_k_sub_for
except Exception:                                            # pragma: no cover
    def max_k_sub_for(n_errors, *, margin_bytes=4096):       # fallback: no cap
        return 1 << 30


@dataclass
class ScreenParams:
    """All knobs for one group's search (mirrors the YAML config)."""

    target: int   # CLASSICAL screen threshold (keep A/B with classical d>=target)
    # QUANTUM save/keep floor (None ⇒ target). Set BELOW target to keep the best
    # codes found even when the exact target isn't reached (d_quantum <= target).
    q_floor: Optional[int] = None
    anchor_weight: int = 3
    free_weight: int = 3
    # classical sqetch screen (sqetch only — no bposd on the classical side)
    cl_num_trials: int = 5_000_000
    cl_k_sub: int = 0                  # 0 ⇒ auto = ker dim (=|G|)
    cl_batch_size: int = 50_000
    max_A_pool: int = 200
    max_B_pool: int = 200
    # quantum sqetch screen (SCREEN, first)
    q_num_trials: int = 200_000
    q_k_sub: int = 192
    q_batch_size: int = 5000
    q_strategy: str = "auto"
    max_quantum_pass: int = 50
    # BP+OSD confirm (only on sqetch-passers)
    bposd_num_trials: int = 30_000
    bposd_n_workers: int = 32
    bposd_osd_order: int = 5
    # explicit quantum CERTIFY (heavy multi-seed sqetch on screen+confirm
    # passers only — the saved dx/dz are these certified values, not the
    # lighter screen values). Mirrors the classical stress test.
    certify_enabled: bool = True
    certify_num_trials: int = 5_000_000
    certify_k_sub: int = 192
    certify_batch_size: int = 200_000
    certify_seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    # check-weight gate (None = no cap)
    max_check_weight: Optional[int] = None
    # GPU device(s)
    devices: Optional[list] = None
    # Bounding knobs (validation / chunked fan-out; None = full brute)
    anchor_limit: Optional[int] = None     # cap pool anchors used
    free_limit: Optional[int] = None       # cap free-column elements / anchor
    # Early-stop: stop the classical screen once max_*_pool passers are found
    # (we only need *enough* d>=target classical codes — higher d doesn't help
    # once the partner side caps the quantum distance). Set False to force a
    # full ranked brute. ``cl_scan_cap`` bounds total candidates scanned/side
    # so a barren group can't run forever (None = unbounded full brute).
    cl_early_stop: bool = True
    cl_scan_cap: Optional[int] = None


def _device0(params: ScreenParams) -> int:
    return (params.devices[0] if params.devices else 0)


def _classical_k_sub(params: ScreenParams, gd: GroupData) -> int:
    """Resolve classical k_sub: 0 ⇒ ker dim (|G|), capped by GPU ceiling."""
    ker_dim = gd.n                                   # full-rank A_bin ⇒ ker = |G|
    want = params.cl_k_sub if params.cl_k_sub > 0 else ker_dim
    return max(1, min(want, ker_dim, max_k_sub_for(2 * gd.n)))


def _passes(d, target: int) -> bool:
    """Distance PASS: nothing-found (None) OR >= target (strict-< early stop)."""
    return d is None or d >= target


# ─────────────────────────────────────────────────────────────────
# Stage 1: classical sqetch screen (per side, independent)
# ─────────────────────────────────────────────────────────────────


def screen_classical_side(
    gd: GroupData, pool: List[tuple], params: ScreenParams, side: str,
) -> List[dict]:
    """Screen A=[[a0,a1]] (side='A') or B=[[b0,b1]] (side='B') candidates.

    Enumerates a1/b1 over the (brute) full-rank ``pool`` and a0/b0 over all
    weight-``free_weight`` elements; sqetch-screens classical distance; keeps
    the top ``max_*_pool`` by distance among those with ``d >= target``.

    Returns survivor dicts: ``{"free": tuple, "anchor": tuple, "d": int|None}``.
    """
    k_sub = _classical_k_sub(params, gd)
    dev = _device0(params)
    keep = params.max_A_pool if side == "A" else params.max_B_pool
    bin_for = a_bin_for if side == "A" else b_bin_for

    anchors = pool if params.anchor_limit is None else pool[: params.anchor_limit]
    survivors: List[dict] = []
    scanned_total = 0
    for anchor in anchors:
        if params.cl_early_stop and len(survivors) >= keep:
            break
        scanned_this_anchor = 0
        for free in enumerate_weight_elements(gd, params.free_weight):
            if params.free_limit is not None and scanned_this_anchor >= params.free_limit:
                break
            if params.cl_scan_cap is not None and scanned_total >= params.cl_scan_cap:
                break
            scanned_this_anchor += 1
            scanned_total += 1
            M_bin = bin_for(free, anchor, gd)
            d = estimate_classical_distance_sqetch(
                M_bin,
                num_trials=params.cl_num_trials,
                d_target=params.target,
                k_sub=k_sub,
                batch_size=params.cl_batch_size,
                devices=[dev],
            )
            if _passes(d, params.target):
                survivors.append({"free": tuple(free), "anchor": tuple(anchor),
                                  "d": (None if d is None else int(d)),
                                  "k_sub": k_sub})
                # Early stop: we only need `keep` d>=target classical codes.
                if params.cl_early_stop and len(survivors) >= keep:
                    break
        if params.cl_scan_cap is not None and scanned_total >= params.cl_scan_cap:
            break
    # Rank by distance (None ⇒ best). With early-stop this is just ordering
    # the first `keep` passers; on a full brute it selects the top `keep`.
    survivors.sort(key=lambda s: (-(s["d"] if s["d"] is not None else 1 << 30)))
    return survivors[:keep]


# ─────────────────────────────────────────────────────────────────
# Stage 2: pair + quantum sqetch screen + bposd confirm + save
# ─────────────────────────────────────────────────────────────────


def _run_record(stage, backend, *, num_trials, k_sub=None, batch_size=None,
                dx=None, dz=None, d=None, osd_order=None, seed=None) -> dict:
    """Build a distance-run record — sqetch stages ALWAYS carry k_sub."""
    rec = {"stage": stage, "backend": backend, "num_trials": int(num_trials)}
    if backend == "sqetch":
        rec["k_sub"] = int(k_sub)                    # mandatory for sqetch
        if batch_size is not None:
            rec["batch_size"] = int(batch_size)
    if osd_order is not None:
        rec["osd_order"] = int(osd_order)
    if seed is not None:
        rec["seed"] = int(seed)
    # Only attach the distance fields relevant to the stage (classical → d;
    # quantum → dx/dz).
    if d is not None:
        rec["d"] = d
    if dx is not None:
        rec["dx"] = dx
    if dz is not None:
        rec["dz"] = dz
    return rec


def certify_quantum(code, basis, params: ScreenParams, dev: int, k_sub: int):
    """Heavy multi-seed sqetch certification of quantum dx/dz.

    Runs ``len(certify_seeds)`` independent sqetch passes (each
    ``certify_num_trials`` at ``k_sub``, strict-< early stop at ``target``).
    Returns ``(cert_dx, cert_dz, runs)`` where the certified distances are the
    **min over seeds** (most conservative). If any seed finds a logical of
    weight < target, the corresponding certified value is < target → the caller
    treats the code as refuted.
    """
    cert_dx = None
    cert_dz = None
    runs: List[dict] = []

    def _mn(cur, v):
        return cur if v is None else (v if cur is None else min(cur, v))

    qf = params.q_floor if params.q_floor is not None else params.target
    for seed in params.certify_seeds:
        dx, dz = estimate_quantum_distances_sqetch(
            code["Hx"], code["Hz"], Lx=basis["Lx"], Lz=basis["Lz"],
            num_trials=params.certify_num_trials, d_target=qf,
            return_logical=False, devices=[dev], k_sub=k_sub,
            batch_size=params.certify_batch_size, seed=int(seed),
        )
        cert_dx = _mn(cert_dx, dx)
        cert_dz = _mn(cert_dz, dz)
        runs.append(_run_record("certify", "sqetch",
                                num_trials=params.certify_num_trials, k_sub=k_sub,
                                batch_size=params.certify_batch_size,
                                dx=dx, dz=dz, seed=int(seed)))
    return cert_dx, cert_dz, runs


def _evaluate_one_pair(gd, A, B, params: ScreenParams, dev, q_k_sub, out_dir, *,
                       gap_expr, tag, provenance):
    """Evaluate one (A,B) pair: screen→confirm→certify→save.

    Returns ``(code_dir|None, dx, dz, single_orbit)`` where dx/dz are the
    quantum-screen distances (for best-seen tracking) and ``single_orbit`` is
    True iff k==|G| (the pair was actually screened).
    """
    qf = params.q_floor if params.q_floor is not None else params.target
    A_ring = [[canonicalize(A["free"]), canonicalize(A["anchor"])]]
    B_ring = [[canonicalize(B["free"]), canonicalize(B["anchor"])]]
    if params.max_check_weight is not None:
        cw = quantum_check_weights(weight_matrix(A_ring), weight_matrix(B_ring))
        if max(cw["Hx_check_weight"], cw["Hz_check_weight"]) > params.max_check_weight:
            return None, None, None, False

    Hx = build_Hx(A_ring, B_ring, gd)
    Hz = build_Hz(A_ring, B_ring, gd)
    Lx, Lz, k = screen_basis(Hx, Hz)                 # fast bit-packed basis (~30x)
    if k != gd.n:                                    # single-orbit gate
        return None, None, None, False

    # SCREEN (sqetch first) — gate on the quantum floor qf
    dx, dz = estimate_quantum_distances_sqetch(
        Hx, Hz, Lx=Lx, Lz=Lz, num_trials=params.q_num_trials,
        d_target=qf, return_logical=False, devices=[dev],
        strategy=params.q_strategy, k_sub=q_k_sub, batch_size=params.q_batch_size)
    if not (_passes(dx, qf) and _passes(dz, qf)):
        return None, dx, dz, True

    code = build_canonical_code(A["free"], A["anchor"], B["free"], B["anchor"], gd)
    try:
        basis = canonical_logical_basis(code, gd)
    except NotSingleOrbit:
        return None, dx, dz, True

    # CONFIRM (bposd, passers only)
    dx_b, dz_b = estimate_quantum_distances_bposd(
        code["Hx"], code["Hz"], Lx=basis["Lx"], Lz=basis["Lz"],
        num_trials=params.bposd_num_trials, n_workers=params.bposd_n_workers,
        d_target=qf, osd_order=params.bposd_osd_order)
    if not (_passes(dx_b, qf) and _passes(dz_b, qf)):
        return None, dx, dz, True

    # CERTIFY (heavy multi-seed sqetch)
    cert_runs: List[dict] = []
    cdx = cdz = None
    if params.certify_enabled:
        cert_k_sub = max(1, min(params.certify_k_sub, max_k_sub_for(5 * gd.n)))
        cdx, cdz, cert_runs = certify_quantum(code, basis, params, dev, cert_k_sub)
        if not (_passes(cdx, qf) and _passes(cdz, qf)):
            return None, dx, dz, True

    def _mind(*vals):
        vs = [v for v in vals if v is not None]
        return min(vs) if vs else None
    save_dx = _mind(dx, dx_b, cdx)
    save_dz = _mind(dz, dz_b, cdz)
    if not (_passes(save_dx, qf) and _passes(save_dz, qf)):
        return None, dx, dz, True

    runs = [
        _run_record("classical_A", "sqetch", num_trials=params.cl_num_trials,
                    k_sub=A["k_sub"], batch_size=params.cl_batch_size, d=A["d"]),
        _run_record("classical_B", "sqetch", num_trials=params.cl_num_trials,
                    k_sub=B["k_sub"], batch_size=params.cl_batch_size, d=B["d"]),
        _run_record("quantum_screen", "sqetch", num_trials=params.q_num_trials,
                    k_sub=q_k_sub, batch_size=params.q_batch_size, dx=dx, dz=dz),
        _run_record("bposd_confirm", "bposd", num_trials=params.bposd_num_trials,
                    osd_order=params.bposd_osd_order, dx=dx_b, dz=dz_b),
    ] + cert_runs
    code_dir = save_passed_code(
        out_dir, gd=gd, gap_expr=gap_expr, tag=tag, code=code, basis=basis,
        dx=save_dx, dz=save_dz, classical_dA=A["d"], classical_dB=B["d"],
        distance_runs=runs, provenance=provenance)
    return str(code_dir), dx, dz, True


def _track_best(cur, v):
    # highest per-pair quantum distance seen (closest to target)
    return cur if v is None else (v if cur is None else max(cur, v))


def pair_and_verify(
    gd: GroupData, A_survivors: List[dict], B_survivors: List[dict],
    params: ScreenParams, out_dir: Path, *,
    gap_expr: str, tag: str, provenance: Optional[dict] = None,
) -> dict:
    """Pair every A×B (full grids supplied), screen→confirm→certify→save."""
    dev = _device0(params)
    q_k_sub = max(1, min(params.q_k_sub, max_k_sub_for(5 * gd.n)))
    passed: List[str] = []
    n_pairs = 0
    n_single = 0
    best_dx = best_dz = None
    for A in A_survivors:
        if len(passed) >= params.max_quantum_pass:
            break
        for B in B_survivors:
            if len(passed) >= params.max_quantum_pass:
                break
            n_pairs += 1
            cd, dx, dz, so = _evaluate_one_pair(
                gd, A, B, params, dev, q_k_sub, out_dir,
                gap_expr=gap_expr, tag=tag, provenance=provenance)
            if so:
                n_single += 1
                best_dx = _track_best(best_dx, dx)
                best_dz = _track_best(best_dz, dz)
            if cd:
                passed.append(cd)
    return {"passed": passed, "n_pairs": n_pairs, "n_single_orbit": n_single,
            "best_dx_seen": best_dx, "best_dz_seen": best_dz}


# ─────────────────────────────────────────────────────────────────
# Top-level per-group entry
# ─────────────────────────────────────────────────────────────────


def run_group(
    group: GroupSpec, params: ScreenParams, out_dir: Path,
    *, gd=None, provenance: Optional[dict] = None,
) -> dict:
    """Run the full funnel for one group. Returns a summary dict.

    ``gd`` may be a real :class:`core.group.GroupData` (needs GAP) or any
    GAP-free duck-typed equivalent. If ``None``, a ``GroupData`` is
    built from ``group.gap_expr`` (requires GAP).
    """
    if gd is None:
        gd = GroupData(group.gap_expr)
    out_dir = Path(out_dir)

    pool = build_full_rank_block_pool_brute(gd, params.anchor_weight,
                                            force_identity=True)
    if not pool:
        summary = {"group": group.tag, "order": gd.n, "pool_size": 0,
                   "note": f"no weight-{params.anchor_weight} units; bump weight",
                   "passed": [], "n_pairs": 0, "n_single_orbit": 0,
                   "best_dx_seen": None, "best_dz_seen": None,
                   "n_A_survivors": 0, "n_B_survivors": 0, "target": params.target}
        _save_stats(out_dir, summary)
        return summary

    A_surv = screen_classical_side(gd, pool, params, "A")
    B_surv = screen_classical_side(gd, pool, params, "B")
    result = pair_and_verify(
        gd, A_surv, B_surv, params, out_dir,
        gap_expr=group.gap_expr, tag=group.tag, provenance=provenance,
    )
    summary = {
        "group": group.tag, "order": gd.n, "pool_size": len(pool),
        "target": params.target,
        "n_A_survivors": len(A_surv), "n_B_survivors": len(B_surv),
        "A_d_classical": sorted({s["d"] for s in A_surv},
                                key=lambda d: (d is None, 0 if d is None else d)),
        "B_d_classical": sorted({s["d"] for s in B_surv},
                                key=lambda d: (d is None, 0 if d is None else d)),
        **result,
    }
    # Record statistics ALWAYS — even when nothing passes (best dx/dz seen tells
    # us how close this group got).
    _save_stats(out_dir, summary)
    return summary


def _save_stats(out_dir: Path, summary: dict) -> None:
    """Persist a per-group stats JSON (passers list trimmed to dir names)."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = dict(summary)
    s["n_passed"] = len(s.get("passed", []))
    (out_dir / f"_stats_{summary['group']}.json").write_text(json.dumps(s, indent=2))


# ─────────────────────────────────────────────────────────────────
# Streaming driver: interleave screen + pair (pair as survivors are found)
# ─────────────────────────────────────────────────────────────────


def iter_side_survivors(gd, pool, params: ScreenParams, side: str):
    """Generator yielding classical-d≥target survivors AS FOUND (scan order).

    Same screen as :func:`screen_classical_side` but streamed — lets the caller
    start pairing immediately instead of waiting for the whole pool to fill.
    """
    k_sub = _classical_k_sub(params, gd)
    dev = _device0(params)
    bin_for = a_bin_for if side == "A" else b_bin_for
    anchors = pool if params.anchor_limit is None else pool[: params.anchor_limit]
    scanned = 0
    for anchor in anchors:
        cnt = 0
        for free in enumerate_weight_elements(gd, params.free_weight):
            if params.free_limit is not None and cnt >= params.free_limit:
                break
            if params.cl_scan_cap is not None and scanned >= params.cl_scan_cap:
                return
            cnt += 1
            scanned += 1
            d = estimate_classical_distance_sqetch(
                bin_for(free, anchor, gd), num_trials=params.cl_num_trials,
                d_target=params.target, k_sub=k_sub,
                batch_size=params.cl_batch_size, devices=[dev])
            if _passes(d, params.target):
                yield {"free": tuple(free), "anchor": tuple(anchor),
                       "d": (None if d is None else int(d)), "k_sub": k_sub}


def run_group_streaming(group: GroupSpec, params: ScreenParams, out_dir,
                        *, gd=None, provenance: Optional[dict] = None) -> dict:
    """Per-group search that PAIRS AS SURVIVORS ARE FOUND (no pool-fill wait).

    Pulls A/B survivors from the streamed screen; each new A pairs against all
    accumulated B (and vice-versa) — every (A,B) evaluated exactly once. Passers
    are screened/confirmed/certified and saved the moment they're found, so a
    verified code can surface within minutes of the first survivors rather than
    after the full 1000-deep pool fills. Stops at ``max_quantum_pass`` passers,
    pools full (``max_A_pool``×``max_B_pool``), or the scan cap.
    """
    if gd is None:
        gd = GroupData(group.gap_expr)
    out_dir = Path(out_dir)
    pool = build_full_rank_block_pool_brute(gd, params.anchor_weight,
                                            force_identity=True)
    if not pool:
        summary = {"group": group.tag, "order": gd.n, "pool_size": 0,
                   "note": f"no weight-{params.anchor_weight} units",
                   "passed": [], "n_pairs": 0, "n_single_orbit": 0,
                   "best_dx_seen": None, "best_dz_seen": None,
                   "n_A_survivors": 0, "n_B_survivors": 0, "target": params.target}
        _save_stats(out_dir, summary)
        return summary

    import sys
    import time as _time
    dev = _device0(params)
    q_k_sub = max(1, min(params.q_k_sub, max_k_sub_for(5 * gd.n)))
    A_it = iter_side_survivors(gd, pool, params, "A")
    B_it = iter_side_survivors(gd, pool, params, "B")
    A_surv: List[dict] = []
    B_surv: List[dict] = []
    passed: List[str] = []
    n_pairs = 0
    n_single = 0
    best_dx = best_dz = None
    a_done = b_done = False
    _t0 = _time.time()
    _last_log = [0.0]

    def _log(force=False):
        now = _time.time()
        if force or now - _last_log[0] > 30:     # progress line every ~30 s
            _last_log[0] = now
            print(f"[{group.tag} |G|={gd.n}] t={int(now-_t0)}s A={len(A_surv)} "
                  f"B={len(B_surv)} pairs={n_pairs} single={n_single} "
                  f"pass={len(passed)} best=({best_dx},{best_dz})", flush=True)

    def _pair_new(new_entry, others):
        nonlocal n_pairs, n_single, best_dx, best_dz
        for o in others:
            if len(passed) >= params.max_quantum_pass:
                return
            a, b = (new_entry, o) if new_entry["_side"] == "A" else (o, new_entry)
            n_pairs += 1
            cd, dx, dz, so = _evaluate_one_pair(
                gd, a, b, params, dev, q_k_sub, out_dir,
                gap_expr=group.gap_expr, tag=group.tag, provenance=provenance)
            if so:
                n_single += 1
                best_dx = _track_best(best_dx, dx)
                best_dz = _track_best(best_dz, dz)
            if cd:
                print(f"[{group.tag}] *** PASSER #{len(passed)+1}: saved {cd}",
                      flush=True)
                passed.append(cd)
            _log()

    while not (a_done and b_done) and len(passed) < params.max_quantum_pass:
        progressed = False
        if not a_done and len(A_surv) < params.max_A_pool:
            try:
                a = next(A_it); a["_side"] = "A"; progressed = True
                _pair_new(a, B_surv)
                A_surv.append(a)
            except StopIteration:
                a_done = True
        if len(passed) >= params.max_quantum_pass:
            break
        if not b_done and len(B_surv) < params.max_B_pool:
            try:
                b = next(B_it); b["_side"] = "B"; progressed = True
                _pair_new(b, A_surv)
                B_surv.append(b)
            except StopIteration:
                b_done = True
        _log()                                   # progress even while screening
        if len(A_surv) >= params.max_A_pool and len(B_surv) >= params.max_B_pool:
            break
        if not progressed:
            break

    _log(force=True)
    summary = {"group": group.tag, "order": gd.n, "pool_size": len(pool),
               "target": params.target, "n_A_survivors": len(A_surv),
               "n_B_survivors": len(B_surv), "passed": passed, "n_pairs": n_pairs,
               "n_single_orbit": n_single, "best_dx_seen": best_dx,
               "best_dz_seen": best_dz}
    _save_stats(out_dir, summary)
    return summary
