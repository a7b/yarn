"""Random-sampling group sweep over the fixed canonical box.

Same search box as :mod:`search.canonical.run_group` (shape ``(1,2)``,
weight-3 entries, ``a1``/``b1`` are identity-containing **units** so
``L[a1]``/``R[b1]`` are full rank → clean single-orbit canonical basis with
``Lx·Lzᵀ = I``), but the inner loop is **randomized sampling with budgets**
instead of an exhaustive brute walk — the strategy adopted for
sweeping *general groups* of a given order:

Per group (see :func:`run_group_sweep`):

1. **Anchor pool** — every weight-3 identity-containing unit (``a1``/``b1``),
   exactly :func:`build_full_rank_block_pool_brute`.
2. **Classical screen** (A side ∥ B side, independent): randomly draw
   ``a0`` (weight-3) and ``a1`` (from the pool); sqetch the classical distance
   of ``ker(A_bin)`` (``d`` upper-bounds the quantum distance); keep
   ``d ≥ target``. Survivor pool capped at ``surv_cap`` per side; at most
   ``cl_sample_budget`` draws per side; **SKIP** the group if a side yields 0
   survivors after ``cl_barren_skip`` draws.
3. **Pairing** (sampled, streamed once both sides have a survivor): draw an
   ``(A, B)`` survivor pair, build ``Hx``/``Hz``, gate ``k == |G|``
   (single-orbit), then a **single** quantum sqetch at the **max** k_sub for
   ``q_num_trials`` with strict-``<`` early stop at ``target``. Save the
   canonical code on a pass. **STOP** the group at ``max_pass`` passers;
   **FAIL** the group at ``pair_fail`` quantum tests with 0 passers; budget
   ``pair_budget`` quantum tests.

There is **no** bposd / multi-seed certify here — the single high-trial,
max-k_sub sqetch is trusted.

Estimators are injected (``classical_estimator`` / ``quantum_estimator``)
so the funnel is testable with no GPU; the defaults are the sqetch backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.f2_fast import screen_basis
from core.group import GroupData, canonicalize
from core.quantum_code import build_Hx, build_Hz
from search.canonical.build import (
    NotSingleOrbit,
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.groups import GroupSpec
from search.canonical.run_group import _passes, _run_record, _track_best
from search.canonical.save import save_passed_code
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
)
from search.sampling._shared.random_ring_element import random_ring_element

try:
    from circuit.spacetime.distance import max_k_sub_for
except Exception:                                            # pragma: no cover
    def max_k_sub_for(n_errors, *, margin_bytes=4096):       # fallback: no cap
        return 1 << 30


# ─────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────


@dataclass
class SweepParams:
    """Knobs for one group's random-sampling sweep (mirrors the YAML/CLI)."""

    target: int = 16                # classical keep + quantum save threshold
    anchor_weight: int = 3
    free_weight: int = 3

    # classical screen (sqetch, per side)
    cl_num_trials: int = 1_000_000
    cl_k_sub: int = 0               # 0 ⇒ ker dim (|G|), capped by GPU ceiling
    cl_batch_size: int = 50_000
    surv_cap: int = 1_000           # survivor pool cap, per side (pairing samples
                                    # ≤500k pairs ⇒ ~700/side suffices; 1k ⇒ 1M combos)
    cl_sample_budget: int = 100_000  # max draws, per side (== cl_barren_skip ⇒ a side
                                     # either fills the pool or stops at 100k)
    cl_barren_skip: int = 100_000   # 0 survivors after this many draws ⇒ SKIP

    # quantum screen (single sqetch, max k_sub)
    q_num_trials: int = 5_000_000
    q_k_sub: int = 0                # 0 ⇒ max possible (max_k_sub_for)
    q_batch_size: int = 200_000
    q_strategy: str = "auto"
    pair_budget: int = 500_000      # max quantum tests (post single-orbit gate)
    pair_fail: int = 100_000        # 0 passers after this many tests ⇒ FAIL
    max_pass: int = 6               # stop the group at this many passers

    devices: Optional[list] = None
    seed: int = 0

    # exhaustion guard for tiny groups (consecutive no-progress draws)
    max_consecutive_misses: int = 50_000


# ─────────────────────────────────────────────────────────────────
# k_sub resolvers
# ─────────────────────────────────────────────────────────────────


def _shmem_k_sub_cap(n_cols: int) -> int:
    """Largest k_sub that fits the GPU shared-memory budget for ``n_cols``
    columns, computed INLINE from sqetch's (possibly patched) shmem ceiling.

    This deliberately does NOT rely on
    ``circuit.spacetime.distance.max_k_sub_for``: that package is optional
    and may be absent, so the import-time fallback there returns ``1<<30``
    — which sqetch then rejects with a shared-memory ``ValueError`` once ``n`` is
    large enough that the full null space no longer fits (n≳1500, i.e. exactly
    the d≥32 regime). The formula mirrors sqetch's own cap
    (``api._estimate_ksub``) with an extra 4 KiB margin, so it is always safely
    below the kernel's hard limit. Raises if CUDA/sqetch is unavailable
    (handled by :func:`_resolve_max_k_sub`)."""
    import math

    import sqetch._arch as _arch                       # patched on workers (227 KiB cc≥9)
    nw = math.ceil(n_cols / 64)
    budget = _arch.max_shmem_per_block_optin() - 4096
    cap = (budget - ((n_cols * 2 + 7) & ~7) - nw * 8 - 8 - 512) // (nw * 8)
    return max(1, int(cap))


def _resolve_max_k_sub(n_cols: int) -> int:
    """Shmem-ceiling k_sub for an ``n_cols``-wide matrix on the current GPU.

    Prefers the inline sqetch computation (correct even when ``circuit``
    is unavailable); falls back to ``max_k_sub_for`` (monkeypatched
    in GPU-free tests) and finally to an effectively-unbounded value."""
    try:
        return _shmem_k_sub_cap(n_cols)
    except Exception:                                    # pragma: no cover - no GPU
        try:
            return int(max_k_sub_for(n_cols))
        except Exception:
            return 1 << 30


def _classical_k_sub(gd: GroupData, params: SweepParams) -> int:
    """Classical k_sub: 0 ⇒ ker dim (|G|), capped by the GPU shmem ceiling."""
    ker_dim = gd.n                                   # full-rank A_bin ⇒ ker = |G|
    want = params.cl_k_sub if params.cl_k_sub > 0 else ker_dim
    return max(1, min(want, ker_dim, _resolve_max_k_sub(2 * gd.n)))


def _quantum_k_sub(gd: GroupData, params: SweepParams) -> int:
    """Quantum k_sub: 0 ⇒ the MAX possible (GPU shmem ceiling for n=5|G|).

    Even an explicit ``q_k_sub`` is clamped to the ceiling so an over-large
    request can never trigger sqetch's shared-memory ``ValueError``."""
    ceil = _resolve_max_k_sub(5 * gd.n)
    want = params.q_k_sub if params.q_k_sub > 0 else ceil
    return max(1, min(want, ceil))


# default estimators (sqetch) — imported lazily so GPU-free tests need no CUDA
def _default_classical_estimator(M_bin, **kw):
    from core.dist.classical import estimate_classical_distance_sqetch
    return estimate_classical_distance_sqetch(M_bin, **kw)


def _default_quantum_estimator(Hx, Hz, **kw):
    from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
    return estimate_quantum_distances_sqetch(Hx, Hz, **kw)


# ─────────────────────────────────────────────────────────────────
# Stage 1: classical screen (random sampling, per side)
# ─────────────────────────────────────────────────────────────────


def sample_classical_side(
    gd: GroupData,
    pool: List[tuple],
    params: SweepParams,
    side: str,
    *,
    rng: np.random.Generator,
    classical_estimator: Optional[Callable] = None,
    log: Optional[Callable] = None,
) -> Tuple[List[dict], str]:
    """Randomly sample ``a0`` (weight-3) × ``a1`` (from ``pool``) and screen
    the classical distance of the side (``"A"`` or ``"B"``).

    Returns ``(survivors, status)`` where ``status`` is ``"ok"`` (≥1 survivor)
    or ``"barren"`` (0 survivors after ``cl_barren_skip`` draws or the sample
    space exhausted). Each survivor: ``{"free", "anchor", "d", "k_sub"}``.
    """
    estimator = classical_estimator or _default_classical_estimator
    bin_for = a_bin_for if side == "A" else b_bin_for
    k_sub = _classical_k_sub(gd, params)
    dev = (params.devices[0] if params.devices else 0)

    survivors: List[dict] = []
    seen: set = set()
    samples = 0
    misses = 0
    pool_arr = list(pool)
    _last = time.time()
    while len(survivors) < params.surv_cap and samples < params.cl_sample_budget:
        if log and time.time() - _last > 20:
            _last = time.time()
            log(f"screen {side}: draws={samples} surv={len(survivors)}")
        a1 = tuple(pool_arr[int(rng.integers(len(pool_arr)))])
        a0 = random_ring_element(gd, params.free_weight, rng=rng)
        if a0 is None:
            misses += 1
            if misses >= params.max_consecutive_misses:
                break
            continue
        a0 = tuple(a0)
        key = (a0, a1)
        if key in seen:
            misses += 1
            if misses >= params.max_consecutive_misses:
                break
            continue
        misses = 0
        seen.add(key)
        samples += 1
        d = estimator(
            bin_for(a0, a1, gd),
            num_trials=params.cl_num_trials, d_target=params.target,
            k_sub=k_sub, batch_size=params.cl_batch_size, devices=[dev],
        )
        if _passes(d, params.target):
            survivors.append({"free": a0, "anchor": a1,
                              "d": (None if d is None else int(d)),
                              "k_sub": k_sub})
        # Early barren skip: nothing after cl_barren_skip draws.
        if samples >= params.cl_barren_skip and not survivors:
            return survivors, "barren"
    status = "ok" if survivors else "barren"
    return survivors, status


# ─────────────────────────────────────────────────────────────────
# Stage 2: pairing (sampled) + single max-k_sub quantum screen + save
# ─────────────────────────────────────────────────────────────────


def pair_sampled(
    gd: GroupData,
    A_surv: List[dict],
    B_surv: List[dict],
    params: SweepParams,
    out_dir: Path,
    *,
    rng: np.random.Generator,
    gap_expr: str,
    tag: str,
    provenance: Optional[dict] = None,
    quantum_estimator: Optional[Callable] = None,
    log: Optional[Callable] = None,
) -> dict:
    """Sample ``(A, B)`` survivor pairs, gate single-orbit, run the single
    max-k_sub quantum sqetch, and save passers.

    Stop conditions: ``max_pass`` passers (success), ``pair_fail`` quantum
    tests with 0 passers (FAIL), or ``pair_budget`` quantum tests (budget).
    The single-orbit gate (``k != |G|``) does **not** count toward the budget.

    Returns a dict with ``verdict`` ∈ {``pass``, ``fail``, ``budget``,
    ``exhausted``}, ``passed`` (code-dir paths), ``n_pairs`` (quantum tests
    run), ``n_gate_skip``, ``best_dx_seen``, ``best_dz_seen``.
    """
    estimator = quantum_estimator or _default_quantum_estimator
    q_k_sub = _quantum_k_sub(gd, params)
    dev = (params.devices[0] if params.devices else 0)
    qf = params.target

    passed: List[str] = []
    seen: set = set()
    n_pairs = 0          # quantum tests actually run (post single-orbit gate)
    n_gate_skip = 0
    best_dx = best_dz = None
    misses = 0
    verdict = "budget"
    _last = time.time()

    while n_pairs < params.pair_budget and len(passed) < params.max_pass:
        if log and time.time() - _last > 20:
            _last = time.time()
            log(f"pair: tests={n_pairs} pass={len(passed)} "
                f"gate_skip={n_gate_skip} best=({best_dx},{best_dz})")
        A = A_surv[int(rng.integers(len(A_surv)))]
        B = B_surv[int(rng.integers(len(B_surv)))]
        key = (A["free"], A["anchor"], B["free"], B["anchor"])
        if key in seen:
            misses += 1
            if misses >= params.max_consecutive_misses:
                verdict = "exhausted"
                break
            continue
        seen.add(key)

        A_ring = [[canonicalize(A["free"]), canonicalize(A["anchor"])]]
        B_ring = [[canonicalize(B["free"]), canonicalize(B["anchor"])]]
        Hx = build_Hx(A_ring, B_ring, gd)
        Hz = build_Hz(A_ring, B_ring, gd)
        Lx, Lz, k = screen_basis(Hx, Hz)             # fast bit-packed basis
        if k != gd.n:                                # single-orbit gate
            n_gate_skip += 1
            misses += 1
            if misses >= params.max_consecutive_misses:
                verdict = "exhausted"
                break
            continue
        misses = 0
        n_pairs += 1

        dx, dz = estimator(
            Hx, Hz, Lx=Lx, Lz=Lz, num_trials=params.q_num_trials,
            d_target=qf, return_logical=False, devices=[dev],
            strategy=params.q_strategy, k_sub=q_k_sub,
            batch_size=params.q_batch_size,
        )
        best_dx = _track_best(best_dx, dx)
        best_dz = _track_best(best_dz, dz)

        if _passes(dx, qf) and _passes(dz, qf):
            code = build_canonical_code(A["free"], A["anchor"],
                                        B["free"], B["anchor"], gd)
            try:
                basis = canonical_logical_basis(code, gd)
            except NotSingleOrbit:
                # k==|G| from screen_basis but the orbit basis is degenerate;
                # treat as a non-pass (rare).
                continue
            runs = [
                _run_record("classical_A", "sqetch",
                            num_trials=params.cl_num_trials, k_sub=A["k_sub"],
                            batch_size=params.cl_batch_size, d=A["d"]),
                _run_record("classical_B", "sqetch",
                            num_trials=params.cl_num_trials, k_sub=B["k_sub"],
                            batch_size=params.cl_batch_size, d=B["d"]),
                _run_record("quantum_screen", "sqetch",
                            num_trials=params.q_num_trials, k_sub=q_k_sub,
                            batch_size=params.q_batch_size, dx=dx, dz=dz),
            ]
            code_dir = save_passed_code(
                out_dir, gd=gd, gap_expr=gap_expr, tag=tag, code=code,
                basis=basis, dx=dx, dz=dz,
                classical_dA=A["d"], classical_dB=B["d"],
                distance_runs=runs, provenance=provenance,
            )
            passed.append(str(code_dir))
            if log:
                log(f"[{tag}] PASSER #{len(passed)}: {code_dir}")

        # FAIL cutoff: nothing after pair_fail quantum tests.
        if n_pairs >= params.pair_fail and not passed:
            verdict = "fail"
            break

    # Any saved passer ⇒ "pass"; otherwise keep the stop-reason
    # ("fail" / "budget" / "exhausted").
    if passed:
        verdict = "pass"

    return {"verdict": verdict, "passed": passed, "n_pairs": n_pairs,
            "n_gate_skip": n_gate_skip,
            "best_dx_seen": best_dx, "best_dz_seen": best_dz}


# ─────────────────────────────────────────────────────────────────
# Top-level per-group sweep
# ─────────────────────────────────────────────────────────────────


def run_group_sweep(
    group: GroupSpec,
    params: SweepParams,
    out_dir,
    *,
    gd=None,
    provenance: Optional[dict] = None,
    classical_estimator: Optional[Callable] = None,
    quantum_estimator: Optional[Callable] = None,
    log: Optional[Callable] = None,
) -> dict:
    """Run the random-sampling funnel for one group; return a summary dict.

    ``gd`` may be a real :class:`core.group.GroupData` (needs GAP) or any
    GAP-free duck-typed equivalent. If ``None`` a ``GroupData`` is
    built from ``group.gap_expr`` (requires GAP).

    The ``verdict`` field is the group's outcome:
        ``no_units``         — no weight-3 identity unit (empty anchor pool).
        ``skip``             — a classical side came back barren.
        ``fail``             — ``pair_fail`` quantum tests, 0 passers.
        ``pass``             — ≥1 passer saved (``max_pass`` ⇒ early stop).
        ``budget``/``exhausted`` — pairing budget / sample space exhausted
                                   without reaching ``max_pass``.
    """
    if gd is None:
        gd = GroupData(group.gap_expr)
    out_dir = Path(out_dir)
    rng = np.random.default_rng(params.seed)
    glog = (lambda m: log(f"[{group.tag} |G|={gd.n}] {m}")) if log else None

    pool = build_full_rank_block_pool_brute(gd, params.anchor_weight,
                                            force_identity=True)
    base = {"group": group.tag, "order": gd.n, "target": params.target,
            "pool_size": len(pool)}
    if not pool:
        summary = {**base, "verdict": "no_units", "passed": [], "n_pairs": 0,
                   "n_A_survivors": 0, "n_B_survivors": 0,
                   "best_dx_seen": None, "best_dz_seen": None,
                   "note": f"no weight-{params.anchor_weight} identity units"}
        _save_stats(out_dir, summary)
        return summary

    A_surv, a_status = sample_classical_side(
        gd, pool, params, "A", rng=rng,
        classical_estimator=classical_estimator, log=glog)
    B_surv, b_status = sample_classical_side(
        gd, pool, params, "B", rng=rng,
        classical_estimator=classical_estimator, log=glog)

    if a_status == "barren" or b_status == "barren":
        summary = {**base, "verdict": "skip",
                   "n_A_survivors": len(A_surv), "n_B_survivors": len(B_surv),
                   "passed": [], "n_pairs": 0,
                   "best_dx_seen": None, "best_dz_seen": None,
                   "note": f"classical barren (A={a_status}, B={b_status})"}
        _save_stats(out_dir, summary)
        return summary

    res = pair_sampled(
        gd, A_surv, B_surv, params, out_dir, rng=rng,
        gap_expr=group.gap_expr, tag=group.tag, provenance=provenance,
        quantum_estimator=quantum_estimator, log=glog)
    summary = {**base,
               "n_A_survivors": len(A_surv), "n_B_survivors": len(B_surv),
               "A_d_classical": sorted({s["d"] for s in A_surv},
                                       key=lambda d: (d is None, 0 if d is None else d)),
               "B_d_classical": sorted({s["d"] for s in B_surv},
                                       key=lambda d: (d is None, 0 if d is None else d)),
               **res}
    _save_stats(out_dir, summary)
    return summary


def _sampled_survivor_stream(
    gd: GroupData,
    pool: List[tuple],
    params: SweepParams,
    side: str,
    *,
    rng: np.random.Generator,
    classical_estimator: Optional[Callable] = None,
    log: Optional[Callable] = None,
):
    """Generator: random ``(a0,a1)`` draws → classical screen → **yield survivors
    as found** (the streamed twin of :func:`sample_classical_side`).

    Stops (``StopIteration``) at ``surv_cap`` yields, ``cl_sample_budget`` draws,
    the consecutive-miss guard, or — for a barren side — the moment
    ``cl_barren_skip`` draws have produced zero survivors. Yielding as found lets
    the streaming driver start pairing on the first survivor of each side instead
    of waiting for the whole pool to fill."""
    estimator = classical_estimator or _default_classical_estimator
    bin_for = a_bin_for if side == "A" else b_bin_for
    k_sub = _classical_k_sub(gd, params)
    dev = (params.devices[0] if params.devices else 0)
    pool_arr = list(pool)
    seen: set = set()
    draws = 0
    yielded = 0
    misses = 0
    _last = time.time()
    while yielded < params.surv_cap and draws < params.cl_sample_budget:
        if log and time.time() - _last > 20:
            _last = time.time()
            log(f"stream {side}: draws={draws} surv={yielded}")
        a1 = tuple(pool_arr[int(rng.integers(len(pool_arr)))])
        a0 = random_ring_element(gd, params.free_weight, rng=rng)
        if a0 is None:
            misses += 1
            if misses >= params.max_consecutive_misses:
                break
            continue
        a0 = tuple(a0)
        key = (a0, a1)
        if key in seen:
            misses += 1
            if misses >= params.max_consecutive_misses:
                break
            continue
        misses = 0
        seen.add(key)
        draws += 1
        d = estimator(
            bin_for(a0, a1, gd),
            num_trials=params.cl_num_trials, d_target=params.target,
            k_sub=k_sub, batch_size=params.cl_batch_size, devices=[dev],
        )
        if _passes(d, params.target):
            yielded += 1
            yield {"free": a0, "anchor": a1,
                   "d": (None if d is None else int(d)), "k_sub": k_sub}
        # Barren: nothing after cl_barren_skip draws ⇒ stop the side.
        if draws >= params.cl_barren_skip and yielded == 0:
            return


def run_group_sweep_streaming(
    group: GroupSpec,
    params: SweepParams,
    out_dir,
    *,
    gd=None,
    provenance: Optional[dict] = None,
    classical_estimator: Optional[Callable] = None,
    quantum_estimator: Optional[Callable] = None,
    log: Optional[Callable] = None,
) -> dict:
    """Streaming twin of :func:`run_group_sweep`: **pair as survivors are found**.

    Interleaves the two classical streams — each new A survivor pairs against all
    accumulated B (and vice-versa), every (A,B) once — so the first quantum test
    fires within minutes of the first survivor on each side instead of after both
    classical pools fully fill. Same single-orbit gate, single max-k_sub quantum
    sqetch, save-on-pass, and stop rules (``max_pass`` / ``pair_fail`` /
    ``pair_budget``) as the batch funnel. As a bonus, a barren side short-circuits
    the *other* side's screen (no pairs are possible), so skips cost one side, not
    two. Identical verdict vocabulary to :func:`run_group_sweep`."""
    if gd is None:
        gd = GroupData(group.gap_expr)
    out_dir = Path(out_dir)
    rng = np.random.default_rng(params.seed)
    glog = (lambda m: log(f"[{group.tag} |G|={gd.n}] {m}")) if log else None
    qest = quantum_estimator or _default_quantum_estimator

    pool = build_full_rank_block_pool_brute(gd, params.anchor_weight,
                                            force_identity=True)
    base = {"group": group.tag, "order": gd.n, "target": params.target,
            "pool_size": len(pool)}
    if not pool:
        summary = {**base, "verdict": "no_units", "passed": [], "n_pairs": 0,
                   "n_A_survivors": 0, "n_B_survivors": 0,
                   "best_dx_seen": None, "best_dz_seen": None,
                   "note": f"no weight-{params.anchor_weight} identity units"}
        _save_stats(out_dir, summary)
        return summary

    q_k_sub = _quantum_k_sub(gd, params)
    dev = (params.devices[0] if params.devices else 0)
    qf = params.target
    A_it = _sampled_survivor_stream(gd, pool, params, "A", rng=rng,
                                    classical_estimator=classical_estimator,
                                    log=glog)
    B_it = _sampled_survivor_stream(gd, pool, params, "B", rng=rng,
                                    classical_estimator=classical_estimator,
                                    log=glog)
    A_surv: List[dict] = []
    B_surv: List[dict] = []
    passed: List[str] = []
    n_pairs = 0
    n_gate_skip = 0
    best_dx = best_dz = None
    seen_pairs: set = set()
    a_done = b_done = False

    def _pair_new(new_entry: dict, side: str, others: List[dict]) -> None:
        nonlocal n_pairs, n_gate_skip, best_dx, best_dz
        for o in others:
            if len(passed) >= params.max_pass or n_pairs >= params.pair_budget:
                return
            A, B = (new_entry, o) if side == "A" else (o, new_entry)
            key = (A["free"], A["anchor"], B["free"], B["anchor"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            A_ring = [[canonicalize(A["free"]), canonicalize(A["anchor"])]]
            B_ring = [[canonicalize(B["free"]), canonicalize(B["anchor"])]]
            Hx = build_Hx(A_ring, B_ring, gd)
            Hz = build_Hz(A_ring, B_ring, gd)
            Lx, Lz, k = screen_basis(Hx, Hz)
            if k != gd.n:                            # single-orbit gate
                n_gate_skip += 1
                continue
            n_pairs += 1
            dx, dz = qest(
                Hx, Hz, Lx=Lx, Lz=Lz, num_trials=params.q_num_trials,
                d_target=qf, return_logical=False, devices=[dev],
                strategy=params.q_strategy, k_sub=q_k_sub,
                batch_size=params.q_batch_size,
            )
            best_dx = _track_best(best_dx, dx)
            best_dz = _track_best(best_dz, dz)
            if _passes(dx, qf) and _passes(dz, qf):
                code = build_canonical_code(A["free"], A["anchor"],
                                            B["free"], B["anchor"], gd)
                try:
                    basis = canonical_logical_basis(code, gd)
                except NotSingleOrbit:
                    continue
                runs = [
                    _run_record("classical_A", "sqetch",
                                num_trials=params.cl_num_trials, k_sub=A["k_sub"],
                                batch_size=params.cl_batch_size, d=A["d"]),
                    _run_record("classical_B", "sqetch",
                                num_trials=params.cl_num_trials, k_sub=B["k_sub"],
                                batch_size=params.cl_batch_size, d=B["d"]),
                    _run_record("quantum_screen", "sqetch",
                                num_trials=params.q_num_trials, k_sub=q_k_sub,
                                batch_size=params.q_batch_size, dx=dx, dz=dz),
                ]
                code_dir = save_passed_code(
                    out_dir, gd=gd, gap_expr=group.gap_expr, tag=group.tag,
                    code=code, basis=basis, dx=dx, dz=dz,
                    classical_dA=A["d"], classical_dB=B["d"],
                    distance_runs=runs, provenance=provenance,
                )
                passed.append(str(code_dir))
                if glog:
                    glog(f"PASSER #{len(passed)}: {code_dir}")

    while (not (a_done and b_done) and len(passed) < params.max_pass
           and n_pairs < params.pair_budget):
        progressed = False
        if not a_done and len(A_surv) < params.surv_cap:
            try:
                a = next(A_it)
                progressed = True
                _pair_new(a, "A", B_surv)
                A_surv.append(a)
            except StopIteration:
                a_done = True
        if a_done and not A_surv:                    # A barren ⇒ no pairs possible
            break
        if len(passed) >= params.max_pass or n_pairs >= params.pair_budget:
            break
        if not b_done and len(B_surv) < params.surv_cap:
            try:
                b = next(B_it)
                progressed = True
                _pair_new(b, "B", A_surv)
                B_surv.append(b)
            except StopIteration:
                b_done = True
        if b_done and not B_surv:                    # B barren ⇒ no pairs possible
            break
        if n_pairs >= params.pair_fail and not passed:   # FAIL cutoff
            break
        if not progressed:
            break

    if passed:
        verdict = "pass"
    elif not A_surv or not B_surv:
        verdict = "skip"
    elif n_pairs >= params.pair_fail:
        verdict = "fail"
    elif n_pairs >= params.pair_budget:
        verdict = "budget"
    else:
        verdict = "exhausted"

    summary = {**base, "verdict": verdict, "passed": passed, "n_pairs": n_pairs,
               "n_gate_skip": n_gate_skip,
               "n_A_survivors": len(A_surv), "n_B_survivors": len(B_surv),
               "A_d_classical": sorted({s["d"] for s in A_surv},
                                       key=lambda d: (d is None, 0 if d is None else d)),
               "B_d_classical": sorted({s["d"] for s in B_surv},
                                       key=lambda d: (d is None, 0 if d is None else d)),
               "best_dx_seen": best_dx, "best_dz_seen": best_dz}
    _save_stats(out_dir, summary)
    return summary


def _save_stats(out_dir: Path, summary: dict) -> None:
    """Persist a per-group stats JSON (passers trimmed to dir names)."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = dict(summary)
    s["n_passed"] = len(s.get("passed", []))
    (out_dir / f"_sweep_{summary['group']}.json").write_text(json.dumps(s, indent=2))
