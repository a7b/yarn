"""Pairing (quantum) search phase: build Hx/Hz from saved classical (A, B) pairs.

Loads the saved classical pool from
``{results_dir}/{group_tag}/{ma}x{na}/classical_A/`` (and ``classical_B/``
for non-abelian). For abelian, each saved A is paired with itself, and the
second base matrix is built as ``B = A*`` (elementwise conjugate) in
:func:`_evaluate_pair` — NOT ``B = A`` (which caps distance; see there). For
each candidate ``(meta_A, meta_B)``:

1. Skip if already tried (``tried_pairs.json``).
2. Apply :func:`apply_quantum_pairing_filters` cheapest-first
   (group/shape/distance/girth/check-weight caps).
3. Build ``Hx``, ``Hz`` from the recovered ring matrices via
   :func:`build_quantum_code`.
4. Estimate ``(dx, dz)`` with BP+OSD; PASS iff each is ``None`` or
   ``>= d_target`` (the search-pipeline convention).
5. Optional heavier verification via SQetch (GPU random-ISD, QDistRnd-style) when
   ``cfg.pairing.sqetch_verify.enabled``.
6. Save a quantum JSON to ``quantum/`` and append to manifest.

Pair caps:
- ``pair_mode = "full_pool"``: iterate over every (A, B) cross-product.
  ``"new_only"`` is currently not implemented (raises).
- ``max_pairs``: hard cap on pair attempts.
- ``min_quantum_pool_size``: stop after this many passing quantum codes.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.classical_code import (
    build_A_bin,
    build_B_bin,
    weight_matrix,
)
from core.dist.quantum_bposd import estimate_quantum_distances_bposd
from core.dist.quantum_sqetch import estimate_quantum_distances_sqetch
from core.group import GroupData
from core.quantum_code import build_quantum_code, compute_k, quantum_check_weights
from search.configs.config import SearchConfig
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    manifest_path,
    quantum_dir,
    quantum_filename,
    tried_pairs_path,
    weight_tag,
)
from search.configs.provenance import build_provenance
from search.filters.classical._shared.girth_tanner import girth_tanner
from search.filters.config import (
    QuantumPairingFilterConfig,
    apply_quantum_pairing_filters,
)


def run_pairing(cfg: SearchConfig) -> dict:
    """Run the pairing (quantum) stage.

    Args:
        cfg: parsed :class:`SearchConfig`. ``cfg.pairing`` must be set.

    Returns:
        Summary ``{"new_quantum": [paths], "n_pairs_tried": int,
        "n_pairs_passed": int}``.

    Raises:
        ValueError: ``cfg.pairing`` is None, or unsupported
        ``pair_mode``.
    """
    if cfg.pairing is None:
        raise ValueError("run_pairing needs cfg.pairing to be configured.")
    if cfg.pairing.pool.pair_mode != "full_pool":
        raise ValueError(
            f"pair_mode={cfg.pairing.pool.pair_mode!r} not yet supported "
            f"(only 'full_pool' for now)."
        )

    gd = GroupData(cfg.group.gap_expr)
    is_abelian = gd.is_abelian

    pool_A = _load_classical_pool(classical_A_dir(cfg))
    if is_abelian:
        pool_B = pool_A   # same pool; B's ring matrix = A* (conjugated in _evaluate_pair)
    else:
        pool_B = _load_classical_pool(classical_B_dir(cfg))

    out_dir = quantum_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    tried = _load_tried_pairs(cfg)

    filter_cfg = _build_pairing_filter_cfg(cfg.pairing.filters)

    saved_paths: list = []
    n_tried = 0
    n_passed = 0
    max_pairs = cfg.pairing.pool.max_pairs
    min_pool = cfg.pairing.pool.min_quantum_pool_size
    # min_quantum_pool_size is a cap on the POOL, so codes saved by previous
    # runs count toward it (rerunning a satisfied config is a no-op).
    n_existing = _count_existing_quantum(out_dir)

    for a_meta in pool_A:
        for b_meta in (pool_B if not is_abelian else [a_meta]):
            if (min_pool > 0 and n_existing + len(saved_paths) >= min_pool):
                break
            if max_pairs is not None and n_tried >= max_pairs:
                break

            pair_key = _pair_key(a_meta, b_meta)
            if pair_key in tried:
                continue
            tried.add(pair_key)
            n_tried += 1

            if not apply_quantum_pairing_filters(a_meta, b_meta, filter_cfg, gd=gd):
                continue

            ok, qcode, dist_info = _evaluate_pair(
                a_meta, b_meta, gd, cfg,
            )
            if not ok:
                continue

            n_passed += 1
            path = _save_quantum(
                qcode, dist_info, cfg, out_dir,
                source_A_meta=a_meta,
                source_B_meta=b_meta,
            )
            saved_paths.append(str(path))
            if cfg.verbose:
                print(f"[pairing] saved {Path(path).name} "
                      f"(dx={dist_info['dx']}, dz={dist_info['dz']})")

        if (min_pool > 0 and n_existing + len(saved_paths) >= min_pool):
            break
        if max_pairs is not None and n_tried >= max_pairs:
            break

    _save_tried_pairs(cfg, tried)
    if saved_paths:
        _append_manifest_entry(cfg, saved_paths)
    return {
        "new_quantum": saved_paths,
        "n_pairs_tried": n_tried,
        "n_pairs_passed": n_passed,
    }


# ─────────────────────────────────────────────────────────────────
# Classical-pool loader
# ─────────────────────────────────────────────────────────────────


def _load_classical_pool(root: Path) -> list:
    """Discover all ``d*.json`` files under ``root`` (recursive) and load them.

    Returns a list of meta dicts; each dict carries the loaded JSON
    contents plus ``"path"`` (absolute string) and ``"weight_matrix"``
    (numpy ndarray for downstream filter ergonomics).
    """
    if not root.exists():
        return []
    pool: list = []
    for path in sorted(root.rglob("d*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["path"] = str(path)
        if "weight_matrix" in data:
            data["weight_matrix"] = np.array(data["weight_matrix"], dtype=int)
        # Treat missing dist/girth keys as None so pairing filters can be
        # configured to skip codes that don't carry the relevant field.
        data.setdefault("dist", None)
        data.setdefault("girth_tanner", None)
        # Pairing filter expects "girth" key.
        data.setdefault("girth", data.get("girth_tanner"))
        pool.append(data)
    return pool


def _count_existing_quantum(out_dir: Path) -> int:
    """Quantum codes already saved by previous runs (``k*_*.json``)."""
    if not out_dir.exists():
        return 0
    return sum(1 for _ in out_dir.glob("k*_*.json"))


def _pair_key(a_meta: dict, b_meta: dict) -> tuple:
    return (a_meta.get("path"), b_meta.get("path"))


def _load_tried_pairs(cfg: SearchConfig) -> set:
    path = tried_pairs_path(cfg)
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    return {tuple(item) for item in raw}


def _save_tried_pairs(cfg: SearchConfig, tried: set) -> None:
    path = tried_pairs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(t) for t in sorted(tried)], indent=2))


# ─────────────────────────────────────────────────────────────────
# Filter config builder
# ─────────────────────────────────────────────────────────────────


def _build_pairing_filter_cfg(f) -> QuantumPairingFilterConfig:
    return QuantumPairingFilterConfig(
        require_same_group=f.require_same_group,
        require_same_shape=f.require_same_shape,
        min_classical_distance=f.min_classical_distance,
        min_classical_girth=f.min_classical_girth,
        max_Hx_check_weight=f.max_Hx_check_weight,
        max_Hz_check_weight=f.max_Hz_check_weight,
        min_full_extractor_bridge_d=f.min_full_extractor_bridge_d,
    )


# ─────────────────────────────────────────────────────────────────
# Per-pair evaluation
# ─────────────────────────────────────────────────────────────────


def _evaluate_pair(a_meta: dict, b_meta: dict, gd: GroupData,
                   cfg: SearchConfig) -> tuple:
    """Build the quantum code for a pair, estimate distances, decide PASS.

    Returns ``(ok, qcode, dist_info)`` where:
      - ``ok``: True iff the pair passes BPOSD and (if enabled) sqetch.
      - ``qcode``: dict from :func:`build_quantum_code` — has Hx, Hz,
        canonical A / B, perms, etc. ``None`` if not built.
      - ``dist_info``: dict with ``k``, ``dx``, ``dz``, ``estimator``,
        ``bposd_num_trials``, ``sqetch_num_trials``, ``girth_hx``,
        ``girth_hz``. ``None`` if not built.
    """
    A = _ring_matrix_from_meta(a_meta)
    B = _ring_matrix_from_meta(b_meta)
    if gd.is_abelian:
        # Abelian LP convention: B = A* (elementwise conjugate), NOT B = A.
        # B = A is unbalanced and caps the distance at min(na,nb)+min(ma,mb)
        # via A-independent diagonal logicals; B = A* restores the proper
        # hypergraph-product form. (a_meta == b_meta for abelian, so this
        # conjugates A.) dagger is rep-independent (uses gd.inv).
        from core.group import dagger
        B = [[tuple(sorted(dagger(e, gd))) for e in row] for row in A]
    qcode = build_quantum_code(A, B, gd)
    Hx, Hz = qcode["Hx"], qcode["Hz"]

    k = compute_k(Hx, Hz)
    if k == 0:
        # No logical qubits => no distance to estimate; the estimators
        # would return (None, None), which the PASS convention reads as
        # an optimistic pass — a k = 0 code must never be saved.
        return False, None, None

    d_target_bposd = cfg.pairing.bposd.d_target
    dx, dz = estimate_quantum_distances_bposd(
        Hx, Hz,
        num_trials=cfg.pairing.bposd.num_trials,
        n_workers=cfg.pairing.bposd.n_workers,
        d_target=d_target_bposd,
        osd_order=cfg.pairing.bposd.osd_order,
    )
    if not _pass(dx, dz, d_target_bposd):
        return False, None, None

    estimator = "bposd"
    sqetch_trials = 0
    sqetch_k_sub = None
    sqetch_strategy = None
    sqetch_batch_size = None
    if cfg.pairing.sqetch_verify.enabled:
        sv = cfg.pairing.sqetch_verify
        d_target_sq = sv.d_target if sv.d_target is not None else d_target_bposd
        dx_sq, dz_sq = estimate_quantum_distances_sqetch(
            Hx, Hz,
            num_trials=sv.num_trials,
            d_target=d_target_sq,
            return_logical=False,
            devices=sv.devices,
            strategy=sv.strategy,
            k_sub=sv.k_sub,
            batch_size=sv.batch_size,
        )
        if not _pass(dx_sq, dz_sq, d_target_sq):
            return False, None, None
        dx, dz = dx_sq, dz_sq
        estimator = "bposd+sqetch"
        sqetch_trials = sv.num_trials
        # A sqetch distance record is uninterpretable without its k_sub
        # (and strategy/batch) — always persist them alongside num_trials.
        sqetch_k_sub = sv.k_sub
        sqetch_strategy = sv.strategy
        sqetch_batch_size = sv.batch_size

    ghx = girth_tanner(Hx)
    ghz = girth_tanner(Hz)
    dist_info = {
        "k": k,
        "dx": dx,
        "dz": dz,
        "estimator": estimator,
        "bposd_num_trials": cfg.pairing.bposd.num_trials,
        "sqetch_num_trials": sqetch_trials,
        "sqetch_k_sub": sqetch_k_sub,
        "sqetch_strategy": sqetch_strategy,
        "sqetch_batch_size": sqetch_batch_size,
        "girth_hx": ghx,
        "girth_hz": ghz,
    }
    return True, qcode, dist_info


def _pass(dx, dz, d_target) -> bool:
    """PASS iff (each is None) OR (each >= d_target)."""
    def ok(d):
        return d is None or d >= d_target
    return ok(dx) and ok(dz)


def _ring_matrix_from_meta(meta: dict) -> list:
    """Reconstruct a ring matrix (list of lists of tuples) from saved JSON."""
    M = meta["matrix"]
    return [[tuple(int(g) for g in entry) for entry in row] for row in M]


# ─────────────────────────────────────────────────────────────────
# Save quantum + manifest
# ─────────────────────────────────────────────────────────────────


def _save_quantum(qcode: dict, dist_info: dict,
                  cfg: SearchConfig, out_dir: Path,
                  *,
                  source_A_meta: dict,
                  source_B_meta: dict) -> Path:
    A = qcode["A_canonical"]
    B = qcode["B_canonical"]
    W_A = weight_matrix(A)
    W_B = weight_matrix(B)
    wA = weight_tag(W_A)
    wB = weight_tag(W_B)
    ts = int(time.time() * 1_000_000)
    filename = quantum_filename(
        k=dist_info["k"], dx=dist_info["dx"], dz=dist_info["dz"],
        wA_tag=wA, wB_tag=wB,
        bposd_trials=dist_info["bposd_num_trials"],
        sqetch_trials=dist_info.get("sqetch_num_trials", 0),
        timestamp=ts,
    )

    Hx = qcode["Hx"]
    Hz = qcode["Hz"]
    n_phys = int(Hx.shape[1])
    n_x_checks = int(Hx.shape[0])
    n_z_checks = int(Hz.shape[0])
    cw = quantum_check_weights(W_A, W_B)
    ma = len(A)
    na = len(A[0])
    mb = len(B)
    nb = len(B[0])

    data = {
        "gap_expr": cfg.group.gap_expr,
        "group_tag": cfg.group.tag,
        "shape": list(cfg.shape),
        "ma": ma, "na": na, "mb": mb, "nb": nb,
        "n_phys": n_phys,
        "n_x_checks": n_x_checks,
        "n_z_checks": n_z_checks,
        "k": dist_info["k"],
        "dx": dist_info["dx"],
        "dz": dist_info["dz"],
        "estimator": dist_info["estimator"],
        "bposd_num_trials": dist_info["bposd_num_trials"],
        "sqetch_num_trials": dist_info["sqetch_num_trials"],
        "sqetch_k_sub": dist_info.get("sqetch_k_sub"),
        "sqetch_strategy": dist_info.get("sqetch_strategy"),
        "sqetch_batch_size": dist_info.get("sqetch_batch_size"),
        "girth_hx": dist_info["girth_hx"],
        "girth_hz": dist_info["girth_hz"],
        "weight_A": W_A.tolist(),
        "weight_B": W_B.tolist(),
        "Hx_check_weight": cw["Hx_check_weight"],
        "Hz_check_weight": cw["Hz_check_weight"],
        "A": _ring_matrix_to_json(A),
        "B": _ring_matrix_to_json(B),
        "perm_a": qcode["perm_a"],
        "perm_b": qcode["perm_b"],
        "has_full_rank_a": qcode["has_full_rank_a"],
        "has_full_rank_b": qcode["has_full_rank_b"],
        "source_A": _classical_summary(source_A_meta),
        "source_B": _classical_summary(source_B_meta),
        "provenance": build_provenance(
            cfg, phase="pairing",
            module="search.phases.pairing",
            function="run_pairing",
        ),
        "timestamp": ts,
    }
    path = out_dir / filename
    path.write_text(json.dumps(data, indent=2))
    return path


def _classical_summary(meta: dict) -> dict:
    """Compact summary of a source classical JSON for embedding.

    Includes the file path so callers can ``json.load`` the full record
    if needed, plus the few fields useful for downstream filtering /
    diagnostics: dist, weight_matrix, structural flags, girth, canonical
    status.
    """
    if not meta:
        return {}
    # `weight_matrix` may have been converted to ndarray on load — back to list.
    wm = meta.get("weight_matrix")
    if hasattr(wm, "tolist"):
        wm = wm.tolist()
    return {
        "path": meta.get("path"),
        "side": meta.get("side"),
        "dist": meta.get("dist"),
        "weight_matrix": wm,
        "girth_tanner": meta.get("girth_tanner"),
        "f2_rank_M_bin": meta.get("f2_rank_M_bin"),
        "column_space_coverage": meta.get("column_space_coverage"),
        "any_block_col_full_rank": meta.get("any_block_col_full_rank"),
        "has_block_col_containment": meta.get("has_block_col_containment"),
        "is_canonical": meta.get("is_canonical"),
    }


def _ring_matrix_to_json(M) -> list:
    return [[list(entry) for entry in row] for row in M]


def _append_manifest_entry(cfg: SearchConfig, new_quantum: list) -> None:
    """Append a quantum-stage record to manifest.json (creates if missing)."""
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append({
        "timestamp": int(time.time()),
        "new_quantum": new_quantum,
    })
    path.write_text(json.dumps(existing, indent=2))
