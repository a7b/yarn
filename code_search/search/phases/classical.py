"""Classical search phase: sample ring matrices, filter, estimate distance, save.

Chunk 1 supports the **non-abelian** path only. Abelian uses a two-stage
weight-pattern + ring-placement trick that lands in Chunk 2 (needs a
weight-pattern enumerator).

Entry point::

    run_classical(cfg: SearchConfig) -> dict

Returns a small summary dict ``{"new_A": [paths], "new_B": [paths]}``.
Side effect: writes one JSON per accepted code under
``{results_dir}/{group_tag}/{ma}x{na}/classical_A/w{wtag}/...`` (resp.
``classical_B/``), and appends a manifest entry to ``manifest.json``.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.classical_code import (
    build_A_bin,
    build_B_bin,
    canonical_form_A,
    canonical_form_B,
    weight_matrix,
)
from core.dist.classical import estimate_classical_distance
from core.f2 import f2_rank
from core.group import GroupData
from search.configs.config import SearchConfig
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    classical_filename,
    group_dir,
    manifest_path,
    weight_tag,
)
from search.configs.provenance import build_provenance
from search.filters.classical._shared.any_block_col_full_rank import (
    any_block_col_full_rank,
)
from search.filters.config import (
    ClassicalFilterConfig,
    apply_classical_filters,
)
from search.filters.classical._shared.girth_tanner import girth_tanner
from search.sampling._shared.random_ring_matrix import random_ring_matrix
from search.sampling._shared.weight_matrix import random_weight_patterns


def run_classical(cfg: SearchConfig) -> dict:
    """Run the classical sampling stage.

    Args:
        cfg: parsed :class:`SearchConfig`.

    Returns:
        Summary dict ``{"new_A": list[str], "new_B": list[str]}`` — paths
        of newly-saved JSON files. For abelian searches ``new_B`` is empty —
        only ``classical_A/`` is written, because the abelian second matrix is
        ``B = A*`` (elementwise conjugate, built in the pairing stage) and
        ``A*`` has the SAME classical distance/girth as ``A`` (dagger is a qubit
        relabeling), so no separate B-pool is needed.

    Raises:
        ValueError: missing required fields for the resolved branch
            (e.g. abelian search without ``weight_pattern``).
    """
    gd = GroupData(cfg.group.gap_expr)
    if gd.is_abelian:
        return _run_abelian(cfg, gd)
    return _run_non_abelian(cfg, gd)


def _run_non_abelian(cfg: SearchConfig, gd: GroupData) -> dict:
    if cfg.classical.weight_A is None or cfg.classical.weight_B is None:
        raise ValueError(
            "Non-abelian classical phase requires both weight_A and weight_B."
        )

    rng = np.random.default_rng(cfg.classical.sampling.seed)
    filter_cfg = _build_filter_cfg(cfg.classical.filters)

    new_A_paths: list = []
    new_B_paths: list = []

    for side, weight_list in (("A", cfg.classical.weight_A),
                              ("B", cfg.classical.weight_B)):
        W = np.array(weight_list, dtype=int)
        wtag = weight_tag(W)
        out_dir = (classical_A_dir(cfg) if side == "A"
                   else classical_B_dir(cfg)) / f"w{wtag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = _run_one_side(
            side=side, W=W, wtag=wtag, gd=gd, cfg=cfg,
            filter_cfg=filter_cfg, rng=rng, out_dir=out_dir,
            total_samples=cfg.classical.sampling.total_samples,
        )
        if side == "A":
            new_A_paths.extend(saved_paths)
        else:
            new_B_paths.extend(saved_paths)

    _update_manifest(cfg, new_A_paths, new_B_paths)
    return {"new_A": new_A_paths, "new_B": new_B_paths}


def _run_abelian(cfg: SearchConfig, gd: GroupData) -> dict:
    """Two-stage abelian classical search.

    Stage 1: enumerate integer weight patterns ``W`` from
    ``[0, entry_max]^(ma·na)`` and apply weight-only filters.
    Stage 2: for each surviving pattern, sample ``ring_samples_per_weight``
    ring matrices, apply ring-level filters, estimate classical distance,
    save.

    Only ``classical_A/`` is written; the abelian second matrix is ``B = A*``
    (built in pairing) and shares A's classical distance, so ``new_B`` is
    always empty in the return.
    """
    if cfg.classical.weight_pattern is None:
        raise ValueError(
            "Abelian classical phase requires `weight_pattern` config."
        )

    wp = cfg.classical.weight_pattern
    rng = np.random.default_rng(cfg.classical.sampling.seed)
    filter_cfg = _build_filter_cfg(cfg.classical.filters)

    new_A_paths: list = []

    patterns = random_weight_patterns(
        cfg.shape, wp.entry_max, wp.num_weight_samples,
        rng=rng,
        entry_min=wp.entry_min,
        max_row_weight=wp.max_row_weight,
        max_col_weight=wp.max_col_weight,
        min_base_girth_bound=wp.min_base_girth_bound,
        min_weight_distance_bound=wp.min_weight_distance_bound,
        gd=gd,
    )

    for W in patterns:
        wtag = weight_tag(W)
        out_dir = classical_A_dir(cfg) / f"w{wtag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = _run_one_side(
            side="A", W=W, wtag=wtag, gd=gd, cfg=cfg,
            filter_cfg=filter_cfg, rng=rng, out_dir=out_dir,
            total_samples=wp.ring_samples_per_weight,
        )
        new_A_paths.extend(saved_paths)

    _update_manifest(cfg, new_A_paths, [])
    return {"new_A": new_A_paths, "new_B": []}


def _run_one_side(*, side: str, W: np.ndarray, wtag: str,
                  gd: GroupData, cfg: SearchConfig,
                  filter_cfg: ClassicalFilterConfig,
                  rng: np.random.Generator, out_dir: Path,
                  total_samples: int) -> list:
    """Run sampling/filtering/distance for ONE side (A or B).

    Returns the list of new file paths (str) saved this call.
    """
    pool = cfg.classical.pool
    d_target = cfg.classical.distance.d_target

    # Pool reuse / skip
    existing_count = _count_existing(out_dir, d_target)
    if (not pool.force_new
            and pool.min_pool_size > 0
            and existing_count >= pool.min_pool_size):
        return []

    # Content dedup: never save a matrix that is already in the pool (from a
    # previous run or earlier this run) — rerunning a config CONTINUES the
    # search instead of duplicating it. Keyed on the canonical JSON form.
    seen_matrices: set = set()
    if out_dir.exists():
        for p in out_dir.glob("d*.json"):
            try:
                meta = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if "matrix" in meta:
                seen_matrices.add(repr(meta["matrix"]))

    saved_paths: list = []
    saved_count = existing_count
    max_saved = pool.max_saved

    for _ in range(total_samples):
        if max_saved > 0 and saved_count >= max_saved:
            break

        M = random_ring_matrix(
            gd, W, rng=rng,
            include_identity=cfg.classical.sampling.include_identity,
            min_element_order=cfg.classical.sampling.min_element_order,
            avoid_same_coset=cfg.classical.sampling.avoid_same_coset,
            max_tries=cfg.classical.sampling.max_tries,
            canonicalize=cfg.classical.sampling.canonicalize,
        )
        if M is None:
            continue

        matrix_key = repr(_ring_matrix_to_json(M))
        if matrix_key in seen_matrices:
            continue

        if side == "A":
            M_bin = build_A_bin(M, gd)
        else:
            M_bin = build_B_bin(M, gd)

        try:
            passes = apply_classical_filters(M, M_bin, gd, filter_cfg)
        except ValueError as e:
            # Re-raise configuration errors verbatim (e.g. ma>1 with ma=1-only filter).
            raise

        if not passes:
            continue

        # Tanner girth on the binary lift (optional informational field).
        g = girth_tanner(M_bin)

        dist = estimate_classical_distance(
            M_bin,
            num_trials=cfg.classical.distance.num_trials,
            n_workers=cfg.classical.distance.n_workers,
            d_target=d_target,
            osd_order=cfg.classical.distance.osd_order,
        )
        # PASS semantics: dist is None (no codeword found) OR dist >= d_target.
        if dist is not None and dist < d_target:
            continue
        # If dist is None, we don't know the real distance — record the
        # `d_target` value as the conservative lower bound? We record None
        # in JSON to be honest. The search uses d_target downstream.
        dist_for_filename = dist if dist is not None else d_target

        path = _save_classical(
            M=M, M_bin=M_bin, dist=dist, girth=g,
            wtag=wtag, side=side, gd=gd, cfg=cfg, out_dir=out_dir,
            dist_filename=dist_for_filename,
        )
        seen_matrices.add(matrix_key)
        saved_paths.append(str(path))
        saved_count += 1
        if cfg.verbose:
            print(f"[classical:{side}] saved {path.name} (d={dist})")

    return saved_paths


def _build_filter_cfg(f) -> ClassicalFilterConfig:
    """Translate ClassicalFiltersConfig → search.filters.config.ClassicalFilterConfig."""
    return ClassicalFilterConfig(
        min_base_girth_bound=f.min_base_girth_bound,
        min_girth_tanner_A_bin=f.min_girth_tanner_A_bin,
        require_any_block_col_full_rank=f.require_any_block_col_full_rank,
        min_entry_order_bound=f.min_entry_order_bound,
        min_abelianization_bound=f.min_abelianization_bound,
        min_weight_distance_bound=f.min_weight_distance_bound,
        min_ring_distance_bound=f.min_ring_distance_bound,
    )


def _count_existing(out_dir: Path, d_target: int) -> int:
    """Count JSONs under ``out_dir`` whose filename starts with ``d{≥d_target}_``."""
    if not out_dir.exists():
        return 0
    count = 0
    for p in out_dir.glob("d*.json"):
        # Filename: d{dist}_w{...}_t{...}_{ts}.json
        stem = p.stem
        if not stem.startswith("d"):
            continue
        try:
            dist = int(stem[1:].split("_", 1)[0])
        except ValueError:
            continue
        if dist >= d_target:
            count += 1
    return count


def _save_classical(*, M, M_bin, dist: Optional[int],
                    girth: Optional[int], wtag: str, side: str,
                    gd: GroupData, cfg: SearchConfig, out_dir: Path,
                    dist_filename: int) -> Path:
    """Write one classical-code JSON. Returns the path."""
    ts = int(time.time() * 1_000_000)
    W = weight_matrix(M)
    ma = len(M)
    na = len(M[0])
    n = gd.n

    # Structural flags (rep-independent — rank(L[x]) = rank(R[x])).
    f2rank = int(f2_rank(M_bin))
    coverage = bool(f2rank == M_bin.shape[0])
    anybl_full = bool(any_block_col_full_rank(M_bin, n, ma=ma))

    # Block-col containment map (ma=1 only): for each ordered pair
    # (i, j), True iff col(L[a_i]) ⊆ col(L[a_j]) over F2.
    containment_map: Optional[dict] = None
    has_containment: Optional[bool] = None
    if ma == 1:
        from core.group import left_rep
        import numpy as _np
        blocks = [left_rep(M[0][j], gd) for j in range(na)]
        block_ranks = [int(f2_rank(b)) for b in blocks]
        raw_map: dict = {}
        for i in range(na):
            for j in range(na):
                if i == j:
                    continue
                combined = _np.hstack([blocks[j], blocks[i]])
                raw_map[(i, j)] = bool(f2_rank(combined) == block_ranks[j])
        containment_map = {f"{i}_{j}": bool(v) for (i, j), v in raw_map.items()}
        has_containment = bool(any(raw_map.values()))

    # Active canonical-form verification: rerun canonicalize and check
    # the permutation is identity. Confirms both entry-canonical AND
    # block-col-canonical position.
    if side == "A":
        _, _, perm_check, _ = canonical_form_A(M, gd)
    else:
        _, _, perm_check, _ = canonical_form_B(M, gd)
    is_canonical = (perm_check == list(range(na)))

    s = cfg.classical.sampling
    sampling_metadata = {
        "seed": s.seed,
        "include_identity": s.include_identity,
        "min_element_order": s.min_element_order,
        "avoid_same_coset": s.avoid_same_coset,
        "max_tries": s.max_tries,
        "canonicalize": s.canonicalize,
    }

    data = {
        "gap_expr": cfg.group.gap_expr,
        "group_tag": cfg.group.tag,
        "n": n,
        "ma": ma,
        "na": na,
        "side": side,
        "shape": list(cfg.shape),
        "matrix_shape": list(M_bin.shape),
        "weight_matrix": W.tolist(),
        "matrix": _ring_matrix_to_json(M),
        "dist": dist,
        "dist_estimator": "bposd",
        "dist_num_trials": cfg.classical.distance.num_trials,
        "dist_n_workers": cfg.classical.distance.n_workers,
        "dist_osd_order": cfg.classical.distance.osd_order,
        "girth_tanner": girth,
        "f2_rank_M_bin": f2rank,
        "column_space_coverage": coverage,
        "any_block_col_full_rank": anybl_full,
        "block_col_containment_map": containment_map,
        "has_block_col_containment": has_containment,
        "is_canonical": is_canonical,
        "sampling_metadata": sampling_metadata,
        "provenance": build_provenance(
            cfg, phase="classical",
            module="search.phases.classical",
            function="run_classical",
        ),
        "timestamp": ts,
    }
    filename = classical_filename(
        dist_filename, wtag, cfg.classical.distance.num_trials, ts,
    )
    path = out_dir / filename
    path.write_text(json.dumps(data, indent=2))
    return path


def _ring_matrix_to_json(M) -> list:
    """Convert ring matrix (list of lists of tuples of int) to JSON-safe list."""
    return [[list(entry) for entry in row] for row in M]


def _update_manifest(cfg: SearchConfig, new_A: list, new_B: list) -> None:
    """Append a run record to manifest.json (creates if missing)."""
    if not new_A and not new_B:
        return
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
        "new_A": new_A,
        "new_B": new_B,
    })
    path.write_text(json.dumps(existing, indent=2))
