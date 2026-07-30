"""Regressions for the adversarial-review findings on the YAML interface.

Each test pins a fix: rerun-continues semantics (classical content dedup +
pool-counting quantum stop), the trivial-kernel BP+OSD segfault guard, loader
validation of run_stages / weight shapes / groups.orders, and auto_report.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from search.configs.loader import from_dict, load_config

EXAMPLES_DIR = (Path(__file__).resolve().parent.parent
                / "search" / "configs" / "examples")

BASE = {
    "shape": [1, 2],
    "group": {"gap_expr": "SymmetricGroup(3)", "tag": "S3"},
    "run_stages": ["classical", "pairing"],
    "classical": {
        "weight_A": [[3, 3]], "weight_B": [[3, 3]],
        "sampling": {"total_samples": 8, "seed": 42},
        "distance": {"d_target": 4, "num_trials": 50},
    },
    "pairing": {
        "bposd": {"d_target": 3, "num_trials": 100},
        "pool": {"max_pairs": 20, "min_quantum_pool_size": 1},
    },
}


def _cfg(tmp_path, **over):
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    raw["results_dir"] = str(tmp_path)
    return from_dict(raw)


# ─────────────────────────────────────────────────────────────────
# Finding 1: reruns must CONTINUE, not duplicate
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
@pytest.mark.bposd
def test_rerun_classical_saves_nothing_new(tmp_path):
    from search.configs.paths import classical_A_dir
    from search.phases.classical import run_classical

    cfg = _cfg(tmp_path)
    first = run_classical(cfg)
    assert len(first["new_A"]) >= 1
    n_files = len(list(classical_A_dir(cfg).rglob("d*.json")))
    # identical seeded rerun: every sampled matrix is already in the pool
    second = run_classical(_cfg(tmp_path))
    assert second["new_A"] == [] and second["new_B"] == []
    assert len(list(classical_A_dir(cfg).rglob("d*.json"))) == n_files


@pytest.mark.gap
@pytest.mark.bposd
def test_intra_run_duplicate_samples_saved_once(tmp_path):
    """total_samples >> distinct matrices: the pool must hold unique
    matrices only (content dedup, not per-draw timestamped files)."""
    from search.configs.paths import classical_A_dir
    from search.phases.classical import run_classical

    over = json.loads(json.dumps(BASE))
    over["classical"]["sampling"] = {"total_samples": 60, "seed": 1}
    cfg = _cfg(tmp_path, **{"classical": over["classical"]})
    run_classical(cfg)
    files = list(classical_A_dir(cfg).rglob("d*.json"))
    mats = [repr(json.loads(p.read_text())["matrix"]) for p in files]
    assert len(mats) == len(set(mats)), "duplicate matrices saved"


@pytest.mark.gap
@pytest.mark.bposd
def test_min_quantum_pool_counts_previous_runs(tmp_path):
    from search.phases.classical import run_classical
    from search.phases.pairing import run_pairing

    cfg = _cfg(tmp_path)
    run_classical(cfg)
    first = run_pairing(cfg)
    assert len(first["new_quantum"]) >= 1
    # pool already satisfies min_quantum_pool_size=1 → rerun does nothing,
    # and stops BEFORE trying pairs (not merely via tried-pairs skipping)
    second = run_pairing(_cfg(tmp_path))
    assert second["new_quantum"] == []
    assert second["n_pairs_tried"] == 0


# ─────────────────────────────────────────────────────────────────
# Finding 3: trivial-kernel BP+OSD guard + shape/weight validation
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_bposd_classical_trivial_kernel_returns_none():
    from core.dist.classical import estimate_classical_distance_bposd
    H = np.eye(4, dtype=np.uint8)          # ker(H) = 0
    d = estimate_classical_distance_bposd(
        H, num_trials=50, n_workers=1, d_target=4, osd_order=5)
    assert d is None                        # used to SIGSEGV in the backend


@pytest.mark.fast
def test_loader_rejects_weight_shape_mismatch():
    raw = json.loads(json.dumps(BASE))
    raw["classical"]["weight_A"] = [[3, 3], [3, 3]]     # 2x2 vs shape 1x2
    with pytest.raises(ValueError, match="weight_A"):
        from_dict(raw)


# ─────────────────────────────────────────────────────────────────
# Finding 5: run_stages validated at load time
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_loader_rejects_unknown_stage():
    raw = json.loads(json.dumps(BASE))
    raw["run_stages"] = ["classical", "paring"]
    with pytest.raises(ValueError, match="paring"):
        from_dict(raw)


# ─────────────────────────────────────────────────────────────────
# Finding 6 (superseded by the explicit-groups redesign): a structured
# groups value must fail with the helpful list-of-expressions message
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_canonical_group_rejects_structured_values():
    from search.configs.loader import canonical_from_dict
    with pytest.raises(ValueError, match="ONE GAP group expression"):
        canonical_from_dict({
            "mode": "canonical",
            "group": {"orders": [84, 100]},
            "params": {"target": 16},
        })


# ─────────────────────────────────────────────────────────────────
# Finding 4: auto_report is wired
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
@pytest.mark.bposd
def test_auto_report_runs_report_after_pairing(tmp_path, monkeypatch):
    import search.runners.search as runner

    raw = json.loads(json.dumps(BASE))
    raw["report"] = {"auto_report": True}
    raw["results_dir"] = str(tmp_path / "res")
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(raw))

    called = {}
    monkeypatch.setattr(runner, "run_report",
                        lambda cfg: called.setdefault("hit", True) or "# r\n")
    assert runner.main([str(yaml_path), "--quiet"]) == 0
    assert called.get("hit") is True


# ─────────────────────────────────────────────────────────────────
# Finding 2 hardening: e2e smokes must pin code parameters
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
@pytest.mark.bposd
def test_saved_quantum_json_pins_n_phys_and_sqetch_schema(tmp_path):
    from search.phases.classical import run_classical
    from search.phases.pairing import run_pairing

    cfg = _cfg(tmp_path)
    run_classical(cfg)
    saved = run_pairing(cfg)["new_quantum"]
    assert saved
    data = json.loads(Path(saved[0]).read_text())
    assert data["n_phys"] == 30            # [[30, 6]] over S3, shape 1x2
    assert data["k"] == 6
    # sqetch record schema present even when the verify pass is disabled
    for key in ("sqetch_num_trials", "sqetch_k_sub", "sqetch_strategy",
                "sqetch_batch_size"):
        assert key in data


# ─────────────────────────────────────────────────────────────────
# 1x2 review finding: a k = 0 pair must be rejected, never saved
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
def test_pairing_rejects_k_zero_codes(tmp_path, monkeypatch):
    """A k = 0 pair must be rejected BEFORE any distance estimation: its
    (None, None) estimate would otherwise read as PASS and the degenerate
    code would be saved (the trap behind the old README-snippet bug)."""
    import search.phases.pairing as pairing
    from core.group import GroupData

    cfg = _cfg(tmp_path)
    gd = GroupData("SymmetricGroup(3)")
    meta = {"matrix": [[[0, 1, 3], [0, 2, 4]]]}
    monkeypatch.setattr(pairing, "compute_k", lambda Hx, Hz: 0)
    monkeypatch.setattr(
        pairing, "estimate_quantum_distances_bposd",
        lambda *a, **kw: pytest.fail("estimator must not run for k = 0"))
    ok, qcode, dist_info = pairing._evaluate_pair(meta, meta, gd, cfg)
    assert ok is False and qcode is None and dist_info is None
