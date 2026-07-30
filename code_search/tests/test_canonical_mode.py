"""Tests for the ``mode: canonical`` YAML interface.

Canonical mode searches EXACTLY ONE explicitly named group per config (a
GAP expression) — no lists, no enumeration, no selection. Covers: loader
dispatch + strict validation, group-spec parsing, the driver (with stubbed
funnels), the CLI dispatch, and one real shrunk end-to-end run on the GPU.
"""

import json
from pathlib import Path

import pytest
import yaml

from search.configs.config import CanonicalConfig, SearchConfig
from search.configs.loader import canonical_from_dict, from_dict, load_config

MINIMAL = {
    "mode": "canonical",
    "group": "SmallGroup(8,3)",
    "params": {"target": 4},
}


def _write_yaml(tmp_path, data, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


# ─────────────────────────────────────────────────────────────────
# Loader: mode dispatch + rejection before any heavy import
# ─────────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_unknown_mode_rejected(tmp_path):
    p = _write_yaml(tmp_path, {"mode": "bogus"})
    with pytest.raises(ValueError, match="unknown mode"):
        load_config(p)


@pytest.mark.fast
def test_canonical_unknown_top_key_rejected():
    with pytest.raises(ValueError, match="Unknown key"):
        canonical_from_dict({**MINIMAL, "tyop": 1})


@pytest.mark.fast
def test_canonical_removed_surfaces_rejected():
    """Band keys, plural groups lists, and the strategy key may not load."""
    with pytest.raises(ValueError, match="Unknown key"):
        canonical_from_dict({**MINIMAL, "stop_at_first_verified_order": True})
    raw = {k: v for k, v in MINIMAL.items() if k != "group"}
    with pytest.raises(ValueError, match="Unknown key"):
        canonical_from_dict({**raw, "groups": ["SmallGroup(8,3)"]})
    with pytest.raises(ValueError, match="Unknown key"):
        canonical_from_dict({**MINIMAL, "strategy": "brute"})


@pytest.mark.fast
def test_canonical_missing_required_keys_rejected():
    for missing in ("group", "params"):
        raw = {k: v for k, v in MINIMAL.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            canonical_from_dict(raw)


@pytest.mark.fast
def test_canonical_group_must_be_one_expression_string():
    for bad in (["SmallGroup(8,3)"], ["SmallGroup(8,3)", "SmallGroup(8,4)"],
                8, "", {"gap_expr": "SmallGroup(8,3)"}):
        with pytest.raises(ValueError, match="ONE GAP group expression"):
            canonical_from_dict({**MINIMAL, "group": bad})


@pytest.mark.fast
def test_pipeline_mode_key_still_loads_pipeline():
    raw = {
        "mode": "pipeline",
        "shape": [1, 2],
        "group": {"gap_expr": "SymmetricGroup(3)"},
        "run_stages": ["classical"],
        "classical": {
            "weight_A": [[3, 3]], "weight_B": [[3, 3]],
            "sampling": {"total_samples": 1},
            "distance": {"d_target": 2, "num_trials": 5},
        },
    }
    assert isinstance(from_dict(raw), SearchConfig)


# ─────────────────────────────────────────────────────────────────
# Loader: canonical schema (imports the GAP-backed param dataclasses)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
def test_canonical_minimal_loads_with_native_defaults():
    from search.canonical.sample_search import SweepParams
    cfg = canonical_from_dict(MINIMAL)
    assert isinstance(cfg, CanonicalConfig)
    assert cfg.group == "SmallGroup(8,3)"
    assert isinstance(cfg.params, SweepParams)
    assert cfg.params.target == 4
    # untouched knobs keep the dataclass defaults
    assert cfg.params.max_pass == SweepParams().max_pass
    assert cfg.out_dir == Path("canonical_results")


@pytest.mark.gap
def test_canonical_params_strictly_validated():
    with pytest.raises(ValueError, match="params"):
        canonical_from_dict({**MINIMAL, "params": {"target": 4, "tyop": 1}})
    with pytest.raises(ValueError, match="target"):
        canonical_from_dict({**MINIMAL, "params": {"seed": 1}})
    # a knob of the library-only exhaustive funnel (ScreenParams) is a
    # typo here, not a silent no-op
    with pytest.raises(ValueError, match="params"):
        canonical_from_dict({**MINIMAL, "params": {"target": 4,
                                                   "certify_seeds": [1]}})


# ─────────────────────────────────────────────────────────────────
# Group-spec parsing
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
def test_spec_for_small_group_and_generic_expressions():
    from search.canonical.driver import spec_for
    from search.canonical.groups import GroupSpec

    s = spec_for("SmallGroup(84,11)")
    assert isinstance(s, GroupSpec)
    assert (s.order, s.small_group_id, s.tag) == (84, 11, "Sg84_11")
    assert s.gap_expr == "SmallGroup(84,11)"

    g = spec_for("DirectProduct(CyclicGroup(5),SymmetricGroup(3))")
    assert g.gap_expr == "DirectProduct(CyclicGroup(5),SymmetricGroup(3))"
    assert g.tag and "(" not in g.tag and " " not in g.tag


# ─────────────────────────────────────────────────────────────────
# Driver (funnels stubbed — semantics only)
# ─────────────────────────────────────────────────────────────────


def _canonical_cfg(tmp_path, **over):
    raw = {**MINIMAL, "out_dir": str(tmp_path / "out"), **over}
    return canonical_from_dict(raw)


@pytest.mark.gap
def test_driver_runs_the_one_group_and_persists_summary(tmp_path, monkeypatch):
    import search.canonical.driver as driver
    calls = []

    def fake_sweep(spec, params, out_dir, *, gd=None, provenance=None, log=None):
        calls.append(spec.tag)
        return {"passed": ["codedir"], "verdict": "pass"}

    monkeypatch.setattr(driver, "run_group_sweep", fake_sweep)
    cfg = _canonical_cfg(tmp_path)
    res = driver.run_canonical(cfg)

    assert calls == ["Sg8_3"]
    assert res["group"] == "Sg8_3"
    assert res["gap_expr"] == "SmallGroup(8,3)"
    assert res["n_passed"] == 1 and res["verdict"] == "pass"
    saved = json.loads(
        (Path(cfg.out_dir) / "canonical_summary.json").read_text())
    assert saved == res


# ─────────────────────────────────────────────────────────────────
# CLI dispatch
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
def test_cli_dispatches_canonical(tmp_path, monkeypatch):
    import search.canonical.driver as driver
    from search.runners.search import main

    got = {}

    def fake_run(cfg, *, log=None):
        got["cfg"] = cfg
        return {"group": "Sg8_3", "n_passed": 0, "verdict": "fail"}

    monkeypatch.setattr(driver, "run_canonical", fake_run)
    p = _write_yaml(tmp_path, {**MINIMAL, "out_dir": str(tmp_path / "o")})
    assert main([str(p), "--quiet"]) == 0
    assert isinstance(got["cfg"], CanonicalConfig)
    assert got["cfg"].group == "SmallGroup(8,3)"


@pytest.mark.gap
def test_cli_rejects_stages_for_canonical(tmp_path):
    from search.runners.search import main
    p = _write_yaml(tmp_path, {**MINIMAL, "out_dir": str(tmp_path / "o")})
    assert main([str(p), "--stages", "classical"]) == 2


# ─────────────────────────────────────────────────────────────────
# Real shrunk end-to-end run (GPU)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.gap
@pytest.mark.gpu
def test_canonical_e2e_real_d8(tmp_path):
    """Real GAP group + real sqetch funnel on D8, target 2."""
    from search.canonical.driver import run_canonical

    cfg = canonical_from_dict({
        "mode": "canonical",
        "out_dir": str(tmp_path / "camp"),
        "group": "SmallGroup(8,3)",
        "params": {"target": 2, "cl_num_trials": 20_000,
                   "q_num_trials": 50_000, "surv_cap": 5,
                   "cl_sample_budget": 500, "cl_barren_skip": 500,
                   "pair_budget": 50, "pair_fail": 50, "max_pass": 1,
                   "seed": 3},
    })
    res = run_canonical(cfg)
    assert res["group"] == "Sg8_3"
    assert res["n_passed"] >= 1
    summary = json.loads(
        (Path(cfg.out_dir) / "canonical_summary.json").read_text())
    assert summary["gap_expr"] == "SmallGroup(8,3)"
    # the funnel's own artifacts landed too (sample funnel: _sweep_<tag>.json)
    assert list(Path(cfg.out_dir).glob("_sweep_*.json"))
    code_jsons = list(Path(cfg.out_dir).glob("Sg8_*/code.json"))
    assert code_jsons
    rich = json.loads(code_jsons[0].read_text())
    assert rich["provenance"]["mode"] == "canonical"
