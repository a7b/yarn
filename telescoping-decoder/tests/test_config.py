"""The flat config must cover every field the stage workers read.

S3 and S4 worker code reads a flat namespace produced by
``TelescopeConfig._flatten()``. A missing key there fails only at decode
time, deep inside a worker process, so assert the coverage statically.
"""
import re
from pathlib import Path

import pytest

from telescoping_decoder import TelescopeConfig
from telescoping_decoder.config import (DEFAULT_S1_SYSTEM, S1_PRESETS,
                                        s1_config_for)

PKG = Path(__file__).resolve().parent.parent / "telescoping_decoder"


def _cfg_reads(module_name: str) -> set:
    """Every attribute read off a `cfg` object in the given module."""
    src = (PKG / module_name).read_text()
    attrs = set(re.findall(r"\bcfg(?:_obj)?\.([A-Za-z_][A-Za-z0-9_]*)", src))
    # getattr(cfg, "name", default) reads too
    attrs |= set(re.findall(
        r'getattr\(\s*cfg(?:_obj)?\s*,\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]',
        src))
    return attrs


@pytest.mark.parametrize("module", ["s3.py", "s4_ip.py"])
def test_flatten_covers_worker_reads(module):
    flat = TelescopeConfig()._flatten()
    missing = sorted(a for a in _cfg_reads(module) if not hasattr(flat, a))
    assert not missing, (
        f"{module} reads cfg attributes absent from "
        f"TelescopeConfig._flatten(): {missing}")


def test_flatten_is_stage_prefixed():
    flat = vars(TelescopeConfig()._flatten())
    allowed_bare = {"base_seed", "n_measured_observables"}
    for name in flat:
        if name in allowed_bare:
            continue
        assert re.match(r"^s(3[abc]|4)_", name), (
            f"flat config field {name!r} should be stage-prefixed "
            "(s3a_/s3b_/s3c_/s4_) or one of the shared fields")


def test_production_defaults():
    """Guard the benchmarked defaults against accidental edits."""
    cfg = TelescopeConfig()
    # S1 ships the init_dets operating point used by the bundled benchmark.
    assert (cfg.s1.k, cfg.s1.n_iters, cfg.s1.algorithm) == (32, 10, "hybrid")
    assert cfg.s1.hybrid_sp_iters == 5      # 5 off GARI, 6 on it
    assert (cfg.s2.k, cfg.s2.batch_size, cfg.s2.quorum) == (64, 2048, 3)
    assert (cfg.s2.num_sets, cfg.s2.leg_max_iter) == (40, 60)
    assert len(cfg.s3.a_variants) == 5
    assert len(cfg.s3.b_variants) == 6
    assert len(cfg.s3.c_variants) == 12
    assert (cfg.s3.b_quorum, cfg.s3.b_alpha_max) == (3, 0.90)
    assert (cfg.s4.gap_threshold, cfg.s4.full_gap_threshold) == (0.5, 0.10)
    assert cfg.s4.mip_focus == 2


def test_s2_has_one_convergence_threshold():
    """The quorum is both S2's freeze threshold and acceptance count."""
    assert "stop_nconv" not in vars(TelescopeConfig().s2)


def test_s1_defaults_are_a_preset_row():
    """S1Config's knob defaults must be exactly one S1_PRESETS row."""
    s1 = TelescopeConfig().s1
    row = S1_PRESETS[DEFAULT_S1_SYSTEM]
    assert {k: getattr(s1, k) for k in row} == row


@pytest.mark.parametrize("resolved", ["gari", "original"])
def test_untouched_knobs_take_the_resolved_systems_preset(resolved):
    """Falling back off init_dets must not ship a mixed knob/system pairing
    (hybrid_sp_iters=5 on GARI oscillates)."""
    s1 = TelescopeConfig().s1
    out, note = s1_config_for(s1, resolved)
    row = S1_PRESETS[resolved]
    assert note is not None and resolved in note
    assert {k: getattr(out, k) for k in row} == row
    assert s1.k == S1_PRESETS[DEFAULT_S1_SYSTEM]["k"]   # input untouched


def test_tuned_knobs_win_over_the_preset():
    s1 = TelescopeConfig().s1
    s1.n_iters = 17
    out, note = s1_config_for(s1, "gari")
    assert note is None and out is s1 and out.n_iters == 17


def test_default_system_keeps_its_own_knobs():
    s1 = TelescopeConfig().s1
    out, note = s1_config_for(s1, DEFAULT_S1_SYSTEM)
    assert note is None and out is s1


def test_invalid_system_rejected():
    with pytest.raises(ValueError):
        cfg = TelescopeConfig()
        cfg.s1.system = "nonsense"
        cfg.__post_init__()
