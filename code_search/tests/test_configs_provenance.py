"""Tests for ``search/configs/provenance.py`` and the ``source_path`` plumbing."""

from pathlib import Path

import pytest
import yaml

from search.configs.config import (
    ClassicalDistanceConfig,
    ClassicalStageConfig,
    GroupConfig,
    SamplingConfig,
    SearchConfig,
)
from search.configs.loader import load_config
from search.configs.provenance import build_provenance


pytestmark = pytest.mark.fast


def _minimal_yaml(tmp_path, name="cfg.yaml"):
    cfg = {
        "shape": [1, 2],
        "group": {"gap_expr": "SymmetricGroup(3)", "tag": "S3"},
        "run_stages": ["classical"],
        "classical": {
            "weight_A": [[2, 2]],
            "weight_B": [[2, 2]],
            "sampling": {"total_samples": 0},
            "distance": {"d_target": 4, "num_trials": 10},
        },
    }
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


class TestSourcePath:
    def test_load_config_sets_source_path(self, tmp_path):
        path = _minimal_yaml(tmp_path)
        cfg = load_config(path)
        assert cfg.source_path == path.resolve()

    def test_programmatic_cfg_has_no_source_path(self):
        cfg = SearchConfig(
            shape=(1, 2),
            group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
            run_stages=["classical"],
            classical=ClassicalStageConfig(
                weight_A=[[2, 2]], weight_B=[[2, 2]],
                sampling=SamplingConfig(total_samples=0),
                distance=ClassicalDistanceConfig(d_target=4, num_trials=10),
            ),
        )
        assert cfg.source_path is None


class TestBuildProvenance:
    def test_required_fields_present(self, tmp_path):
        path = _minimal_yaml(tmp_path)
        cfg = load_config(path)
        prov = build_provenance(
            cfg, phase="classical",
            module="search.phases.classical", function="run_classical",
        )
        assert prov["phase"] == "classical"
        assert prov["phase_module"] == "search.phases.classical"
        assert prov["phase_function"] == "run_classical"
        assert prov["config_path"] == str(path.resolve())
        assert "python_executable" in prov
        # git_commit is whatever the surrounding repo reports — may be None
        # or a 40-char SHA.
        assert "git_commit" in prov
        if prov["git_commit"] is not None:
            assert len(prov["git_commit"]) == 40

    def test_no_source_path_renders_as_null(self):
        cfg = SearchConfig(
            shape=(1, 2),
            group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
            run_stages=["classical"],
            classical=ClassicalStageConfig(
                weight_A=[[2, 2]], weight_B=[[2, 2]],
                sampling=SamplingConfig(total_samples=0),
                distance=ClassicalDistanceConfig(d_target=4, num_trials=10),
            ),
        )
        prov = build_provenance(
            cfg, phase="classical",
            module="search.phases.classical", function="run_classical",
        )
        assert prov["config_path"] is None
