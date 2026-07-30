"""Tests for ``search/runners/search.py``."""

import json
from pathlib import Path

import pytest
import yaml

from search.configs.paths import report_path
from search.runners.search import main


pytestmark = pytest.mark.gap


def _write_yaml(tmp_path: Path, gap_expr: str = "SymmetricGroup(3)") -> Path:
    """A minimal classical-only YAML suitable for CLI exercise."""
    config = {
        "shape": [1, 2],
        "group": {"gap_expr": gap_expr, "tag": "S3"},
        "run_stages": ["classical"],
        "classical": {
            "weight_A": [[2, 2]],
            "weight_B": [[2, 2]],
            "sampling": {"total_samples": 0, "seed": 0},
            "distance": {"d_target": 4, "num_trials": 10, "n_workers": 1,
                          "osd_order": 0},
        },
        "results_dir": str(tmp_path / "search_results"),
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class TestRunnerBasic:
    def test_runs_default_stages(self, tmp_path, capsys):
        yml = _write_yaml(tmp_path)
        rc = main([str(yml)])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "classical" in captured

    def test_quiet_suppresses_summary(self, tmp_path, capsys):
        yml = _write_yaml(tmp_path)
        rc = main([str(yml), "--quiet"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert captured == ""

    def test_stages_override(self, tmp_path, capsys):
        yml = _write_yaml(tmp_path)
        # Override to JUST report (with empty quantum pool — should be a no-op).
        rc = main([str(yml), "--stages", "report"])
        assert rc == 0
        # The report file lands at the expected place even with an empty pool.
        # Verify by re-parsing the YAML to find the resolved path.
        # (We just check that no classical files were written.)
        from search.configs.loader import load_config
        cfg = load_config(yml)
        # report.md exists; classical_A/ does NOT.
        assert report_path(cfg).exists()
        assert not (cfg.results_dir / "S3" / "1x2" / "classical_A").exists()

    def test_unknown_stage_choice_rejects(self, tmp_path):
        yml = _write_yaml(tmp_path)
        # argparse rejects unknown choices with SystemExit.
        with pytest.raises(SystemExit):
            main([str(yml), "--stages", "bogus_stage"])
