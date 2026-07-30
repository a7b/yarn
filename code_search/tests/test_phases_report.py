"""Tests for ``search/phases/report.py``."""

import json
from pathlib import Path

import pytest

from search.configs.config import (
    ClassicalDistanceConfig,
    ClassicalStageConfig,
    GroupConfig,
    SamplingConfig,
    SearchConfig,
)
from search.configs.paths import quantum_dir, report_path
from search.phases.report import run_report


pytestmark = pytest.mark.fast


def _cfg(tmp_path):
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
        run_stages=["report"],
        classical=ClassicalStageConfig(
            weight_A=[[2, 2]], weight_B=[[2, 2]],
            sampling=SamplingConfig(total_samples=0),
            distance=ClassicalDistanceConfig(d_target=4, num_trials=10),
        ),
        results_dir=tmp_path / "search_results",
    )


def _seed_quantum(cfg, *records):
    qdir = quantum_dir(cfg)
    qdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, r in enumerate(records):
        path = qdir / f"k{r['k']}_dx{r['dx']}_dz{r['dz']}_wA2.2_wB2.2_bposd100_{i}.json"
        path.write_text(json.dumps(r))
        paths.append(path)
    return paths


class TestEmptyPool:
    def test_no_quantum_files(self, tmp_path):
        cfg = _cfg(tmp_path)
        md = run_report(cfg)
        assert "Total quantum codes saved: **0**" in md
        # Report file is created at the default location.
        assert report_path(cfg).exists()


class TestSortAndRender:
    def _records(self):
        return [
            {"k": 4, "dx": 6, "dz": 7, "weight_A": [[2, 2]], "weight_B": [[2, 2]],
             "estimator": "bposd", "bposd_num_trials": 100, "sqetch_num_trials": 0,
             "timestamp": 1},
            {"k": 6, "dx": 4, "dz": 4, "weight_A": [[2, 2]], "weight_B": [[2, 2]],
             "estimator": "bposd+sqetch", "bposd_num_trials": 100, "sqetch_num_trials": 1000,
             "timestamp": 2},
            {"k": 6, "dx": 5, "dz": 5, "weight_A": [[2, 2]], "weight_B": [[2, 2]],
             "estimator": "bposd", "bposd_num_trials": 100, "sqetch_num_trials": 0,
             "timestamp": 3},
        ]

    def test_sort_order_is_k_then_min_d(self, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_quantum(cfg, *self._records())
        md = run_report(cfg)
        # k=6 codes first; among k=6 the one with min_d=5 ranks above min_d=4.
        idx_k6_d5 = md.find("estimator | bposd_trials") + 1
        # Look at the line order after the header.
        body = md.split("|---|---|----|----|----|----|-----------|")[-1]
        # Row order should mention "5 | 5" before "4 | 4" (both have k=6),
        # before "6 | 7" (k=4).
        i_55 = body.find("| 5 | 5 |")
        i_44 = body.find("| 4 | 4 |")
        i_67 = body.find("| 6 | 7 |")
        assert i_55 != -1 and i_44 != -1 and i_67 != -1
        assert i_55 < i_44 < i_67

    def test_estimator_breakdown_in_summary(self, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_quantum(cfg, *self._records())
        md = run_report(cfg)
        assert "`bposd`: 2" in md
        assert "`bposd+sqetch`: 1" in md

    def test_top_n_limits_rows(self, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_quantum(cfg, *self._records())
        md = run_report(cfg, top_n=2)
        # Only two data rows. Count table data row markers.
        data_rows = [line for line in md.splitlines()
                     if line.startswith("| 1 |") or line.startswith("| 2 |")
                     or line.startswith("| 3 |")]
        assert len(data_rows) == 2

    def test_none_dist_renders_as_dash(self, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_quantum(cfg, {
            "k": 8, "dx": None, "dz": None,
            "weight_A": [[2, 2]], "weight_B": [[2, 2]],
            "estimator": "bposd", "bposd_num_trials": 100, "sqetch_num_trials": 0,
            "timestamp": 99,
        })
        md = run_report(cfg)
        # None renders as the em-dash.
        assert "| 8 | — | — |" in md

    def test_skip_write_with_dash_output(self, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_quantum(cfg, *self._records())
        md = run_report(cfg, output="-")
        assert isinstance(md, str)
        # Not written to default location.
        assert not report_path(cfg).exists()
