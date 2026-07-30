"""Verify the new fields land in both classical and quantum saved JSONs."""

import json
from pathlib import Path

import numpy as np
import pytest

from search.configs.config import (
    BPOSDConfig,
    ClassicalDistanceConfig,
    ClassicalStageConfig,
    GroupConfig,
    PairingFiltersConfig,
    PairingPoolConfig,
    PairingStageConfig,
    SamplingConfig,
    SearchConfig,
    SqetchVerifyConfig,
)
from search.configs.paths import classical_A_dir, classical_B_dir


CLASSICAL_NEW_FIELDS = [
    "n", "ma", "na",
    "f2_rank_M_bin",
    "column_space_coverage", "any_block_col_full_rank",
    "block_col_containment_map", "has_block_col_containment",
    "is_canonical",
    "sampling_metadata",
    "provenance",
]

QUANTUM_NEW_FIELDS = [
    "ma", "na", "mb", "nb",
    "n_phys", "n_x_checks", "n_z_checks",
    "Hx_check_weight", "Hz_check_weight",
    "source_A", "source_B",
    "provenance",
]

SOURCE_SUMMARY_FIELDS = [
    "path", "dist", "weight_matrix",
    "column_space_coverage", "any_block_col_full_rank",
    "is_canonical",
]


def _cfg_classical(tmp_path: Path) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
        run_stages=["classical"],
        classical=ClassicalStageConfig(
            weight_A=[[2, 2]], weight_B=[[2, 2]],
            sampling=SamplingConfig(total_samples=2, seed=0, max_tries=200),
            distance=ClassicalDistanceConfig(
                d_target=2, num_trials=10, n_workers=1, osd_order=0,
            ),
        ),
        results_dir=tmp_path / "search_results",
    )


def _cfg_pairing(tmp_path: Path) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
        run_stages=["pairing"],
        classical=ClassicalStageConfig(
            weight_A=[[2, 2]], weight_B=[[2, 2]],
            sampling=SamplingConfig(total_samples=0, seed=0),
            distance=ClassicalDistanceConfig(d_target=2, num_trials=10),
        ),
        pairing=PairingStageConfig(
            bposd=BPOSDConfig(d_target=2, num_trials=20, n_workers=1, osd_order=0),
            sqetch_verify=SqetchVerifyConfig(enabled=False),
            filters=PairingFiltersConfig(
                require_same_group=True, require_same_shape=True,
            ),
            pool=PairingPoolConfig(),
        ),
        results_dir=tmp_path / "search_results",
    )


class TestClassicalNewFields:
    pytestmark = pytest.mark.bposd

    def test_all_new_fields_present(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _cfg_classical(tmp_path)
        out = run_classical(cfg)
        saved = out["new_A"] + out["new_B"]
        assert saved, "expected at least one saved classical code for this fixture"
        for path_str in saved:
            data = json.loads(Path(path_str).read_text())
            for field in CLASSICAL_NEW_FIELDS:
                assert field in data, f"missing {field!r} in {path_str}"
            # Provenance shape.
            prov = data["provenance"]
            assert prov["phase"] == "classical"
            assert prov["phase_module"] == "search.phases.classical"
            assert prov["phase_function"] == "run_classical"
            # is_canonical is True by default (canonicalize=True in SamplingConfig).
            assert data["is_canonical"] is True
            # block_col_containment_map exists for ma=1.
            assert data["block_col_containment_map"] is not None


class TestQuantumNewFields:
    pytestmark = pytest.mark.bposd

    def test_all_new_fields_present(self, tmp_path):
        from search.phases.pairing import run_pairing
        # Seed a small classical pool by running the classical phase first.
        from search.phases.classical import run_classical
        cfg = _cfg_classical(tmp_path)
        run_classical(cfg)
        # Now run pairing.
        cfg.pairing = PairingStageConfig(
            bposd=BPOSDConfig(d_target=2, num_trials=20, n_workers=1, osd_order=0),
            sqetch_verify=SqetchVerifyConfig(enabled=False),
            filters=PairingFiltersConfig(
                require_same_group=True, require_same_shape=True,
            ),
            pool=PairingPoolConfig(),
        )
        out = run_pairing(cfg)
        if not out["new_quantum"]:
            pytest.skip("No quantum codes passed under this fixture's budget.")
        for path_str in out["new_quantum"]:
            data = json.loads(Path(path_str).read_text())
            for field in QUANTUM_NEW_FIELDS:
                assert field in data, f"missing {field!r} in {path_str}"
            # source_A / source_B are summary dicts.
            for src in ("source_A", "source_B"):
                summary = data[src]
                assert isinstance(summary, dict)
                for field in SOURCE_SUMMARY_FIELDS:
                    assert field in summary, f"missing {field!r} in {src}"
            # Provenance.
            prov = data["provenance"]
            assert prov["phase"] == "pairing"
            assert prov["phase_module"] == "search.phases.pairing"
            # n_phys / n_x_checks / n_z_checks are positive ints.
            assert data["n_phys"] > 0
            assert data["n_x_checks"] > 0
            assert data["n_z_checks"] > 0
