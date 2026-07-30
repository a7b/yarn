"""Tests for ``search/configs/loader.py``."""

from pathlib import Path

import pytest

from search.configs.config import SearchConfig
from search.configs.loader import auto_group_tag, from_dict, load_config


pytestmark = pytest.mark.fast


class TestAutoGroupTag:
    def test_symmetric(self):
        assert auto_group_tag("SymmetricGroup(3)") == "S3"

    def test_cyclic(self):
        assert auto_group_tag("CyclicGroup(4)") == "C4"

    def test_dihedral(self):
        assert auto_group_tag("DihedralGroup(8)") == "D8"

    def test_alternating(self):
        assert auto_group_tag("AlternatingGroup(5)") == "A5"

    def test_direct_product_two(self):
        assert auto_group_tag(
            "DirectProduct(SymmetricGroup(3), CyclicGroup(5))"
        ) == "S3_x_C5"

    def test_direct_product_three(self):
        assert auto_group_tag(
            "DirectProduct(CyclicGroup(5), SymmetricGroup(3), SymmetricGroup(3))"
        ) == "C5_x_S3_x_S3"

    def test_fallback_sanitize(self):
        tag = auto_group_tag("SomeWeirdGroup(2, 3)")
        # Just verify it doesn't crash and produces something filename-safe.
        assert " " not in tag
        assert "(" not in tag
        assert ")" not in tag


class TestFromDict:
    def _minimal_raw(self):
        return {
            "shape": [1, 2],
            "group": {"gap_expr": "SymmetricGroup(3)"},
            "run_stages": ["classical"],
            "classical": {
                "weight_A": [[2, 2]],
                "weight_B": [[2, 2]],
                "sampling": {"total_samples": 10},
                "distance": {"d_target": 8, "num_trials": 100},
            },
        }

    def test_minimal_roundtrip(self):
        cfg = from_dict(self._minimal_raw())
        assert isinstance(cfg, SearchConfig)
        assert cfg.shape == (1, 2)
        assert cfg.group.gap_expr == "SymmetricGroup(3)"
        assert cfg.group.tag == "S3"   # auto-derived
        assert cfg.run_stages == ["classical"]
        assert cfg.classical.weight_A == [[2, 2]]
        assert cfg.classical.weight_B == [[2, 2]]

    def test_unknown_top_key_raises(self):
        raw = self._minimal_raw()
        raw["bogus_field"] = 42
        with pytest.raises(ValueError, match="Unknown key"):
            from_dict(raw)

    def test_unknown_filter_key_raises(self):
        raw = self._minimal_raw()
        raw["classical"]["filters"] = {"min_typo_bound": 8}
        with pytest.raises(ValueError, match="Unknown key"):
            from_dict(raw)

    def test_max_canonical_logical_weight_roundtrips(self):
        raw = self._minimal_raw()
        raw["classical"]["filters"] = {"max_canonical_logical_weight": 22}
        cfg = from_dict(raw)
        assert cfg.classical.filters.max_canonical_logical_weight == 22

    def _with_pairing(self, filters):
        raw = self._minimal_raw()
        raw["run_stages"] = ["classical", "pairing"]
        raw["pairing"] = {
            "bposd": {"d_target": 8, "num_trials": 100},
            "filters": filters,
        }
        return raw

    def test_pairing_full_extractor_bridge_field_roundtrips(self):
        cfg = from_dict(self._with_pairing({"min_full_extractor_bridge_d": 8}))
        assert cfg.pairing.filters.min_full_extractor_bridge_d == 8

    def test_unknown_pairing_filter_key_raises(self):
        # mirrors classical.filters strictness: a typo must not silently disable.
        raw = self._with_pairing({"min_full_extractor_bridge_typo": 8})
        with pytest.raises(ValueError, match="Unknown key"):
            from_dict(raw)

    def test_missing_required_raises(self):
        raw = self._minimal_raw()
        del raw["shape"]
        with pytest.raises(ValueError, match="missing required key 'shape'"):
            from_dict(raw)


class TestLoadConfig:
    def test_yaml_roundtrip(self, tmp_path):
        yaml_text = """
shape: [1, 2]
group:
  gap_expr: SymmetricGroup(3)
  tag: S3
run_stages: [classical]
classical:
  weight_A: [[2, 3]]
  weight_B: [[3, 2]]
  sampling:
    total_samples: 100
    seed: 42
  distance:
    d_target: 6
    num_trials: 1000
  filters:
    min_abelianization_bound: 12
  pool:
    min_pool_size: 5
results_dir: my_results
verbose: true
"""
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml_text)
        cfg = load_config(path)
        assert cfg.classical.sampling.seed == 42
        assert cfg.classical.filters.min_abelianization_bound == 12
        assert cfg.classical.pool.min_pool_size == 5
        assert cfg.results_dir == Path("my_results")
        assert cfg.verbose is True
