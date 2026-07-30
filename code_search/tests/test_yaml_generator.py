"""Tests for ``search/configs/yaml_generator.py``."""

import pytest
import yaml

from search.configs.loader import from_dict, load_config
from search.configs.yaml_generator import (
    generate_search_yaml,
    write_search_yaml,
)


pytestmark = pytest.mark.gap


class TestGenerateAbelian:
    def test_yaml_parses_and_loads(self, tmp_path):
        text = generate_search_yaml("CyclicGroup(4)", (1, 2))
        raw = yaml.safe_load(text)
        # Loader accepts the generated dict end-to-end.
        cfg = from_dict(raw)
        assert cfg.classical.weight_pattern is not None
        assert cfg.classical.weight_pattern.entry_max == 3
        # Non-abelian-only fields must NOT appear in the abelian YAML.
        assert "avoid_same_coset" not in text
        assert "min_abelianization_bound" not in text
        # Abelian-only filters DO appear.
        assert "min_base_girth_bound" in text
        assert "min_ring_distance_bound" in text


class TestGenerateNonAbelian:
    def test_yaml_parses_and_loads(self):
        text = generate_search_yaml("SymmetricGroup(3)", (1, 2))
        raw = yaml.safe_load(text)
        cfg = from_dict(raw)
        # runnable starting-point weights (a weight-0 entry samples nothing)
        assert cfg.classical.weight_A == [[3, 3]]
        assert cfg.classical.weight_B == [[3, 3]]
        assert cfg.classical.weight_pattern is None
        # Non-abelian-only fields appear.
        assert "avoid_same_coset" in text
        assert "min_abelianization_bound" in text
        # Abelian-only filters do NOT appear.
        assert "min_base_girth_bound" not in text
        assert "min_ring_distance_bound" not in text
        assert "min_weight_distance_bound" not in text


class TestMaGt1():
    """ma=1-only filters disappear when ma > 1."""

    def test_ma2_omits_ma1_filters(self):
        text = generate_search_yaml("SymmetricGroup(3)", (2, 3))
        assert "min_entry_order_bound" not in text
        assert "min_abelianization_bound" not in text


class TestPairingOptOut:
    def test_no_pairing_excludes_section(self):
        text = generate_search_yaml(
            "SymmetricGroup(3)", (1, 2), include_pairing=False,
        )
        assert "pairing:" not in text
        assert "bposd:" not in text
        assert "sqetch_verify:" not in text

    def test_pairing_uses_sqetch_verify_name(self):
        """User-facing rename: it's `sqetch_verify`, not `gap_verify`."""
        text = generate_search_yaml("SymmetricGroup(3)", (1, 2))
        assert "sqetch_verify:" in text
        assert "gap_verify:" not in text


class TestWriteFile:
    def test_writes_and_round_trips(self, tmp_path):
        out = write_search_yaml(
            tmp_path / "my.yaml", "SymmetricGroup(3)", (1, 2),
        )
        assert out.exists()
        cfg = load_config(out)
        assert cfg.shape == (1, 2)
        assert cfg.group.gap_expr == "SymmetricGroup(3)"
        assert cfg.group.tag == "S3"


class TestCustomTag:
    def test_tag_override(self, tmp_path):
        text = generate_search_yaml(
            "SymmetricGroup(3)", (1, 2), group_tag="my_group",
        )
        assert "tag: my_group" in text


class TestCLI:
    def test_cli_runs(self, tmp_path, capsys):
        from search.configs.yaml_generator import _cli
        _cli(["SymmetricGroup(3)", "1", "2",
              "--output", str(tmp_path / "cli.yaml")])
        cfg = load_config(tmp_path / "cli.yaml")
        assert cfg.shape == (1, 2)
        assert cfg.group.tag == "S3"
