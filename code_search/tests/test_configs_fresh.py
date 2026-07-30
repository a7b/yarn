"""Fresh-eye contract tests for ``search/configs/``.

Covers the loader's rejection surface (unknown keys / missing required
fields in EVERY nested section that validates), the documented defaults,
a full non-default round-trip against a hand-built dataclass tree, YAML
failure modes, adversarial group-tag derivation, filename helpers, and
the yaml_generator -> load_config round trip with value-level checks.
"""

import copy
from pathlib import Path

import pytest
import yaml

from search.configs.config import (
    BPOSDConfig,
    ClassicalDistanceConfig,
    ClassicalFiltersConfig,
    ClassicalStageConfig,
    GroupConfig,
    PairingFiltersConfig,
    PairingPoolConfig,
    PairingStageConfig,
    PoolConfig,
    ReportStageConfig,
    SamplingConfig,
    SearchConfig,
    SqetchVerifyConfig,
    WeightPatternConfig,
)
from search.configs.loader import auto_group_tag, from_dict, load_config
from search.configs.paths import quantum_filename, report_path, tried_pairs_path


def _minimal_raw():
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


# ─────────────────────────────────────────────────────────────────
# Unknown-key rejection in every validated nested section
# ─────────────────────────────────────────────────────────────────


class TestUnknownKeysRejectedEverywhere:
    pytestmark = pytest.mark.fast

    @pytest.mark.parametrize("mutate,where", [
        (lambda r: r["group"].__setitem__("gap_exprr", "X"), "group"),
        (lambda r: r["classical"].__setitem__("weightA", 1), "classical"),
        (lambda r: r["classical"]["sampling"].__setitem__("seedd", 1),
         "classical.sampling"),
        (lambda r: r["classical"]["distance"].__setitem__("dtarget", 1),
         "classical.distance"),
        (lambda r: r["classical"].__setitem__("pool", {"min_poolsize": 1}),
         "classical.pool"),
        (lambda r: r.__setitem__("report", {"autoreport": True}), "report"),
    ])
    def test_typo_key_raises_and_names_the_section(self, mutate, where):
        raw = _minimal_raw()
        mutate(raw)
        with pytest.raises(ValueError) as ei:
            from_dict(raw)
        msg = str(ei.value)
        assert "Unknown key" in msg
        assert where in msg

    def test_weight_pattern_typo_key_raises(self):
        raw = _minimal_raw()
        del raw["classical"]["weight_A"], raw["classical"]["weight_B"]
        raw["classical"]["weight_pattern"] = {
            "entry_max": 2, "num_weight_samples": 5,
            "ring_samples_per_weight": 2, "entry_mim": 1,
        }
        with pytest.raises(ValueError, match="classical.weight_pattern"):
            from_dict(raw)

    def test_pairing_top_typo_key_raises(self):
        raw = _minimal_raw()
        raw["pairing"] = {
            "bposd": {"d_target": 6, "num_trials": 10},
            "bpsod": {},
        }
        with pytest.raises(ValueError, match="pairing"):
            from_dict(raw)


# ─────────────────────────────────────────────────────────────────
# Missing required fields fail loudly with a pointed message
# ─────────────────────────────────────────────────────────────────


class TestMissingRequiredFields:
    pytestmark = pytest.mark.fast

    @pytest.mark.parametrize("key", ["shape", "group", "run_stages",
                                     "classical"])
    def test_missing_top_level(self, key):
        raw = _minimal_raw()
        del raw[key]
        with pytest.raises(ValueError, match=key):
            from_dict(raw)

    def test_missing_group_gap_expr(self):
        raw = _minimal_raw()
        raw["group"] = {"tag": "S3"}
        with pytest.raises(ValueError, match="gap_expr"):
            from_dict(raw)

    def test_missing_sampling_section(self):
        raw = _minimal_raw()
        del raw["classical"]["sampling"]
        with pytest.raises(ValueError, match="sampling"):
            from_dict(raw)

    def test_missing_distance_section(self):
        raw = _minimal_raw()
        del raw["classical"]["distance"]
        with pytest.raises(ValueError, match="distance"):
            from_dict(raw)

    def test_missing_sampling_total_samples(self):
        raw = _minimal_raw()
        raw["classical"]["sampling"] = {"seed": 1}
        with pytest.raises(ValueError, match="total_samples"):
            from_dict(raw)

    @pytest.mark.parametrize("key", ["d_target", "num_trials"])
    def test_missing_distance_required(self, key):
        raw = _minimal_raw()
        del raw["classical"]["distance"][key]
        with pytest.raises(ValueError, match=key):
            from_dict(raw)

    @pytest.mark.parametrize("key", ["entry_max", "num_weight_samples",
                                     "ring_samples_per_weight"])
    def test_missing_weight_pattern_required(self, key):
        raw = _minimal_raw()
        del raw["classical"]["weight_A"], raw["classical"]["weight_B"]
        wp = {"entry_max": 2, "num_weight_samples": 5,
              "ring_samples_per_weight": 2}
        del wp[key]
        raw["classical"]["weight_pattern"] = wp
        with pytest.raises(ValueError, match=key):
            from_dict(raw)

    def test_missing_pairing_bposd(self):
        raw = _minimal_raw()
        raw["pairing"] = {"filters": {}}
        with pytest.raises(ValueError, match="bposd"):
            from_dict(raw)

    def test_both_weight_flavors_raise(self):
        raw = _minimal_raw()
        raw["classical"]["weight_pattern"] = {
            "entry_max": 2, "num_weight_samples": 5,
            "ring_samples_per_weight": 2,
        }
        with pytest.raises(ValueError, match="BOTH"):
            from_dict(raw)

    def test_neither_weight_flavor_raises(self):
        raw = _minimal_raw()
        del raw["classical"]["weight_A"], raw["classical"]["weight_B"]
        with pytest.raises(ValueError, match="weight_A"):
            from_dict(raw)

    @pytest.mark.parametrize("shape", [[1], [1, 2, 3]])
    def test_bad_shape_length_raises(self, shape):
        raw = _minimal_raw()
        raw["shape"] = shape
        with pytest.raises(ValueError, match="shape"):
            from_dict(raw)


# ─────────────────────────────────────────────────────────────────
# Documented defaults — asserted field by field
# ─────────────────────────────────────────────────────────────────


class TestDefaults:
    pytestmark = pytest.mark.fast

    def test_minimal_config_defaults_sweep(self):
        cfg = from_dict(_minimal_raw())
        # Top level
        assert cfg.results_dir == Path("search_results")
        assert cfg.verbose is False
        assert cfg.pairing is None
        assert cfg.source_path is None
        assert cfg.report == ReportStageConfig(auto_report=False)
        # Sampling defaults
        s = cfg.classical.sampling
        assert (s.seed, s.include_identity, s.min_element_order,
                s.avoid_same_coset, s.max_tries, s.canonicalize) == (
                    None, True, 1, False, 1000, True)
        # Distance defaults
        d = cfg.classical.distance
        assert (d.n_workers, d.osd_order) == (1, 5)
        # Pool defaults
        p = cfg.classical.pool
        assert (p.min_pool_size, p.max_saved, p.force_new) == (0, 0, False)
        # Classical filters: everything disabled
        assert cfg.classical.filters == ClassicalFiltersConfig()
        f = cfg.classical.filters
        assert f.min_girth_tanner_A_bin is None
        assert f.require_any_block_col_full_rank is False
        assert f.max_canonical_logical_weight is None

    def test_pairing_defaults_sweep(self):
        raw = _minimal_raw()
        raw["pairing"] = {"bposd": {"d_target": 6, "num_trials": 10}}
        cfg = from_dict(raw)
        pr = cfg.pairing
        assert pr.bposd == BPOSDConfig(d_target=6, num_trials=10,
                                       n_workers=1, osd_order=0)
        sv = pr.sqetch_verify
        assert (sv.enabled, sv.d_target, sv.num_trials, sv.devices,
                sv.strategy, sv.k_sub, sv.batch_size) == (
                    False, None, 100_000, None, "auto", 64, 50_000)
        # NOTE: pairing filters default same-group/same-shape to True.
        assert pr.filters == PairingFiltersConfig()
        assert pr.filters.require_same_group is True
        assert pr.filters.require_same_shape is True
        assert pr.pool == PairingPoolConfig(pair_mode="full_pool",
                                            max_pairs=None,
                                            min_quantum_pool_size=0)

    def test_explicit_null_pairing_and_report(self):
        raw = _minimal_raw()
        raw["pairing"] = None
        raw["report"] = None
        cfg = from_dict(raw)
        assert cfg.pairing is None
        assert cfg.report == ReportStageConfig()


# ─────────────────────────────────────────────────────────────────
# Full non-default round trip vs a hand-built dataclass tree
# ─────────────────────────────────────────────────────────────────


class TestFullRoundTrip:
    pytestmark = pytest.mark.fast

    def test_every_field_lands_where_documented(self):
        raw = {
            "shape": [2, 3],
            "group": {"gap_expr": "CyclicGroup(6)", "tag": "MYC6"},
            "run_stages": ["classical", "pairing", "report"],
            "classical": {
                "weight_A": [[1, 2, 3], [3, 2, 1]],
                "weight_B": [[2, 2, 2], [1, 1, 1]],
                "sampling": {
                    "total_samples": 77, "seed": 9,
                    "include_identity": False, "min_element_order": 2,
                    "avoid_same_coset": True, "max_tries": 55,
                    "canonicalize": False,
                },
                "filters": {
                    "min_girth_tanner_A_bin": 6,
                    "require_any_block_col_full_rank": True,
                    "max_canonical_logical_weight": 15,
                },
                "distance": {"d_target": 4, "num_trials": 12,
                             "n_workers": 3, "osd_order": 2},
                "pool": {"min_pool_size": 2, "max_saved": 7,
                         "force_new": True},
            },
            "pairing": {
                "bposd": {"d_target": 5, "num_trials": 40,
                          "n_workers": 2, "osd_order": 1},
                "sqetch_verify": {
                    "enabled": True, "d_target": 9, "num_trials": 123,
                    "devices": [0, 1], "strategy": "trial_split",
                    "k_sub": 96, "batch_size": 1000,
                },
                "filters": {
                    "require_same_group": False,
                    "require_same_shape": False,
                    "min_classical_distance": 4,
                    "min_classical_girth": 6,
                    "max_Hx_check_weight": 9,
                    "max_Hz_check_weight": 10,
                    "min_full_extractor_bridge_d": 4,
                },
                "pool": {"pair_mode": "full_pool", "max_pairs": 3,
                         "min_quantum_pool_size": 2},
            },
            "report": {"auto_report": True},
            "results_dir": "custom_out",
            "verbose": True,
        }
        expected = SearchConfig(
            shape=(2, 3),
            group=GroupConfig(gap_expr="CyclicGroup(6)", tag="MYC6"),
            run_stages=["classical", "pairing", "report"],
            classical=ClassicalStageConfig(
                weight_A=[[1, 2, 3], [3, 2, 1]],
                weight_B=[[2, 2, 2], [1, 1, 1]],
                weight_pattern=None,
                sampling=SamplingConfig(
                    total_samples=77, seed=9, include_identity=False,
                    min_element_order=2, avoid_same_coset=True,
                    max_tries=55, canonicalize=False,
                ),
                filters=ClassicalFiltersConfig(
                    min_girth_tanner_A_bin=6,
                    require_any_block_col_full_rank=True,
                    max_canonical_logical_weight=15,
                ),
                distance=ClassicalDistanceConfig(
                    d_target=4, num_trials=12, n_workers=3, osd_order=2,
                ),
                pool=PoolConfig(min_pool_size=2, max_saved=7,
                                force_new=True),
            ),
            pairing=PairingStageConfig(
                bposd=BPOSDConfig(d_target=5, num_trials=40,
                                  n_workers=2, osd_order=1),
                sqetch_verify=SqetchVerifyConfig(
                    enabled=True, d_target=9, num_trials=123,
                    devices=[0, 1], strategy="trial_split",
                    k_sub=96, batch_size=1000,
                ),
                filters=PairingFiltersConfig(
                    require_same_group=False, require_same_shape=False,
                    min_classical_distance=4, min_classical_girth=6,
                    max_Hx_check_weight=9, max_Hz_check_weight=10,
                    min_full_extractor_bridge_d=4,
                ),
                pool=PairingPoolConfig(pair_mode="full_pool", max_pairs=3,
                                       min_quantum_pool_size=2),
            ),
            report=ReportStageConfig(auto_report=True),
            results_dir=Path("custom_out"),
            verbose=True,
        )
        assert from_dict(raw) == expected

    def test_weight_pattern_round_trip_with_entry_min_default(self):
        raw = _minimal_raw()
        del raw["classical"]["weight_A"], raw["classical"]["weight_B"]
        raw["classical"]["weight_pattern"] = {
            "entry_max": 3, "num_weight_samples": 40,
            "ring_samples_per_weight": 5, "max_row_weight": 6,
            "min_base_girth_bound": 8,
        }
        cfg = from_dict(raw)
        assert cfg.classical.weight_pattern == WeightPatternConfig(
            entry_max=3, num_weight_samples=40, ring_samples_per_weight=5,
            entry_min=0, max_row_weight=6, max_col_weight=None,
            min_base_girth_bound=8, min_weight_distance_bound=None,
        )
        assert cfg.classical.weight_A is None
        assert cfg.classical.weight_B is None

    def test_from_dict_does_not_mutate_input(self):
        raw = _minimal_raw()
        snapshot = copy.deepcopy(raw)
        from_dict(raw)
        assert raw == snapshot

    def test_run_stages_order_preserved(self):
        raw = _minimal_raw()
        raw["run_stages"] = ["report", "classical"]
        cfg = from_dict(raw)
        assert cfg.run_stages == ["report", "classical"]


# ─────────────────────────────────────────────────────────────────
# YAML file failure modes
# ─────────────────────────────────────────────────────────────────


class TestYamlFailureModes:
    pytestmark = pytest.mark.fast

    def test_empty_yaml_file_raises_value_error(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(p)

    def test_syntactically_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("shape: [1, 2\ngroup: {")
        with pytest.raises(yaml.YAMLError):
            load_config(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_source_path_is_resolved_absolute(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(_minimal_raw()))
        cfg = load_config(p)
        assert cfg.source_path == p.resolve()
        assert cfg.source_path.is_absolute()

    def test_yaml_dump_load_identity(self, tmp_path):
        # A config written by safe_dump must load to the same object as
        # the dict route.
        raw = _minimal_raw()
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(raw))
        via_file = load_config(p)
        via_dict = from_dict(raw)
        via_dict.source_path = via_file.source_path
        assert via_file == via_dict


# ─────────────────────────────────────────────────────────────────
# auto_group_tag — adversarial expressions
# ─────────────────────────────────────────────────────────────────


class TestAutoGroupTagFresh:
    pytestmark = pytest.mark.fast

    def test_nested_direct_product(self):
        expr = ("DirectProduct(DirectProduct(CyclicGroup(2), "
                "CyclicGroup(3)), SymmetricGroup(3))")
        assert auto_group_tag(expr) == "C2_x_C3_x_S3"

    def test_direct_product_with_spaces(self):
        expr = "DirectProduct( CyclicGroup(5) , SymmetricGroup(3) )"
        assert auto_group_tag(expr) == "C5_x_S3"

    def test_two_digit_order(self):
        assert auto_group_tag("CyclicGroup(12)") == "C12"
        assert auto_group_tag("SymmetricGroup(10)") == "S10"

    def test_fallback_is_filename_safe(self):
        tag = auto_group_tag("SmallGroup(24, 3)")
        assert tag == "SmallGroup_24_3"
        for ch in " (),":
            assert ch not in tag

    def test_leading_trailing_whitespace_stripped(self):
        assert auto_group_tag("  CyclicGroup(6)  ") == "C6"


# ─────────────────────────────────────────────────────────────────
# Filename / path helpers
# ─────────────────────────────────────────────────────────────────


class TestPathHelpersFresh:
    pytestmark = pytest.mark.fast

    def test_quantum_filename_none_distance_renders_X(self):
        name = quantum_filename(k=6, dx=None, dz=4, wA_tag="1.1",
                                wB_tag="1.1", bposd_trials=100,
                                timestamp=7)
        assert name == "k6_dxX_dz4_wA1.1_wB1.1_bposd100_7.json"

    def test_quantum_filename_sqetch_part_only_when_positive(self):
        with_sq = quantum_filename(k=6, dx=4, dz=4, wA_tag="1.1",
                                   wB_tag="1.1", bposd_trials=100,
                                   timestamp=7, sqetch_trials=500)
        without = quantum_filename(k=6, dx=4, dz=4, wA_tag="1.1",
                                   wB_tag="1.1", bposd_trials=100,
                                   timestamp=7, sqetch_trials=0)
        assert "_sqetch500_" in with_sq
        assert "sqetch" not in without

    def test_tried_pairs_and_report_locations(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(results_dir=Path("root"),
                              group=SimpleNamespace(tag="C6"),
                              shape=(1, 2))
        assert tried_pairs_path(cfg) == Path(
            "root/C6/1x2/quantum/tried_pairs.json")
        assert report_path(cfg) == Path("root/C6/1x2/report.md")


# ─────────────────────────────────────────────────────────────────
# yaml_generator -> load_config value-level round trip (needs GAP)
# ─────────────────────────────────────────────────────────────────


class TestYamlGeneratorRoundTripValues:
    pytestmark = pytest.mark.gap

    def test_non_abelian_generated_values_load_as_documented(self, tmp_path):
        from search.configs.yaml_generator import write_search_yaml
        out = write_search_yaml(tmp_path / "s3.yaml", "SymmetricGroup(3)",
                                (1, 2))
        cfg = load_config(out)
        assert cfg.shape == (1, 2)
        assert cfg.group.tag == "S3"
        # run_stages is consistent with the emitted sections (pairing
        # included by default => it is listed as a stage).
        assert cfg.run_stages == ["classical", "pairing", "report"]
        # Non-abelian flavor: fixed weight matrices, no weight_pattern.
        # Weight-3 starting points are runnable (weight-0 samples nothing).
        assert cfg.classical.weight_A == [[3, 3]]
        assert cfg.classical.weight_B == [[3, 3]]
        assert cfg.classical.weight_pattern is None
        # Template defaults documented in the generator.
        assert cfg.classical.sampling.total_samples == 1000
        assert cfg.classical.sampling.seed == 42
        assert cfg.classical.distance.d_target == 6
        # The no-op canonical-logical-weight default = na * |G| = 2 * 6.
        assert cfg.classical.filters.max_canonical_logical_weight == 12
        # ma=1 non-abelian: abelianization filter offered (disabled).
        assert cfg.classical.filters.min_abelianization_bound is None
        # Abelian-only filters must NOT appear for S3.
        assert cfg.classical.filters.min_base_girth_bound is None
        assert cfg.pairing is not None
        assert cfg.pairing.bposd.d_target == 6
        assert cfg.pairing.sqetch_verify.enabled is False

    def test_abelian_generated_yaml_uses_weight_pattern(self, tmp_path):
        from search.configs.yaml_generator import write_search_yaml
        out = write_search_yaml(tmp_path / "c6.yaml", "CyclicGroup(6)",
                                (1, 2))
        cfg = load_config(out)
        assert cfg.group.tag == "C6"
        assert cfg.classical.weight_A is None
        wp = cfg.classical.weight_pattern
        assert wp is not None
        assert wp.entry_max == 3
        assert wp.entry_min == 0
        assert wp.num_weight_samples == 100
        assert wp.ring_samples_per_weight == 10
        assert cfg.classical.filters.max_canonical_logical_weight == 12

    def test_no_pairing_flag_loads_with_pairing_none(self, tmp_path):
        from search.configs.yaml_generator import write_search_yaml
        out = write_search_yaml(tmp_path / "c6np.yaml", "CyclicGroup(6)",
                                (1, 2), include_pairing=False)
        cfg = load_config(out)
        assert cfg.pairing is None

    def test_tag_override_flows_to_loaded_config(self, tmp_path):
        from search.configs.yaml_generator import write_search_yaml
        out = write_search_yaml(tmp_path / "t.yaml", "CyclicGroup(6)",
                                (1, 2), group_tag="mytag")
        cfg = load_config(out)
        assert cfg.group.tag == "mytag"

    def test_bad_shape_raises(self):
        from search.configs.yaml_generator import generate_search_yaml
        with pytest.raises(ValueError, match="shape"):
            generate_search_yaml("CyclicGroup(6)", (0, 2))
