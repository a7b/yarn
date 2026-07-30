"""Guard tests for the shipped example configs and the YAML generator.

Every YAML in ``search/configs/examples/`` must load through the strict
loader, and the generator's output must itself be loadable — if the schema
drifts, these fail immediately rather than letting the shipped interface rot.
The e2e smoke actually RUNS a (shrunk) quickstart to keep the examples honest.
"""

from pathlib import Path

import pytest
import yaml

from search.configs.config import CanonicalConfig, SearchConfig
from search.configs.loader import from_dict, load_config

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "search" / "configs" / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.yaml"))
PIPELINE_EXAMPLES = [p for p in EXAMPLE_FILES if p.name != "canonical_sample.yaml"]


@pytest.mark.fast
def test_examples_dir_is_populated():
    names = {p.name for p in EXAMPLE_FILES}
    assert {"quickstart_nonabelian.yaml", "quickstart_abelian.yaml",
            "gpu_verify.yaml", "template_full.yaml",
            "canonical_sample.yaml"} <= names


@pytest.mark.fast
@pytest.mark.parametrize("path", PIPELINE_EXAMPLES, ids=lambda p: p.name)
def test_example_loads_through_strict_loader(path):
    cfg = load_config(path)
    assert isinstance(cfg, SearchConfig)
    assert cfg.source_path == path.resolve()
    # Interface promises kept by every example:
    assert not Path(str(cfg.results_dir)).is_absolute()
    assert cfg.classical.sampling.seed is not None, "examples must be reproducible"


@pytest.mark.gap
def test_canonical_example_loads():
    """The canonical example loads as a CanonicalConfig (gap: the canonical
    param dataclasses live behind the GAP-backed import chain)."""
    cfg = load_config(EXAMPLES_DIR / "canonical_sample.yaml")
    assert isinstance(cfg, CanonicalConfig)
    assert cfg.params.target == 4
    assert cfg.params.seed is not None, "examples must be reproducible"
    assert not Path(str(cfg.out_dir)).is_absolute()
    # one explicit group per config — no lists, no enumeration
    assert cfg.group == "SmallGroup(8,3)"


@pytest.mark.fast
def test_quickstart_nonabelian_shape():
    cfg = load_config(EXAMPLES_DIR / "quickstart_nonabelian.yaml")
    assert cfg.run_stages == ["classical", "pairing", "report"]
    assert cfg.classical.weight_A == [[3, 3]]
    assert cfg.pairing is not None and not cfg.pairing.sqetch_verify.enabled


@pytest.mark.fast
def test_quickstart_abelian_uses_weight_pattern():
    cfg = load_config(EXAMPLES_DIR / "quickstart_abelian.yaml")
    assert cfg.classical.weight_pattern is not None
    assert cfg.classical.weight_A is None


@pytest.mark.fast
def test_gpu_verify_enables_sqetch():
    cfg = load_config(EXAMPLES_DIR / "gpu_verify.yaml")
    sv = cfg.pairing.sqetch_verify
    assert sv.enabled and sv.num_trials >= 100_000


@pytest.mark.gap
@pytest.mark.parametrize("gap_expr,shape", [
    ("SymmetricGroup(3)", (1, 2)),      # non-abelian flavor
    ("CyclicGroup(6)", (2, 4)),         # abelian flavor
])
def test_generator_output_loads(gap_expr, shape):
    from search.configs.yaml_generator import generate_search_yaml
    text = generate_search_yaml(gap_expr, shape)
    cfg = from_dict(yaml.safe_load(text))
    assert cfg.shape == shape
    # The skeleton must be internally consistent: pairing section emitted
    # implies pairing listed in run_stages (and vice versa).
    assert ("pairing" in cfg.run_stages) == (cfg.pairing is not None)
    if cfg.classical.weight_A is not None:
        # Placeholder weights must be runnable (weight-0 entries are not).
        assert all(w > 0 for row in cfg.classical.weight_A for w in row)


@pytest.mark.gap
@pytest.mark.bposd
def test_quickstart_nonabelian_e2e_smoke(tmp_path):
    """The shipped quickstart, shrunk, must actually run and save codes."""
    from search.phases.classical import run_classical
    from search.phases.pairing import run_pairing
    from search.configs.paths import quantum_dir

    cfg = load_config(EXAMPLES_DIR / "quickstart_nonabelian.yaml")
    cfg.results_dir = tmp_path
    cfg.classical.sampling.total_samples = 8
    cfg.classical.distance.num_trials = 50
    cfg.pairing.bposd.num_trials = 100
    cfg.pairing.pool.max_pairs = 20
    cfg.pairing.pool.min_quantum_pool_size = 1

    res_c = run_classical(cfg)
    assert len(res_c["new_A"]) >= 1, "seeded shrunk run must keep >=1 A-side code"
    res_p = run_pairing(cfg)
    saved = res_p["new_quantum"]
    assert len(saved) >= 1
    for p in saved:
        assert Path(p).exists()
        assert Path(p).parent == quantum_dir(cfg)
