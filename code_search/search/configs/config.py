"""Dataclasses for the search-pipeline YAML schema.

Hierarchy::

    SearchConfig
    ├── shape: (ma, na)
    ├── group: GroupConfig
    ├── results_dir: Path
    ├── run_stages: list[str]   subset of ["classical", "pairing", "report"]
    ├── classical: ClassicalStageConfig
    │   ├── weight_A, weight_B (non-abelian) | weight_pattern (abelian)
    │   ├── sampling: SamplingConfig
    │   ├── filters: ClassicalFiltersConfig
    │   ├── distance: ClassicalDistanceConfig
    │   └── pool: PoolConfig
    ├── pairing: PairingStageConfig | None
    │   ├── bposd: BPOSDConfig
    │   ├── sqetch_verify: SqetchVerifyConfig
    │   ├── filters: PairingFiltersConfig
    │   └── pool: PairingPoolConfig
    ├── report: ReportStageConfig
    └── verbose: bool

These mirror the YAML schema closely. The loader (``loader.py``) does
the dict → dataclass mapping; phases consume the dataclasses directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class GroupConfig:
    """Group identification."""

    gap_expr: str
    tag: Optional[str] = None   # auto-derived from gap_expr if None


@dataclass
class SamplingConfig:
    """Per-entry sampling knobs forwarded to ``random_ring_matrix``."""

    total_samples: int
    seed: Optional[int] = None
    include_identity: bool = True
    min_element_order: int = 1
    avoid_same_coset: bool = False
    max_tries: int = 1000
    canonicalize: bool = True


@dataclass
class ClassicalFiltersConfig:
    """Mirrors ``search.filters.config.ClassicalFilterConfig`` field-for-field.

    Loader passes these straight through to the dispatcher.
    """

    # Group-agnostic
    min_girth_tanner_A_bin: Optional[int] = None
    require_any_block_col_full_rank: bool = False
    max_canonical_logical_weight: Optional[int] = None

    # ma=1 only
    min_entry_order_bound: Optional[int] = None
    min_abelianization_bound: Optional[int] = None

    # Abelian only
    min_base_girth_bound: Optional[int] = None
    min_weight_distance_bound: Optional[int] = None
    min_ring_distance_bound: Optional[int] = None


@dataclass
class ClassicalDistanceConfig:
    """Forwarded to ``estimate_classical_distance``."""

    d_target: int
    num_trials: int
    n_workers: int = 1
    osd_order: int = 5


@dataclass
class PoolConfig:
    """Reuse / skip behavior on the classical side."""

    min_pool_size: int = 0     # skip a weight if existing pool already has ≥ this many
    max_saved: int = 0          # 0 = no cap; else cap on the TOTAL pool per weight (pre-existing saved files count toward it)
    force_new: bool = False     # ignore min_pool_size check


@dataclass
class WeightPatternConfig:
    """Abelian-only: weight-pattern enumeration knobs (Chunk 2's two-stage trick).

    Stage 1 of the abelian classical search samples integer weight
    matrices ``W`` from ``[entry_min, entry_max]^(ma·na)``. Stage 2
    places ring elements into the surviving patterns.

    Set ``entry_min = 1`` to forbid empty entries (no zero placeholders).
    """

    entry_max: int
    num_weight_samples: int
    ring_samples_per_weight: int
    entry_min: int = 0

    # Pattern-level filters (apply to W directly, before any ring elements).
    max_row_weight: Optional[int] = None
    max_col_weight: Optional[int] = None
    min_base_girth_bound: Optional[int] = None
    min_weight_distance_bound: Optional[int] = None


@dataclass
class ClassicalStageConfig:
    """Classical sampling/filtering stage.

    Two flavors: **non-abelian** uses ``weight_A`` / ``weight_B`` as
    fixed per-entry weight matrices; **abelian** uses
    ``weight_pattern`` for the two-stage enumeration. Exactly one of
    these must be supplied; the phase code raises otherwise.
    """

    distance: ClassicalDistanceConfig
    sampling: SamplingConfig
    # Non-abelian path
    weight_A: Optional[List[List[int]]] = None
    weight_B: Optional[List[List[int]]] = None
    # Abelian path
    weight_pattern: Optional[WeightPatternConfig] = None
    # Shared
    filters: ClassicalFiltersConfig = field(default_factory=ClassicalFiltersConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)


@dataclass
class BPOSDConfig:
    """BP+OSD CSS distance estimator config (pairing stage)."""

    d_target: int
    num_trials: int
    n_workers: int = 1
    osd_order: int = 0


@dataclass
class SqetchVerifyConfig:
    """SQetch confirmation pass for the pairing stage.

    SQetch is the GPU implementation of QDistRnd's DistRandCSS algorithm;
    this is the heavier follow-up to the BP+OSD fast filter. Configures
    only the knobs the wrapper actually consumes: ``num_trials``,
    ``d_target``, ``devices``, ``strategy``, plus the kernel knobs
    ``k_sub``, ``batch_size``.
    """

    enabled: bool = False
    d_target: Optional[int] = None
    num_trials: int = 100_000
    devices: Optional[list] = None    # CUDA device indices; None = all visible
    strategy: str = "auto"            # auto / direction_split / trial_split
    k_sub: int = 64
    batch_size: int = 50_000


@dataclass
class PairingFiltersConfig:
    """Mirrors ``search.filters.config.QuantumPairingFilterConfig``."""

    require_same_group: bool = True
    require_same_shape: bool = True
    min_classical_distance: Optional[int] = None
    min_classical_girth: Optional[int] = None
    max_Hx_check_weight: Optional[int] = None
    max_Hz_check_weight: Optional[int] = None
    min_full_extractor_bridge_d: Optional[int] = None


@dataclass
class PairingPoolConfig:
    pair_mode: str = "full_pool"      # ("new_only" is not implemented yet)
    max_pairs: Optional[int] = None
    min_quantum_pool_size: int = 0


@dataclass
class PairingStageConfig:
    bposd: BPOSDConfig
    sqetch_verify: SqetchVerifyConfig = field(default_factory=SqetchVerifyConfig)
    filters: PairingFiltersConfig = field(default_factory=PairingFiltersConfig)
    pool: PairingPoolConfig = field(default_factory=PairingPoolConfig)


@dataclass
class ReportStageConfig:
    auto_report: bool = False


@dataclass
class CanonicalConfig:
    """Top-level config for ``mode: canonical`` (the canonical search).

    ``group`` is ONE explicit GAP group expression (e.g.
    ``"SmallGroup(84,11)"`` or ``"DirectProduct(CyclicGroup(5),
    SymmetricGroup(3))"``): one config = one group = one funnel run. There
    is no group enumeration or selection — to search several groups, write
    several configs.

    The funnel is the randomized, budgeted sweep
    (:func:`search.canonical.sample_search.run_group_sweep`); ``params``
    is its native knob dataclass
    (:class:`search.canonical.sample_search.SweepParams`), built by the
    loader from the YAML ``params`` section (strictly validated against
    the dataclass fields).
    """

    group: str                        # one GAP expression
    params: object                    # SweepParams instance
    out_dir: Path = field(default_factory=lambda: Path("canonical_results"))
    verbose: bool = False
    source_path: Optional[Path] = None


@dataclass
class SearchConfig:
    """Top-level search-run config."""

    shape: Tuple[int, int]
    group: GroupConfig
    run_stages: List[str]
    classical: ClassicalStageConfig
    pairing: Optional[PairingStageConfig] = None
    report: ReportStageConfig = field(default_factory=ReportStageConfig)
    results_dir: Path = field(default_factory=lambda: Path("search_results"))
    verbose: bool = False
    # Set by ``load_config`` to the absolute path of the YAML this config
    # was loaded from. ``None`` for programmatically-constructed configs.
    # Embedded in every saved JSON's ``provenance.config_path`` field.
    source_path: Optional[Path] = None
