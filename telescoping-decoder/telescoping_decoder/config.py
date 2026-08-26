"""Configuration objects and shipped parameter sets.

Stages: S1 (GPU flooding BP), S2 (GPU relay-BP with coset quorum), S3-A/B/C
(CPU relay stages), S4 (Gurobi IP certification).

The defaults match the bundled benchmark configuration. Comments document
parameter combinations whose behavior depends on the selected linear system.
``tests/test_config.py`` checks values used by the stage builders.

Every BP stage has a ``system`` field selecting which linear system it
decodes:

  "auto"      — the stage's default chain (the default). S1 prefers the
                init-basis-detector system: init_dets -> gari -> original.
                S2/S3 prefer GARI: gari -> original. A candidate is skipped
                when the decoder was not built with what it needs.
  "gari"      — the GARI-transformed system, relevant-half acceptance.
  "original"  — the correlated XYZ DEM as-is, full-row acceptance.
  "init_dets" — the init-basis-detectors-only system derived from the
                original matrices (needs the X/Z detector mask + init basis).

An explicit system is strict: asking for one the decoder cannot build is an
error, never a silent fallback. Only "auto" walks a chain.

S4 always solves the original correlated XYZ DEM (optionally restricted to
the init-detector family first via ``init_dets_only``).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

_SYSTEMS = ("auto", "gari", "original", "init_dets")

# ---------------------------------------------------------------------------
# S1 operating points, one per decode system.
# ---------------------------------------------------------------------------
# Each row is a benchmarked pairing of BP knobs with a system; a row taken
# apart is not benchmarked. In particular hybrid_sp_iters=5 — right for the
# init-dets and original systems — lands the min-sum tail on an oscillating
# phase of the GARI graph and accepts only 19.3%, where sp_iters>=6 accepts
# ~62%. S1Config ships the init_dets row used by the bundled benchmark; when
# S1 resolves to another system and the knobs are still untouched defaults,
# ``s1_config_for()`` swaps in that system's row rather than shipping an
# unbenchmarked mix.
S1_PRESETS = {
    "init_dets": {"k": 32,  "n_iters": 10, "hybrid_sp_iters": 5,
                  "shots_per_batch": 2048},
    "original":  {"k": 192, "n_iters": 10, "hybrid_sp_iters": 5,
                  "shots_per_batch": 512},
    "gari":      {"k": 224, "n_iters": 30, "hybrid_sp_iters": 6,
                  "shots_per_batch": 256},
}
# The row S1Config's field defaults are copied from.
DEFAULT_S1_SYSTEM = "init_dets"
# Knobs that make a preset a preset. shots_per_batch is excluded: it is a
# throughput/VRAM knob (see the sizing table in the README), so tuning it
# must not look like "the user tuned the operating point".
_S1_PRESET_KEYS = ("k", "n_iters", "hybrid_sp_iters")

# ---------------------------------------------------------------------------
# S3 variant tables used by the bundled benchmark.
# ---------------------------------------------------------------------------

DEFAULT_S3A_VARIANTS = (
    # (label, prior_scale, noise_std, seed_offset)
    ("std", 1.0, 0.0,         0),
    ("rnd", 1.0, 0.0,  1_000_000),
    ("opt", 0.8, 0.0,    100_000),
    ("pes", 1.2, 0.0,    200_000),
    ("thm", 1.0, 0.2,    300_000),
)

DEFAULT_S3B_VARIANTS = (
    (1.0, 0.0), (0.8, 0.0), (1.2, 0.0),
    (1.0, 0.2), (0.8, 0.2), (1.2, 0.2),
)

# S3-C sequential BP configurations. Each 6-tuple contains:
#   (label, prior_scale, iters, noise_std, alpha_max, n_seeds_override).
# n_seeds_override=None falls back to c_n_seeds_default (10 for noisy
# variants, 1 when noise_std=0). A set of deeper rescue variants used to
# follow these; they were removed from the shipped set because their convergence
# rode the budget edge — marginal basins BP cannot certify — and produced the
# only wrong-coset acceptance seen at this stage. Shots that fail the range
# below now go to S4, which is exact and faster per shot. The indices below
# are unchanged by that removal, so these decodes are bit-identical to the
# recorded benchmark runs.
DEFAULT_S3C_VARIANTS = (
    ("gc1",  0.8, 1000, 0.3, 0.99,  None),
    ("gc2",  0.6, 1000, 0.1, 0.9,   None),
    ("gc3",  0.7,  500, 0.2, 0.99,  None),
    ("gc5",  0.6,  200, 0.1, 0.8,   None),
    ("gc6",  1.2,  500, 0.3, 0.9,   None),
    ("gc7",  1.1, 1000, 0.2, 0.9,   None),
    ("gc8",  1.5, 1000, 0.2, 0.9,   None),
    ("gc9",  0.5,  200, 0.2, 0.9,   None),
    ("gc10", 1.1,  500, 0.3, 0.8,   None),
    ("gc11", 1.3,  500, 0.0, 0.99,  None),
    ("gc12", 0.5, 1000, 0.1, 0.8,   None),
    ("gc13", 0.5, 1000, 0.3, 0.95,  None),
)


@dataclasses.dataclass
class S1Config:
    """S1 — GPU flooding/layered BP, run on every shot.

    The knob defaults are the ``init_dets`` row of ``S1_PRESETS`` — the
    shipped first stage: the init-basis-detector system is ~8x smaller
    than GARI (660x9600 vs 19575x94620 on the bundled [[150,30,10]]
    artifacts) and accepts ~96.5% of shots at p=1e-3, so S2 sees a third of
    the traffic it would otherwise. ``system="auto"`` selects it whenever
    the decoder knows the X/Z detector mask + init basis.
    """
    enabled: bool = True
    system: str = "auto"
    # Throughput/VRAM knob, never a result knob — see the README sizing
    # table. 2048 shots is ~0.8 GB on the init-dets system, ~5 GB on GARI.
    shots_per_batch: int = 2048
    k: int = 32
    n_iters: int = 10
    algorithm: str = "hybrid"
    # 5 pairs with init_dets/original. On GARI this must be >=6 — see the
    # S1_PRESETS comment; s1_config_for() handles the swap.
    hybrid_sp_iters: int = 5
    var_nwy: int = 4
    alpha_max: float = 0.95
    alpha_tau: float = 4.0


@dataclasses.dataclass
class S2Config:
    """S2 — GPU relay-BP with coset quorum, run on the S1 rejects."""
    enabled: bool = True
    system: str = "auto"
    shots_per_batch: int = 384
    k: int = 64
    # Single inner-decoder batch size, distinct from the outer driver slice.
    # The batch position participates in S2's gamma seed, so changing this
    # value can change an S2 result as well as memory use and throughput.
    batch_size: int = 2048
    # Single prior configuration: (prior_scale, noise_std).
    priors: tuple = ((1.0, 0.0),)
    # Coset-quorum: accept iff this many (half-)converged legs agree on the
    # coset, else defer (NC -> escalate to S3/S4).
    #
    # Defaults to 3, not 2, on the GARI system: under the relevant-half
    # acceptance rule legs converge far more often than under full-row, so
    # two legs corroborating the same wrong coset occurred in the benchmark.
    # With quorum=2, three wrong-coset shots were accepted in five
    # million, all of which quorum=3 correctly deferred to S3/S4 for two
    # extra safe deferrals and about one extra leg per accepted shot. On the
    # original system, where full-row convergence is the rule, the benchmarked
    # value is 2.
    quorum: int = 3
    # Project operating point for the Relay-BP-derived schedule. This is not
    # an exact copy of any code-specific configuration reported in the paper.
    num_slots: int = 1
    num_sets: int = 40
    leg_max_iter: int = 60
    gamma0: float = 0.1
    gamma0_iters: int = 80
    gamma_center: float = 0.21
    gamma_width: float = 0.90
    alpha: float = 1.0
    alpha_const: bool = True
    tau: float = 5.0


@dataclasses.dataclass
class S3Config:
    """S3 — CPU relay stages A/B/C, per-shot pool.

    Field prefixes map to the public sub-stages: ``a_*`` = S3-A (standalone
    variant ensemble), ``b_*`` = S3-B (relay-mem BP with first-to-quorum
    coset voting), ``c_*`` = S3-C (sequential BP parameter sweep).
    """
    enabled: bool = True
    system: str = "auto"
    n_procs: int | None = None            # None -> os.cpu_count()
    mp_start_method: str = "forkserver"   # safe alongside a CUDA parent

    # S3-A
    a_variants: tuple = DEFAULT_S3A_VARIANTS
    a_iters: int = 100
    a_alpha_max: float = 0.99
    a_tau: float = 5.0

    # S3-B
    b_variants: tuple = DEFAULT_S3B_VARIANTS
    b_runs: int = 40
    b_iters: int = 100
    # 0.90 (was 0.99): alpha<=0.95 clears a known wrong-coset LE basin; 0.90
    # produced no S3-B logical errors in the benchmark data, with deferrals
    # resolving correctly at S3-C or S4.
    b_alpha_max: float = 0.90
    b_tau: float = 5.0
    b_min_cost_select: bool = False
    # Relay-BP with memory (per-iteration signed memory, ordered gamma0
    # leg, marginal warm-start) + min-weight/quorum acceptance.
    b_use_memory: bool = True
    b_gamma0: float = 0.1          # ordered leg-0 memory strength
    b_gamma0_iters: int = 80       # leg-0 iteration count
    b_gamma_center: float = 0.21   # disordered gamma ~ U[center +/- width/2]
    b_gamma_width: float = 0.90    # benchmark range [-0.24, 0.66]; tune per graph
    b_stop_nconv: int = 5          # >1 so min-weight has candidates to pick
    b_quorum: int = 3              # >=quorum legs must agree on the coset,
                                   # else defer (NC). 1 = min-weight-only.

    # S3-C
    c_tau: float = 5.0
    c_variants: tuple = DEFAULT_S3C_VARIANTS
    c_n_seeds_default: int = 10    # fallback when n_seeds_override is None


@dataclasses.dataclass
class S4Config:
    """S4 — Gurobi IP certification on the original correlated XYZ DEM.

    ``enabled="auto"``: active iff gurobipy is importable; shots surviving
    S3-C without it are returned as NC (label ``NC_no_ip``).
    """
    enabled: bool | str = "auto"      # "auto" | True | False
    n_procs: int | None = None
    license_file: str | None = None   # -> GRB_LICENSE_FILE in workers
    # Per-shot Gurobi TimeLimit (seconds) for the 1-hop sub-DEM solve.
    time_limit_subdem_s: float = 450.0
    # Per-shot TimeLimit for the full-DEM escalation solve (run only if the
    # sub-DEM solve is rejected by the gap rule).
    full_time_limit_s: float = 900.0
    # Sub-DEM acceptance: accept if status=OPTIMAL or MIPGap < this
    # threshold. Rejection -> escalate to the full-DEM solve.
    gap_threshold: float = 0.5
    # Full-DEM acceptance: accept OPTIMAL, or a time-limited incumbent with
    # MIPGap < this threshold; otherwise the shot remains uncertified
    # (solver give-up, not a proven logical error).
    full_gap_threshold: float = 0.10
    hops: int = 1
    # Solve on the init-basis-detectors-only system instead of the full XYZ
    # DEM (one exact solve, no sub-DEM heuristic): LER-equivalent and ~16x
    # faster per shot on the validated 150-r10 set.
    init_dets_only: bool = False
    # MIPFocus=2 (proof-focused). Production adjudication found a shot
    # reported as an uncertified give-up under MIPFocus=1 that this setting
    # solves to proven optimality in minutes, so the give-up was a false
    # positive rather than a real logical error.
    mip_focus: int = 2


@dataclasses.dataclass
class TelescopeConfig:
    """Top-level knobs + the four stage configs."""
    base_seed: int = 42
    # Master switch for the GARI system. False = never build/use GARI (all
    # "auto" systems resolve to "original").
    use_gari: bool = True
    # When > 0, LE shots are additionally classified by observable group:
    # bit0 = rows [:n] (measured/time-like), bit1 = rows [n:] (unmeasured).
    # Post-hoc accounting only; never affects decoding.
    n_measured_observables: int = 0
    s1: S1Config = dataclasses.field(default_factory=S1Config)
    s2: S2Config = dataclasses.field(default_factory=S2Config)
    s3: S3Config = dataclasses.field(default_factory=S3Config)
    s4: S4Config = dataclasses.field(default_factory=S4Config)

    def __post_init__(self) -> None:
        for name in ("s1", "s2", "s3"):
            sys_ = getattr(self, name).system
            if sys_ not in _SYSTEMS:
                raise ValueError(
                    f"{name}.system={sys_!r} not in {_SYSTEMS}")

    def _flatten(self) -> SimpleNamespace:
        """Flatten the nested stage configs into one namespace.

        The S3 and S4 worker code reads a single flat config object rather
        than the nested dataclasses, because worker processes receive it
        through their pool initializer and per-shot code should not have to
        know which stage it belongs to. Field names are prefixed by stage
        (``s3a_*``, ``s3b_*``, ``s3c_*``, ``s4_*``).

        Every ``cfg.<attr>`` read in ``s3.py`` and ``s4_ip.py`` must have a
        key here; ``tests/test_config.py`` asserts that.
        """
        s3, s4 = self.s3, self.s4
        return SimpleNamespace(
            base_seed=self.base_seed,
            n_measured_observables=self.n_measured_observables,
            # S3-A
            s3a_variants=tuple(s3.a_variants),
            s3a_iters=s3.a_iters,
            s3a_alpha_max=s3.a_alpha_max,
            s3a_tau=s3.a_tau,
            # S3-B
            s3b_variants=tuple(s3.b_variants),
            s3b_runs=s3.b_runs,
            s3b_iters=s3.b_iters,
            s3b_alpha_max=s3.b_alpha_max,
            s3b_tau=s3.b_tau,
            s3b_min_cost_select=s3.b_min_cost_select,
            s3b_use_memory=s3.b_use_memory,
            s3b_gamma0=s3.b_gamma0,
            s3b_gamma0_iters=s3.b_gamma0_iters,
            s3b_gamma_center=s3.b_gamma_center,
            s3b_gamma_width=s3.b_gamma_width,
            s3b_stop_nconv=s3.b_stop_nconv,
            s3b_quorum=s3.b_quorum,
            # S3-C
            s3c_tau=s3.c_tau,
            s3c_variants=tuple(s3.c_variants),
            s3c_n_seeds_default=s3.c_n_seeds_default,
            # S4
            s4_time_limit_subdem_s=s4.time_limit_subdem_s,
            s4_full_time_limit_s=s4.full_time_limit_s,
            s4_gap_threshold=s4.gap_threshold,
            s4_full_gap_threshold=s4.full_gap_threshold,
            s4_hops=s4.hops,
            s4_init_dets_only=s4.init_dets_only,
            s4_mip_focus=s4.mip_focus,
        )


def s1_config_for(s1: S1Config, resolved_system: str):
    """The S1 config to actually decode ``resolved_system`` with.

    S1Config's defaults are the ``init_dets`` operating point. If S1 resolved
    to a different system and the knobs are untouched, return a copy carrying
    that system's ``S1_PRESETS`` row plus a message explaining the swap; the
    alternative is using an unbenchmarked knob/system pairing (and on
    GARI, one that oscillates). Any user edit to ``k``/``n_iters``/
    ``hybrid_sp_iters`` disables the swap — an explicit choice wins.

    Returns ``(config, message_or_None)``; the input config is never mutated.
    """
    preset = S1_PRESETS.get(resolved_system)
    if preset is None or resolved_system == DEFAULT_S1_SYSTEM:
        return s1, None
    shipped = S1_PRESETS[DEFAULT_S1_SYSTEM]
    if any(getattr(s1, key) != shipped[key] for key in _S1_PRESET_KEYS):
        return s1, None          # user tuned the operating point; respect it
    changes = {key: preset[key] for key in _S1_PRESET_KEYS}
    if s1.shots_per_batch == shipped["shots_per_batch"]:
        changes["shots_per_batch"] = preset["shots_per_batch"]
    return dataclasses.replace(s1, **changes), (
        f"s1.system resolved to {resolved_system!r}, not the default "
        f"{DEFAULT_S1_SYSTEM!r} the S1 knobs are benchmarked for; using the "
        f"{resolved_system!r} preset instead "
        + ", ".join(f"{k}={v}" for k, v in changes.items())
        + ". Set the S1 knobs explicitly to silence this.")


__all__ = [
    "S1Config", "S2Config", "S3Config", "S4Config", "TelescopeConfig",
    "DEFAULT_S3A_VARIANTS", "DEFAULT_S3B_VARIANTS", "DEFAULT_S3C_VARIANTS",
    "S1_PRESETS", "DEFAULT_S1_SYSTEM", "s1_config_for",
]
