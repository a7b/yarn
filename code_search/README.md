# code_search

A search toolkit for high-rate lifted-product (LP) CSS quantum codes over
finite groups. It bundles the math core (GF(2) + group-algebra linear
algebra, code builders, distance estimators), canonical logical bases, and a
**YAML-driven search pipeline**: you fix a group and a shape in a config
file, run one command, and the tool samples, filters, distance-screens, and
saves codes with full provenance.

```bash
cp search/configs/examples/quickstart_nonabelian.yaml my_run.yaml
python -m search.runners.search my_run.yaml     # ~15 s on CPU
```

## Layout

| Folder | What it provides |
|---|---|
| `core/` | GF(2) linear algebra (`f2`, bit-packed `f2_fast`); GAP-backed group machinery (`GroupData`, F₂[G] ring arithmetic, left/right regular representations); classical (`A_bin`/`B_bin`) and CSS (`Hx`/`Hz`) LP code builders; `core/dist/` distance estimators |
| `logical_basis/` | Canonical paired logical bases: `Lx·Lzᵀ = I_k`, G-orbit-preserving construction |
| `search/` | The search pipeline: `sampling/`, `filters/`, `families/`, `phases/`, `configs/` (+ examples), `runners/` (CLI), `canonical/` (the canonical-basis funnel) |
| `tests/` | ~1000 tests; markers `fast` / `gap` / `bposd` / `gpu` |

Each folder has its own README with details; `search/README.md` is the
pipeline quickstart.

## Requirements

- Python 3.11+, `numpy`, `pyyaml` — always.
- **GAP** with the `gappy` bindings — for constructing groups
  (`core.group.GroupData`), i.e. for any actual search.
- Optional, per feature: `ldpc` (BP+OSD distance screens), CUDA +
  `torch` + the `sqetch` package — shipped in this repository — for GPU
  distance estimation and the canonical funnel.

Run commands from this folder (the package root), or put it on
`PYTHONPATH`.

## The YAML interface

One command drives everything; the top-level `mode` key picks the workflow:

```bash
python -m search.runners.search <config.yaml> [--stages ...] [--quiet]
python -m search.runners.new_config "<gap_expr>" <ma> <na> -o my.yaml   # tailored skeleton
```

Both modes share the interface contract:

- **Strict configs.** Unknown or misspelled keys anywhere in the YAML fail
  at load time with the allowed-key list — no silent typos, and no compute
  spent before validation.
- **Distances are estimates.** Every reported distance is an upper bound
  from randomized trials; a stage's `d_target` PASS means "nothing lighter
  was found within the budget" (finding nothing at all also passes). Cheap
  screens are deliberately optimistic — confirmation passes with larger
  budgets (GPU sqetch) exist to refute them.
- **Provenance.** Every saved artifact embeds the config path, entry
  point, and git commit that produced it.
- **Reruns continue.** Running the same config again reuses saved pools
  (content-deduped), counts previously saved codes toward stop conditions,
  and skips already-tried pairs. Delete the output dir to start fresh.

### Pipeline mode (default): fix a group and a shape, search ring matrices

Stages, run in order per `run_stages`:

1. **classical** — sample random ring matrices over your fixed group and
   `(ma, na)` shape, in one of two flavors: fixed per-entry **weight
   matrices** (`weight_A`/`weight_B`), or — abelian groups only — two-stage
   **weight-pattern** enumeration (`weight_pattern`: sample integer weight
   matrices first, then place ring elements into survivors). Cheap
   structural filters reject candidates before the BP+OSD distance screen
   keeps sides with `d ≥ d_target`.
2. **pairing** — pair surviving A/B sides (abelian: `B = A*` automatically),
   apply pair filters, build `Hx`/`Hz`, screen `(dx, dz)` with BP+OSD, and
   optionally confirm with **sqetch on the GPU** (`sqetch_verify`) before
   saving. Caps: `max_pairs`, `min_quantum_pool_size`.
3. **report** — render `report.md` ranking the saved quantum codes
   (`auto_report: true` runs it right after pairing).

Available filters (enable by setting a value; `null` = off):

- Classical, any group: Tanner-graph girth (`min_girth_tanner_A_bin`),
  invertible block-column (`require_any_block_col_full_rank`), canonical
  logical weight cap (`max_canonical_logical_weight`).
- Classical, `ma = 1`: entry-order bound; non-abelian additionally the
  abelianization bound.
- Classical, abelian: base-girth, weight-distance, and ring-distance
  bounds (pattern-level variants exist inside `weight_pattern`).
- Pairing: same group/shape requirements, minimum classical distance and
  girth of the sides, `Hx`/`Hz` check-weight caps, and the
  full-extractor bridge-distance filter (`min_full_extractor_bridge_d`).

Sampling constraints: per-entry weights, `include_identity`,
`min_element_order`, `avoid_same_coset` (non-abelian), canonicalization,
seed. Distance screens: trials, `d_target`, workers, OSD order per stage;
GPU verify adds device/strategy/`k_sub`/batch knobs.

Outputs land under `results_dir/<group_tag>/<ma>x<na>/`:
`classical_A/`, `classical_B/` (per-side pools, one JSON + structural flags
per code), `quantum/` (one JSON per passed code with matrices, distances,
girths, weights, back-links to its sources), `report.md`, `manifest.json`.

### Canonical mode: one group, the canonical-basis funnel

```yaml
mode: canonical
group: SmallGroup(8,3)      # ONE GAP expression — one config = one group
params:
  target: 4                 # distance target
  # ... maps 1:1 onto SweepParams (the funnel's knob dataclass)
```

Searches shape-(1,2) LP codes with a clean **single-orbit canonical logical
basis** (`n = 5|G|`, `k = |G|`, unit anchors, `Lx·Lzᵀ = I`) at a target
distance, for exactly the group you name — there is no group enumeration or
selection in the YAML surface (programmatic band-scanning helpers and an
exhaustive-pool funnel exist as Python APIs in `search/canonical/`). The
randomized, budgeted funnel: full-rank anchor pool → classical sqetch
screen per side → pair → quantum sqetch screen (fail-fast) → save, with
every `SweepParams` knob exposed under `params:`.
Requires GAP + a CUDA GPU. Outputs: per-code directories
(`<tag>__<hash>/` with `Hx/Hz/A_bin/B_bin/Lx/Lz.npy` + a rich `code.json`
recording every distance run with its trial count and `k_sub`), per-group
stats, and `canonical_summary.json`.

### Shipped examples (`search/configs/examples/`)

| File | What it shows | Runtime |
|---|---|---|
| `quickstart_nonabelian.yaml` | S3, shape (1,2), weight-3 entries → saves [[30,6]] codes | ~15 s CPU |
| `quickstart_abelian.yaml` | C10 weight-pattern flavor → saves [[50,10]] d=4 codes | ~15 s CPU |
| `gpu_verify.yaml` | quickstart + sqetch GPU confirmation refuting optimistic screens | ~1 min GPU |
| `canonical_sample.yaml` | canonical mode on D8, target 4 → [[40,8]] codes | ~20 s GPU |
| `template_full.yaml` | every knob of the pipeline schema, annotated | reference |

All examples are seeded (reproducible), verified to run at the stated
budgets, and guarded by tests that fail if the schema drifts.

## Using the library directly

Everything the YAML drives is importable:

```python
from core.group import GroupData
from core.quantum_code import build_quantum_code, compute_k
from core.dist.quantum_bposd import estimate_quantum_distances_bposd

gd = GroupData("SymmetricGroup(3)")            # needs GAP
A = [[(0, 1, 3), (0, 2, 4)]]                   # 1x2 ring matrix: tuples of group indices
B = [[(0, 1, 2), (0, 3, 5)]]                   # same shape as A (the LP default)
qcode = build_quantum_code(A, B, gd)           # -> Hx, Hz, canonical forms, perms
k = compute_k(qcode["Hx"], qcode["Hz"])        # -> 6 (a [[30, 6]] code)
dx, dz = estimate_quantum_distances_bposd(     # logicals auto-derived when omitted
    qcode["Hx"], qcode["Hz"], num_trials=2000, n_workers=1, osd_order=0)
```

`core/dist/` backends: BP+OSD (CPU), sqetch (GPU random-ISD,
QDistRnd-style), classical wrappers of both, and a symplectic estimator for
non-CSS stabilizer codes. The canonical funnel's stages
(`search/canonical/run_group.py`, `sample_search.py`) accept a real
`GroupData` or any GAP-free duck-typed equivalent.

## Tests

```bash
pytest -m "not (gap or gpu or bposd)"   # seconds, no heavy deps
pytest -m gap                            # needs GAP
pytest -m bposd                          # needs ldpc
pytest -m gpu                            # needs CUDA + sqetch
```

The suite mixes implementation tests with black-box contract tests written
against the documented behavior (including brute-force distance
cross-checks on small codes), plus guard tests that load every shipped
example against the current schema.
