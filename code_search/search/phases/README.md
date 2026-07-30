# search/phases/

Pipeline stages. Each phase reads from the previous phase's output dir and writes to its own.

## Files

- `classical.py` — sample A (and B if non-abelian) → apply filters → estimate classical distance (BP+OSD) → save. One pool per side; structural properties (e.g. `any_block_col_full_rank`, girth) are recorded as flags in each saved JSON. Saves are content-deduped against the existing pool, so reruns continue a search.
- `pairing.py` — load classical pool → enumerate (A, B) pairs (abelian: `B = A*`) → cheap pair filters → build Hx, Hz → BP+OSD screen → optional sqetch GPU confirmation → save quantum JSON.
- `report.py` — aggregate the quantum pool into a markdown summary.

## Phase boundaries

Phases are deliberately decoupled (file-based handoff) so each can run on different hardware (CPU classical → GPU-assisted pairing → CPU report) and resume on failure or reruns.

## Output layout

Under the configured `results_dir`:
```
<results_dir>/
  {group_tag}/{ma}x{na}/
    classical_A/                 # per-side pool (one JSON per code, flags inside)
    classical_B/                 # non-abelian only
    quantum/                     # one JSON per passed code + tried_pairs.json
    manifest.json
    report.md
```
