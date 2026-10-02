# Mitten code hardware layouts (v2)

Hardware-aware layouts for the codes in the paper, one per code, produced with HAL. Each is
the layout with the lowest hardware complexity $C_{hw}$ found so far for that code. This
updates `../scq_hardware_layouts_HAL/` (v1, July) with the 2026-08-03 re-runs, which
improved five of the eight codes. Codes where no re-run beat v1 keep their v1 layout.

- `mitten_<n>_<k>_<d>_layout/` — the full HAL output for that code
- `placements/mitten_<n>_<k>_<d>_placement.npz` — the qubit placement, as plain arrays
- `index.json` — the table below, with the layout parameters of each code
- `build_folder.py` — rebuilds this folder from `hal_runs/` and v1

## Index

| code | [[n,k,d]] | tiers | $C_{hw}$ | v1 $C_{hw}$ | avg coupler len | avg TSVs/edge | max avg face switches | 
|---|---|---|---|---|---|---|---|---|
| `mitten_150_30_10` | [[150,30,10]] | 6 | **2.0211** | 2.0211 | 6.491 | 3.583 | 4.119 | 
| `mitten_200_40_12` | [[200,40,12]] | 6 | **2.0890** | 2.0890 | 8.384 | 3.766 | 4.121 | 
| `mitten_300_60_14` | [[300,60,14]] | 7 | **2.3256** | 2.3685 | 9.876 | 4.843 | 4.808 | 
| `mitten_500_100_16` | [[500,100,16]] | 9 | **2.5963** | 2.8048 | 12.172 | 5.865 | 4.755 |
| `mitten_540_108_18` | [[540,108,18]] | 10 | **2.7154** | 2.7154 | 11.952 | 6.434 | 5.000 | 
| `mitten_630_126_20` | [[630,126,20]] | 11 | **2.9827** | 3.3743 | 15.580 | 7.083 | 5.799 | 
| `mitten_780_156_22` | [[780,156,22]] | 12 | **3.0794** | 3.4205 | 15.412 | 7.920 | 5.305 | 
| `mitten_975_195_24` | [[975,195,24]] | 14 | **3.3972** | 3.9465 | 18.905 | 8.902 | 5.528 | 

For comparison, HAL reports $C_{hw}$ = 2.12 for the gross code [[144,12,12]] and 2.24 for
[[288,12,18]].

$C_{hw}$ is the sum of four metrics — tier count, average coupler length, max average face
switches, average TSVs per edge — each normalized against the reference values in that
layout's `settings.json`. Lower is better.  These numbers are copied
from each layout's `benchmark.csv`.

## Placements

Each `.npz` gives the position of every node of the code's Tanner graph. Reading them needs
only `numpy`:

```python
import json, numpy as np
d = np.load("placements/mitten_300_60_14_placement.npz")
meta = json.loads(str(d["meta"]))
xy = np.stack([d["x_placed"], d["y_placed"]], axis=1)[d["is_data"]]   # data-qubit coordinates
```

Eight arrays, one entry per node, all in the same order (by `node_index`, then `is_data`):

| array | meaning |
|---|---|
| `node_index` | row index in the parity-check matrix `vstack((Hx, Hz))` — a qubit index if `is_data`, else a check index (X-checks first, then Z-checks) |
| `is_data` | `True` = data qubit, `False` = check |
| `block` | which of the 9 blocks the node belongs to: `d1`–`d5` for data, `x0`/`x1` and `z0`/`z1` for checks |
| `within` | index inside that block, `0..|G|-1` |
| `x_placed`, `y_placed` | position in the compact frame |
| `x_routed`, `y_routed` | position in the routed frame |

**Compact frame.** The layout as designed, and the one to use for chip geometry or qubit
adjacency. The chip is a `grid_cols` x `grid_rows` grid of 3x3 clusters, `cluster_pitch`
cells apart. Each cluster holds one node from each of the 9 blocks.
- v1 layouts: every cluster belongs to one group element. Data blocks `d1`–`d4` sit on
  the corners, `d5` in the center, X-checks top and bottom, Z-checks left and right.
- Aug 3 layouts: each cluster is one layer-0 component, so its nodes can have different
  `within` indices.
- 300: each cluster is also re-oriented, so the block-to-slot assignment varies between
  clusters. Its 60 clusters fill an 8x8 grid with 4 empty cells (`meta.layout.empty_cells`).

Every check is one cell away from the data qubits it acts on in its own cluster.

**Routed frame.** The same layout after HAL stretches it over the 200x200 routing grid to
open channels for the couplers. All reported metrics are measured here, and the coupler
routes in `<layout>/tiers/metrics_*.csv` use these coordinates. The stretch acts on each
axis independently and never reorders nodes, so the two frames describe the same layout at
different spacing. `build_folder.py` checks this for every new layout.

`meta` is a JSON string holding the code parameters (`code_name`, `n`, `k`, `distance`), the
metrics from the index above, `routing_grid_size`, and a `layout` block with `block_size`
(= `|G|`), `grid_cols`, `grid_rows`, `cluster_pitch`, and the chip footprint `chip_w` x
`chip_h`. New layouts also record `source_run_dir`.

## Layout directory contents

- `benchmark.csv` — the four metrics, their normalized values, and `hardware_complexity`
- `settings.json` — the HAL settings that produced the layout
- `perf.json` — `place_time` / `benchmark_time` / `total_time`
- `tanner` — pickled `networkx` Tanner graph, node attribute `pos` (needs `networkx` to
  read; the `.npz` files above do not)
- `tiers/` — per-tier `metrics_*.csv` with the full route of every coupler, plus grid pickles
- `grid_view/` — `tier_grid_*.png` and a combined `tier_grids.pdf`
- `tier_interactive_*.png` / `.svg` — rendered tier images
- `layers/` — empty; these runs did not inject layer tiers
