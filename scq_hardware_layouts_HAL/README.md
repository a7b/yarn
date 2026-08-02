# Mitten code hardware layouts

Hardware-aware layouts for the codes in the paper, one per code, produced with HAL. Each is
the layout with the lowest hardware complexity $C_{hw}$ we found for that code.

- `mitten_<n>_<k>_<d>_layout/` — the full HAL output for that code
- `placements/mitten_<n>_<k>_<d>_placement.npz` — the qubit placement, as plain arrays

## Index

| code | [[n,k,d]] | tiers | $C_{hw}$ | avg coupler len | avg TSVs/edge | max avg face switches |
|---|---|---|---|---|---|---|
| `mitten_150_30_10` | [[150,30,10]] | 6 | **2.0211** | 6.491 | 3.583 | 4.119 |
| `mitten_200_40_12` | [[200,40,12]] | 6 | **2.0890** | 8.384 | 3.766 | 4.121 |
| `mitten_300_60_14` | [[300,60,14]] | 8 | **2.3685** | 6.431 | 5.749 | 4.817 |
| `mitten_500_100_16` | [[500,100,16]] | 10 | **2.8048** | 13.192 | 7.113 | 4.974 |
| `mitten_540_108_18` | [[540,108,18]] | 10 | **2.7154** | 11.952 | 6.434 | 5.000 |
| `mitten_630_126_20` | [[630,126,20]] | 14 | **3.3743** | 17.623 | 8.962 | 5.652 |
| `mitten_780_156_22` | [[780,156,22]] | 14 | **3.4205** | 16.854 | 9.608 | 5.870 |
| `mitten_975_195_24` | [[975,195,24]] | 17 | **3.9465** | 21.489 | 12.244 | 5.712 |

$C_{hw}$ is the sum of four metrics — tier count, average coupler length, max average face
switches, average TSVs per edge — each normalized against the reference values in that
layout's `settings.json`. Lower is better. These numbers are copied from each layout's
`benchmark.csv`.

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
adjacency. Each of the `|G|` group elements owns a 3x3 cluster holding one node from each
block: data blocks `d1`–`d4` on the corners, `d5` in the center, X-checks top and bottom,
Z-checks left and right. Clusters tile a `grid_cols` x `grid_rows` grid with origins
`cluster_pitch` cells apart. Every check is one cell away from the data qubits it acts on.

**Routed frame.** The same layout after HAL stretches it over the 200x200 routing grid to
open channels for the couplers. All reported metrics are measured here, and the coupler
routes in `<layout>/tiers/metrics_*.csv` use these coordinates. The stretch acts on each
axis independently and never reorders nodes, so the two frames describe the same layout at
different spacing — neighbours stay neighbours.

`meta` is a JSON string holding the code parameters (`code_name`, `n`, `k`, `distance`), the
metrics from the index above, `routing_grid_size`, and a `layout` block with `block_size`
(= `|G|`), `grid_cols`, `grid_rows`, `cluster_pitch`, and the chip footprint `chip_w` x
`chip_h`. The layout values are recovered from the placement itself and checked: exactly
`|G|` clusters, 9 nodes each, on a uniform grid.

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


