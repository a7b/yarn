# processor_codes

Finalized quantum-processor code suite: high-rate (k/n = 1/5) lifted-product
CSS codes, each shipped as explicit check matrices with a paired logical
basis, plus — for the `mitten` family — the logical-measurement gadgets
(X/Z/Y and joint XX/ZZ) and the full-extractor augmentation.

## Layout

```
processor_codes/
├── mitten/                 # 8 codes, d = 10 … 24, each with gadgets/
│   └── [[n,k,d]]/
│       ├── Hx.npy  Hz.npy  # X- and Z-type parity checks (0/1 matrices)
│       ├── Lx.npy  Lz.npy  # paired logical bases: Lx · Lzᵀ = I_k
│       ├── hook_free_SE_cycle_schedule.json   # gate-by-gate SE-cycle ordering
│       └── gadgets/
│           ├── X_seed.npz  Z_seed.npz   # single-logical X / Z measurement
│           ├── XX.npz      ZZ.npz       # joint two-logical measurements
│           ├── Y.npz                    # Y-logical measurement (non-CSS merge)
│           └── full_extractor.npz       # extractor-augmented stabilizer spec
├── structured_mitten/      # 6 codes, matrices only (+ SE-cycle schedule for the 5 with movies)
└── abelian_poly_LP/        # 1 code, matrices only
```

Codes:

| Family | Codes |
|---|---|
| `mitten` | [[150,30,10]], [[200,40,12]], [[300,60,14]], [[500,100,16]], [[540,108,18]], [[630,126,20]], [[780,156,22]], [[975,195,24]] |
| `structured_mitten` | [[300,60,9]], [[330,66,12]], [[600,120,14]], [[600,120,16]], [[840,168,18]], [[1200,240,20]] |
| `abelian_poly_LP` | [[560,112,14]] |

## Code files

- `Hx.npy`, `Hz.npy` — binary parity-check matrices; `Hx · Hzᵀ = 0 (mod 2)`.
- `Lx.npy`, `Lz.npy` — `k × n` logical bases with rows in `ker(Hz)` /
  `ker(Hx)` respectively and `Lx · Lzᵀ = I_k (mod 2)`, so row *i* of `Lx`
  and row *i* of `Lz` are the X and Z operators of the same logical qubit.

```python
import numpy as np
d = "processor_codes/mitten/[[150,30,10]]/"
Hx, Hz = np.load(d + "Hx.npy"), np.load(d + "Hz.npy")
Lx, Lz = np.load(d + "Lx.npy"), np.load(d + "Lz.npy")
```

## Gadget files (`mitten` only)

All gadget codes act on the original `n` data qubits (always the FIRST `n`
columns) plus added ancilla qubits and checks. The files carry the plain
matrices only; everything derivable (check weights, embeddings) or
documentable (conventions) lives here instead.

Conventions:
- Every single-logical gadget (`X_seed`, `Z_seed`, `Y`) measures **logical 0**
  — row 0 of `Lx.npy` / `Lz.npy`.
- The full extractor's mixed bridge pairs **X of logical 0 with Z of
  logical 1** (all codes).
- All CSS gadget files are in the native frame, and both original check
  matrices appear verbatim in the first `n` columns of the gadget's
  `Hx`/`Hz`. The gadget's **same-type** checks (X-type for `X_seed`/`XX`,
  Z-type for `Z_seed`/`ZZ`) are exactly zero-padded on the ancilla
  columns; the **opposite-type** original checks are extended onto the
  ancillas (the surgery deformation).

Files:

- **`X_seed.npz` / `Z_seed.npz`** — the merged CSS code that measures
  logical-0's X (resp. Z). Keys: `Hx`, `Hz`.
- **`XX.npz` / `ZZ.npz`** — merged codes measuring a joint product of two
  logicals of the same code. Keys: `Hx`, `Hz`.
- **`Y.npz`** — measures logical-0's Y via a single merged **non-CSS**
  stabilizer code. `HX` and `HZ` have identical shape: row *i* of the pair
  is the X-part / Z-part of stabilizer *i* (over data + ancilla qubits).
  `readout_rows` lists the stabilizer rows whose measurement outcomes
  multiply to the logical-Y result, with `outcome_sign` fixing the sign.
- **`full_extractor.npz`** — the extractor-augmented stabilizer
  specification: single key `S`, symplectic `[X|Z]` convention (columns =
  2 × total qubits).

## Hook-free SE-cycle schedule (`hook_free_SE_cycle_schedule.json`)

Every code with a movie in `SE_cycle_movies/` (the 8 `mitten` codes and 5 of the
`structured_mitten` codes) carries this file: the gate-by-gate order in which its
check qubits are entangled with its data qubits during one syndrome-extraction
(SE) cycle. It is the hook-free ordering the movies follow and the paper's SE-cycle
times were computed for. All indices refer to the `Hx.npy`/`Hz.npy` in the same
folder.

**Structure.** With |G| the group order, the n = 5|G| data qubits form five blocks
D1…D5 (columns `b·|G| … (b+1)·|G|-1` for block b = 0…4); the X checks form two
blocks X0, X1 (rows `0…|G|-1` and `|G|…2|G|-1` of `Hx`), likewise Z0, Z1 in `Hz`.
A **move** entangles one whole check block with one data block through one group
element (|G| CZ gates at once); each check block makes 9 moves per cycle, three
to each of its three data blocks. A **layer** is one global entangling pulse and
carries the moves of the check blocks gating at that moment (one or two). An SE
cycle consists of an **X-check round** and a **Z-check round** of 12 layers each;
the order of those layers is what this file pins down.

**Group-element rule** (same convention as the `L(·)`/`R(·)` cells of the Hx/Hz
panels in the movies and the paper, `L(g): h ↦ g·h`, `R(g): h ↦ h·g⁻¹`): in a move
with element `g`, data qubit `q` of the target block is gated by check `g·q`
(action L) or by check `q·g⁻¹` (action R). Products come from the multiplication
table in the file, so no external software is needed; the element label equals the
panel entry in the movie. The explicit `gates` list is the ground truth.

**Fields.**

| field | meaning |
|---|---|
| `code` | `n, k, d`, `group`, `gap_group_expression` (defines the element numbering via GAP's `Elements(G)`), `group_order`, `movie_tag`, `lp_convention` (how Hx/Hz are built from A and B), paths of the two `movies` |
| `qubit_indexing` | inclusive index ranges of D1…D5 (columns) and X0/X1, Z0/Z1 (rows) |
| `group` | `order`; `multiplication_table[a][b]` = number of `a·b`; `element_labels` (number → label, e.g. `x^2·r^2`; `<factor>_<i>` when a factor has no standard generator names). The table is GAP's: element `a` is `Elements(G)[a+1]` for `G := <gap_group_expression>`, and `multiplication_table[a][b] = Position(Elements(G), Elements(G)[a+1]*Elements(G)[b+1]) - 1`; every table was checked against a fresh GAP computation |
| `ring_elements` | the lifted-product data: `a0`, `a1` (the two entries of A) and `b0`, `b1` (the two entries of B), each a list of element numbers |
| `X_layers`, `Z_layers` | the X-check round and the Z-check round. `description` states the conventions. `order` is the schedule at a glance: one line per layer in execution order, e.g. `layer 0: X0→D1 L(x^2·r^2), X1→D2 L(x^2·r^2)` (check block → data block, and the L/R group element as printed in the movie's Hx/Hz panel). `layers` gives the same layers in full: each has `layer` (index) and `moves`, each move `check_block`, `data_block`, and `cz_gates` = the \|G\| pairs `[i, j]` = `[check row of Hx.npy or Hz.npy, data column]`, i.e. one CZ between check qubit `i` and data qubit `j` (so `[0, 19]` in an X layer means X check 0 is entangled with data qubit 19, and `Hx[0, 19] = 1`) |
