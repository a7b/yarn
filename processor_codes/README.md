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
│       └── gadgets/
│           ├── X_seed.npz  Z_seed.npz   # single-logical X / Z measurement
│           ├── XX.npz      ZZ.npz       # joint two-logical measurements
│           ├── Y.npz                    # Y-logical measurement (non-CSS merge)
│           └── full_extractor.npz       # extractor-augmented stabilizer spec
├── mitten_structured/      # 6 codes, matrices only
└── abelian_poly_LP/        # 1 code, matrices only
```

Codes:

| Family | Codes |
|---|---|
| `mitten` | [[150,30,10]], [[200,40,12]], [[300,60,14]], [[500,100,16]], [[540,108,18]], [[630,126,20]], [[780,156,22]], [[975,195,24]] |
| `mitten_structured` | [[300,60,9]], [[330,66,12]], [[600,120,14]], [[600,120,16]], [[840,168,18]], [[1200,240,20]] |
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
