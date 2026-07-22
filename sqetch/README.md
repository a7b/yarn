# sqetch

## What is sqetch?

sqetch (sketched quantum information-set decoder) estimates the minimum
distance of a CSS quantum code on the GPU by running many randomized
Prange-style information-set decoder (ISD) trials against a
low-dimensional sketch of the logical subspace. Each trial draws a
random column permutation, takes a `k_sub`-dimensional random sketch of
the null space of `H_check`, reduces the sketch to row echelon form,
and keeps the lightest non-trivial logical-coset row it sees. The
minimum over many trials is an upper bound on the code distance; with
enough trials it converges to the true distance.

## Install

```
pip install sqetch
pip install sqetch[gpu]      # adds the CUDA backend (PyTorch)
```

The base install is enough to import the package and inspect the API;
running on the GPU requires the `gpu` extra (or a manually-installed
PyTorch built with CUDA support).

### Platform notes

`pip install sqetch[gpu]` pulls `torch>=2.1` from PyPI. Whether that
wheel can actually run sqetch's kernel depends on your platform:

* **Linux x86_64**: PyPI's default `torch` wheel is built with CUDA 12.
  Works out of the box on any machine with a CUDA 12 driver. This is the
  primary supported configuration.
* **Windows x86_64**: same -- PyPI's `torch` ships with CUDA support by
  default.
* **macOS** (arm64 or x86_64): PyPI's `torch` is CPU-only; there is no
  CUDA build of PyTorch for macOS. sqetch's kernel **cannot** run here.
  You can `pip install sqetch` to inspect the API, but
  `estimate_distance` will refuse.
* **Linux aarch64** (Jetson, GH200, etc.): PyPI's `torch` is CPU-only.
  Install the CUDA-enabled wheel from NVIDIA's index first, then
  `pip install sqetch` *without* the `[gpu]` extra:

  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  pip install sqetch
  ```

## Quickstart

```python
import numpy as np
from sqetch import estimate_distance

# H_X is your X-type parity-check matrix; L_X is a basis of X-logicals
# (rows in ker(H_Z)).  Use these to estimate d_Z = min weight of a
# non-trivial Z-logical, since Z-logicals live in ker(H_X) modulo
# rowspan(H_Z) and a representative is non-trivial iff some L_X row has
# non-zero inner product with it.
H_check = np.load("Hx.npy").astype(np.uint8)
L_logical = np.load("Lx.npy").astype(np.uint8)

result = estimate_distance(
    H_check, L_logical,
    num_trials=100_000,
    d_target=12,
    k_sub=32,
    seed=42,
)
print(f"d_Z bound: {result.best_weight}")
print(f"trials: {result.trials_run}, {result.raw_iter_per_sec:.0f} iter/s")
```

## What the algorithm does

For each trial:

1. Sample a random permutation pi of the n columns.
2. Build a (k_sub x n) sketch `W_sub` of `ker(H_check)` by drawing
   `k_sub` rows of the null-space basis uniformly at random with
   replacement (one row per sketch slot, no XORs).
3. Row-reduce `W_sub` to reduced row echelon form under the column
   order induced by pi.
4. For each non-zero RREF row, check whether its inner product with any
   row of `L_logical` is non-zero. If yes, the row is a non-trivial
   logical-coset representative; record its Hamming weight.
5. The block returns the minimum weight seen. The host loop tracks the
   grid-wide minimum across all trials and can stop early once a
   candidate lighter than `d_target` appears.



`k_sub` is the sketch dimension used per trial. It is bounded above by
the full null-space dimension `dim ker(H_check)` and by the device's
opt-in shared-memory ceiling; if the requested `k_sub` exceeds the
shared-memory budget for your `n`, `estimate_distance` raises a
`ValueError` reporting the maximum that fits.

## Citation

If you use sqetch, please cite both this package and the QDistRnd
algorithm that it implements on the GPU:

```bibtex
@software{ma_sqetch_2026,
  author  = {Ma, Muzhou},
  title   = {sqetch: GPU random information-set decoder for CSS quantum-code distance estimation},
  year    = {2026},
  version = {0.1.0},
}
```

## License

MIT. See `LICENSE`.
