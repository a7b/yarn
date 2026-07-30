# tests/

Test suite. Mirrors the source layout: one test file per source module (where
useful).

## Two test types

- **Type 1 (implementation-aware)** — written alongside the code. Tests
  internal edge cases, tricky branches, specific intermediate values.
- **Type 2 (black-box / contract)** — written from the documented contract
  (READMEs + docstrings) without reference to the implementation, so the
  contract itself stays honest.

## Markers

- `fast` — no heavy dependencies.
- `bposd` — requires the `ldpc` package.
- `gap` — requires `gappy` + GAP + QDistRnd.
- `gpu` — requires CUDA + a GPU (sqetch backend).

`pytest -m "not (gap or gpu or bposd)"` runs in seconds and is the default
green bar; the full suite exercises every backend.
