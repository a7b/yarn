# logical_basis/

Paired X/Z logical bases: `Lx`, `Lz` with `Lx_i · Lz_j = δ_{ij}` (mod 2), rows in the proper kernels.

## File

Everything lives in `logical_basis.py`:

- `find_logical_basis(Hx, Hz, A_bin, BT_bin, n, shape_a, shape_b, save_dir=None) -> dict` — the canonical G-orbit-preserving basis (unit anchors; the construction the canonical search saves).
- `find_logical_noncanonical_RREF(Hx, Hz)` — fast paired basis via RREF; what `core/dist` auto-derives when no basis is passed.
- `find_logical_basis_pivot_aligned(Hx, Hz)` — pivot-aligned variant.
- `reorder_by_pivot(Lx, Lz, ...)` — reorder a paired basis while preserving the pairing.

## Prerequisite (canonical path)

`A_bin` must have at least one full-rank block-column (`any_block_col_full_rank`); same on `B_bin`. Saved classical-pool JSONs record this as a structural flag, and the pairing filters can require it.

## Output

With `save_dir` set, `find_logical_basis` saves `Lx.npy`, `Lz.npy` plus a JSON describing the pairing and the chosen full-rank block columns.
