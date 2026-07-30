# core/

Math primitives. No domain assumptions about LP codes, search, or hardware — these are the building blocks everything else uses.

## Files

- `f2.py` — GF(2) linear algebra (Gaussian elimination, rank, RREF, null space, solve).
- `f2_fast.py` — bit-packed GF(2) kernels for hot paths (`screen_basis`, packed rank/nullspace).
- `group.py` — `GroupData` (GAP-backed), ring arithmetic over F₂[G], dagger operation, left/right regular representations, direct-product decomposition.
- `classical_code.py` — A / B level only.
  - **Forward**: `build_A_bin`, `build_B_bin`.
  - **Inverse**: `A_from_A_bin(A_bin, gd, shape_a)`, `B_from_B_bin(B_bin, gd, shape_b)`. Works because `L`, `R` are faithful: the column at index `gd.identity` of each n×n block uniquely encodes the underlying ring element.
  - **Canonical block-col form**: `canonical_form_A`, `canonical_form_B`, `canonical_form_A_bin`, `canonical_form_B_bin`. Permute block-cols so an invertible block-col subset lands at the LAST positions (required by `find_logical_basis` / downstream measurement machinery).
- `quantum_code.py` — `build_Hx` / `build_Hz` / `build_quantum_code`, `compute_k`, `check_css`, `quantum_check_weights`, and the inverse maps `A_bin_B_bin_from_Hx_Hz` / `AB_from_Hx_Hz`.
- `dist/` — distance estimators (see `dist/README.md`).

## Notation

Key conventions:
- F₂[G] elements stored as sorted tuples of 0-based group indices.
- L[g] (left-regular rep) and R[g] (right-regular, `R[g]·e_h = e_{h·g⁻¹}`).
- L, R commute; `R[x]^T = R[x†]`.
