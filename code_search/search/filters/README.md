# search/filters/

Rejection criteria for sampled matrices. Each file groups filters by *what they bound*.

## Planned files

- `girth.py`
  - `base_girth_bound(weight_matrix)` — weight-matrix-only bound; no GAP. **Abelian only** (the relator identities behind the bound need commutativity).
  - `girth_tanner(H_bin)` — binary Tanner girth.
- `distance_bounds.py`
  - `entry_order_bound(A, gd)` — ma=1, weight-2 entries; `ord(g₁⁻¹g₂)` upper-bounds `d(A_bin)`.
  - `distance_upper_bound(A, gd)` — ma=1, abelianization bound.
  - `ring_distance_bound(A, gd)` — abelian only; permanent-based.
- `rank.py`
  - `column_space_coverage(A_bin)` — rank(A_bin) == n.
  - `any_block_col_full_rank(A_bin)` — at least one block-col is invertible (gives a structured canonical basis).
- `config.py` — `ClassicalFilterConfig` dataclass and `passes_classical_filters(A, gd, cfg)` dispatcher.

## Order

Filters apply cheapest-first inside `passes_classical_filters`:
1. `base_girth_bound` (weight matrix only; abelian only)
2. `entry_order_bound` (element-order table)
3. `distance_upper_bound` (abelianization; non-abelian only)
4. `ring_distance_bound` (J! permanents — most expensive; abelian only)

## Full-rank gate

`rank.py::any_block_col_full_rank` is the gate between the two classical pools. The dual-pool model means we don't reject on rank during the cheap filter pass — we save into both pools.
