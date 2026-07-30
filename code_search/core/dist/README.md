# core/dist/

Distance estimators. One file per backend (no unified dispatcher). Each module exposes its own typed entry-point; callers pick explicitly.

## Implemented

| File | Function | Notes |
|------|----------|-------|
| `quantum_bposd.py` | `estimate_quantum_distances_bposd(Hx, Hz, Lx=None, Lz=None, ...)` | CPU BP+OSD; `n_workers=1` runs inline, `>1` uses a process pool |
| `quantum_sqetch.py` | `estimate_quantum_distances_sqetch(Hx, Hz, Lx=None, Lz=None, ...)` | GPU random-ISD (QDistRnd-style); `return_logical=True` also returns witness codewords |
| `classical.py` | `estimate_classical_distance_bposd(H)` / `estimate_classical_distance_sqetch(H)` | classical `d(ker H)` via the two quantum backends (`estimate_classical_distance` aliases the bposd one) |
| `stab_symplectic.py` | `estimate_stab_distance(HX, HZ, ...)`, `stab_k`, `centralizer_basis`, `pure_sector_css`, `estimate_coset_min_weight` | non-CSS stabilizer codes in symplectic (X\|Z) form; pure-numpy |

Both quantum backends share the cross-type-augmentation, strict-`<` early-stop, and `None`-per-direction conventions described below. `Lx`/`Lz` are optional: when omitted, a consistent paired RREF basis is derived automatically (`logical_basis.find_logical_noncanonical_RREF`); pass explicit bases when you need a specific one.

## Planned (not yet built)

| File | Function | Notes |
|------|----------|-------|
| `quantum_gap.py` | `estimate_quantum_distances_gap(Hx, Hz)` | GAP QDistRandCSS verification backend |

## Conventions

- **Strict `<` early-stop**: a codeword of weight `< d_target` triggers early termination; weight `== d_target` does NOT (boundary case is PASS).
- **`None` per direction**: if all trials exhaust without finding a logical in that direction, the function returns `None` for that entry (not `0`). Caller decides what to do.
- **Cross-type augmentation**: for `dx` (X-logicals), augment Hz with Lz; for `dz` (Z-logicals), augment Hx with Lx. Same-type augmentation finds light stabilizers and falsely reports them.
- **Screens vs verification**: BP+OSD and sqetch are randomized upper-bound estimators; small budgets are deliberately optimistic screens, large budgets refute them. Exact/certified distances are outside this package today (the GAP backend above is the planned verification path).
