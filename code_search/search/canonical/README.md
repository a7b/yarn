# search/canonical/

Brute-force search for the **smallest shape-(1,2) LP code with a clean
canonical logical basis** at a target distance, over non-abelian groups.

## Run it from YAML

```bash
python -m search.runners.search search/configs/examples/canonical_sample.yaml
```

A `mode: canonical` config (see the annotated example) drives `driver.py`:
it runs the randomized, budgeted funnel (`run_group_sweep`) for the ONE
group named in the YAML — one config, one group; to search several groups,
write several configs. The YAML `params:` section maps 1:1 onto
`SweepParams` below. The exhaustive-pool funnel (`run_group`/`ScreenParams`)
and the group-family scanning helpers (`groups.py`) remain available as
Python APIs but are not part of the YAML surface.

`n = 5|G|`, `k = |G|` (single orbit) → a canonical G-orbit basis with
`Lx·Lzᵀ = I`. Smallest code ⇒ smallest group order, so we sweep orders
**ascending and stop at the first order that yields a verified code**.

## The funnel (per group)

```
brute full-rank pool            local CPU   all weight-3, identity in support,
                                            unit (rank L[a1]=|G|); one pool serves
                                            A (left_rep) and B (right_rep)
  → classical sqetch screen      GPU        A=[[a0,a1]], B=[[b0,b1]] independently;
                                            a0,b0 brute over ALL weight-3; keep
                                            top-N with d(ker A_bin),d(ker B_bin) ≥ T
                                            (these UPPER-bound the quantum distance).
                                            5M trials, k_sub = ker dim (=|G|, the max),
                                            d_target=T early-stop, no bposd.
  → pair survivors               GPU        build Hx/Hz directly, one RREF → (Lx,Lz)+k,
                                            keep single-orbit (k=|G|).
  → quantum sqetch SCREEN         GPU        d_target=T; sqetch first. If it finds a
                                            logical < T, SKIP bposd (fail fast).
  → bposd CONFIRM                 CPU        only on sqetch-passers, osd_order≥4.
  → quantum CERTIFY               GPU        heavy multi-seed sqetch (5M × seeds,
                                            k_sub=192). Reported dx/dz = MIN over
                                            sqetch screen, bposd, every certify seed.
  → save                          local      canonical Hx/Hz/A_bin/B_bin/Lx/Lz.npy +
                                            rich code.json (k_sub recorded on EVERY
                                            sqetch run). Stats saved even on 0 passes.
```

Distance is **basis-independent**, so the screen uses sqetch's fast auto-RREF
basis; the canonical G-orbit basis (`find_logical_basis`) is built only for
passers.

Note: the funnel above (with its bposd CONFIRM and multi-seed CERTIFY
stages) is the exhaustive `run_group` API. The YAML-driven randomized
funnel (`run_group_sweep`) deliberately has neither — its single
high-trial, max-`k_sub` sqetch screen is trusted. Recorded `k_sub` values
in saved JSONs are the *requested* sketch size; the GPU kernel clamps to
each direction's kernel dimension internally.

## Modules

- `groups.py` — non-abelian `SmallGroup(o,i)` enumeration over an order band
  (ascending; giant orders 96/144/160 deferred). GAP-backed (local only).
- `build.py` — `build_canonical_code`, `canonical_logical_basis` (raises
  `NotSingleOrbit` if `k≠|G|`), classical lifts `a_bin_for`/`b_bin_for`.
- `save.py` — `save_passed_code`: per-code dir with the rich JSON schema.
- `run_group.py` — `ScreenParams`, `screen_classical_side`, `pair_and_verify`,
  `certify_quantum`, `run_group` (the unit of work; accepts a real `GroupData`
  or any GAP-free duck-typed equivalent).
- `sample_search.py` — `SweepParams`, `run_group_sweep` (randomized funnel
  with draw/pair budgets — the strategy for sweeping general groups).
- `driver.py` — the single-group driver behind `mode: canonical` YAMLs.

## Run

One group, real GAP:
```python
from core.group import GroupData
from search.canonical.groups import GroupSpec
from search.canonical.run_group import run_group, ScreenParams
g = GroupSpec(84, 11)
run_group(g, ScreenParams(target=16, max_A_pool=200, max_B_pool=200), "out/",
          gd=GroupData(g.gap_expr))
```

## Conventions (verified)

- `L[g]·e_h = e_{g·h}`, `R[g]·e_h = e_{h·g⁻¹}` (the inverse is required for R to
  be a homomorphism and to commute with L → `Hx·Hzᵀ=0`).
- `Hx = [L[A]⊗I | I⊗R[B†]]`, `Hz = [I⊗R[B] | L[A†]⊗I]` (daggers give CSS).
- Classical distance is an **upper bound** on quantum distance — a necessary but
  loose pre-filter. Most classical-d=T pairs have quantum distance ≪ T; the
  search hunts the rare pairs that reach T.
- sqetch (GPU random-ISD) and bposd (CPU BP+OSD) reported **identical** quantum
  dx/dz on every cross-checked order-84 pair.
