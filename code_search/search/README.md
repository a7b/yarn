# search/

Code-search pipeline. Decomposed along three axes (sampling × family × filters) plus orchestration (phases, runners, configs).

## Quickstart

The YAML file is the interface — copy an example, edit a few lines, run
(from the package root, or put the package root on `PYTHONPATH`):

```bash
cp search/configs/examples/quickstart_nonabelian.yaml my_run.yaml
python -m search.runners.search my_run.yaml
```

The quickstart finds [[30, 6]] LP codes over S3 in ~15 s on CPU (GAP + the
`ldpc` package required). Outputs land under
`search_results/<group_tag>/<shape>/{classical_A,classical_B,quantum,report.md}`,
and every saved JSON records the config that produced it. **Rerunning the same
YAML continues the search** — saved pools are reused and tried pairs skipped.

Other examples in `search/configs/examples/`: `quickstart_abelian.yaml`
(two-stage weight-pattern flavor), `gpu_verify.yaml` (adds the sqetch GPU
confirmation pass — the BP+OSD screen alone is deliberately optimistic),
`template_full.yaml` (every knob, annotated), and `canonical_sample.yaml`
(`mode: canonical` — the canonical-basis search for one explicitly named
group, same command).
To generate a skeleton tailored to your group (only the applicable filters,
sane defaults):

```bash
python -m search.runners.new_config "SmallGroup(20,3)" 1 2 -o my_run.yaml
```

## Subfolders

- `sampling/` — *how* we parameterize matrix entries (monomial, polynomial, weight-matrix patterns).
- `families/` — *which fixed family* we sample within (structural priors on the LP construction).
- `filters/` — *what we reject*: girth bounds, distance upper bounds, rank/coverage/containment.
- `phases/` — pipeline stages: classical sample-and-filter → pairing/CSS construction → report.
- `configs/` — YAML templates + a small set of representative example configs.
- `runners/` — entry point: top-level CLI dispatcher (`python -m search.runners.search <config.yaml>`).
- `canonical/` — brute-force campaign for the smallest shape-(1,2) LP code with a clean canonical logical basis (see its README).

## Pipeline shape

```
sampling × family ── (entry distribution) ──┐
                                            ▼
filters (girth / rank / distance bounds) ── classical samples
                                            │
                                            ▼
                    classical_A/  classical_B/   (per-side pools; each JSON
                                            │     carries structural flags,
                                            │     e.g. any_block_col_full_rank)
                                            ▼
                              pair (A, B) — phases/pairing
                          (BP+OSD screen → optional sqetch GPU verify)
                                            │
                                            ▼
                                        quantum/  +  report.md
```

## Settled choices

- One pool per side; structural properties (full-rank block-col, girth,
  containment) are recorded as flags in each saved JSON and used by the
  pairing filters — there is no separate full-rank pool directory.
- Pass condition: an estimate passes when it is `None` (nothing found in
  budget) or `>= d_target`; estimators early-stop on strictly-`<` hits
  (see `core/dist/`).
- No same-type BP+OSD augmentation.
- Reruns continue a search: classical saves are content-deduped against the
  existing pool, `min_quantum_pool_size` counts previously saved codes, and
  tried pairs are skipped.
