# search/configs/

Config dataclasses + YAML plumbing for search runs.

## Files

- `examples/` — ready-to-run YAMLs: two ~15 s quickstarts (non-abelian /
  abelian), a GPU-verify variant, a `mode: canonical` campaign demo
  (`canonical_sample.yaml`), and a fully annotated `template_full.yaml`.
  Guarded by tests so they always load against the current schema.

- `config.py` — the typed config schema (dataclasses) consumed by `phases/` and `runners/`.
- `loader.py` — YAML → validated config objects.
- `paths.py` — canonical output-directory layout helpers.
- `provenance.py` — records run provenance (config hash, code version, timestamps) alongside outputs.
- `yaml_generator.py` — programmatic generation of run configs.

## Schema reference

The authoritative schema is the dataclasses in `config.py`; `loader.py`
validates YAML against them. See `phases/` and `runners/` READMEs for how the
stages consume it.
