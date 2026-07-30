# search/runners/

Entry points and orchestration. Single-machine only.

## Files

- `search.py` — top-level CLI dispatcher. Reads YAML and dispatches on its
  `mode`: pipeline configs run the requested `run_stages` in order;
  `mode: canonical` configs run the order-band campaign (`--stages` is
  rejected there, exit code 2).
- `new_config.py` — generate a YAML skeleton tailored to a (group, shape).

## CLI shape

```
python -m search.runners.search <config.yaml> [--stages classical pairing report] [--quiet]
python -m search.runners.new_config "<gap_expr>" <ma> <na> [-o OUT.yaml] [--no-pairing] [--tag TAG]
```
