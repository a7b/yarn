"""CLI runners for the search pipeline.

Public entries (run with ``python -m``):

- :mod:`search.runners.search` — run a search from a YAML config.
- :mod:`search.runners.new_config` — generate a YAML skeleton for a group.

Deliberately does NOT import the runner modules here: re-exporting them
would make ``python -m search.runners.search`` emit a confusing runpy
warning (module already in ``sys.modules``).
"""

__all__: list = []
