"""CLI: generate a YAML config skeleton tailored to a group + shape.

::

    python -m search.runners.new_config "SmallGroup(20,3)" 1 2 -o my_run.yaml

Thin wrapper around :func:`search.configs.yaml_generator.generate_search_yaml`
so the generator has a clean ``-m`` entry point (running the module inside
``search.configs`` directly works too, but triggers a harmless runpy warning
because the package re-exports it).
"""

from search.configs.yaml_generator import _cli

if __name__ == "__main__":
    _cli()
