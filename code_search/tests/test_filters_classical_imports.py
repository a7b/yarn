"""Verify that ``classical/abelian/__init__.py`` and ``classical/non_abelian/__init__.py``
re-export the SAME function objects as the corresponding ``_shared`` modules.

After the shell-file cleanup, the shared filters live ONLY in ``_shared/``;
the abelian/ and non_abelian/ folders are namespace packages that pull
them in via ``__init__.py``. This module just confirms that route works.
"""

import pytest

pytestmark = pytest.mark.fast


SHARED_FILTERS = [
    "any_block_col_full_rank",
    "entry_order_bound",
    "girth_tanner",
]


@pytest.mark.parametrize("filter_name", SHARED_FILTERS)
def test_abelian_namespace_re_exports(filter_name):
    import search.filters.classical.abelian as abelian
    shared = __import__(
        f"search.filters.classical._shared.{filter_name}",
        fromlist=[filter_name],
    )
    assert getattr(abelian, filter_name) is getattr(shared, filter_name)


@pytest.mark.parametrize("filter_name", SHARED_FILTERS)
def test_non_abelian_namespace_re_exports(filter_name):
    import search.filters.classical.non_abelian as non_abelian
    shared = __import__(
        f"search.filters.classical._shared.{filter_name}",
        fromlist=[filter_name],
    )
    assert getattr(non_abelian, filter_name) is getattr(shared, filter_name)


def test_ring_distance_bound_is_abelian_only():
    """`ring_distance_bound` lives only in abelian/; non_abelian/ must not expose it."""
    from search.filters.classical.abelian import ring_distance_bound  # noqa: F401
    import search.filters.classical.non_abelian as nab
    assert not hasattr(nab, "ring_distance_bound")


def test_base_girth_bound_is_abelian_only():
    """`base_girth_bound` lives only in abelian/; non_abelian/ must not expose it."""
    from search.filters.classical.abelian import base_girth_bound  # noqa: F401
    import search.filters.classical.non_abelian as nab
    assert not hasattr(nab, "base_girth_bound")


def test_abelianization_bound_is_non_abelian_only():
    """`abelianization_bound` lives only in non_abelian/; abelian/ must not expose it."""
    from search.filters.classical.non_abelian import abelianization_bound  # noqa: F401
    import search.filters.classical.abelian as ab
    assert not hasattr(ab, "abelianization_bound")


def test_shells_no_longer_exist():
    """The per-filter shell files should be gone — abelian/ and non_abelian/
    must not contain submodule files for any of the shared filters. The
    namespace package via __init__.py is the only path."""
    import importlib
    for name in SHARED_FILTERS:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"search.filters.classical.abelian.{name}")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"search.filters.classical.non_abelian.{name}")


def test_quantum_pairing_namespace_re_exports():
    import search.filters.quantum_pairing.abelian as qa
    import search.filters.quantum_pairing.non_abelian as qna
    from search.filters.quantum_pairing._shared.min_classical_distance import (
        min_classical_distance,
    )
    from search.filters.quantum_pairing._shared.min_classical_girth import (
        min_classical_girth,
    )
    assert qa.min_classical_distance is min_classical_distance
    assert qa.min_classical_girth is min_classical_girth
    assert qna.min_classical_distance is min_classical_distance
    assert qna.min_classical_girth is min_classical_girth
