"""Tests for ``random_ring_element`` and ``random_ring_matrix`` in
``search/sampling/_shared/``.
"""

import numpy as np
import pytest

from search.sampling._shared.random_ring_element import random_ring_element
from search.sampling._shared.random_ring_matrix import random_ring_matrix

pytestmark = pytest.mark.gap


@pytest.fixture(scope="module")
def gd_c4():
    from core.group import GroupData
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


# ─────────────────────────────────────────────────────────────────
# random_ring_element
# ─────────────────────────────────────────────────────────────────


class TestRandomRingElement:
    def test_weight_zero_returns_empty(self, gd_c4):
        x = random_ring_element(gd_c4, 0, rng=np.random.default_rng(0))
        assert x == ()

    def test_returns_canonical_sorted(self, gd_s3):
        rng = np.random.default_rng(7)
        x = random_ring_element(gd_s3, 3, rng=rng)
        assert x is not None
        assert list(x) == sorted(set(x))
        assert len(x) == 3
        assert all(0 <= g < gd_s3.n for g in x)

    def test_include_identity_false_excludes(self, gd_s3):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = random_ring_element(
                gd_s3, 2, rng=rng, include_identity=False
            )
            assert x is not None
            assert gd_s3.identity not in x

    def test_min_element_order(self, gd_s3):
        from core.group import element_order
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = random_ring_element(
                gd_s3, 2, rng=rng, min_element_order=3,
            )
            assert x is not None
            for g in x:
                assert element_order(g, gd_s3) >= 3

    def test_avoid_same_coset_distinct(self, gd_s3):
        """Selected elements lie in distinct left cosets of [G,G]."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            x = random_ring_element(
                gd_s3, 2, rng=rng, avoid_same_coset=True,
                include_identity=True, max_tries=200,
            )
            assert x is not None
            cosets = {gd_s3.coset_id[g] for g in x}
            assert len(cosets) == len(x)

    def test_avoid_same_coset_raises_on_abelian(self, gd_c4):
        with pytest.raises(ValueError, match="non-abelian"):
            random_ring_element(
                gd_c4, 2, rng=np.random.default_rng(0),
                avoid_same_coset=True,
            )

    def test_invalid_weight_raises(self, gd_c4):
        with pytest.raises(ValueError):
            random_ring_element(gd_c4, -1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            random_ring_element(gd_c4, gd_c4.n + 1, rng=np.random.default_rng(0))

    def test_avoid_same_coset_infeasible_returns_none_fast(self, gd_s3):
        """When eligible set has fewer distinct cosets than ``weight``, must
        return None WITHOUT burning ``max_tries`` (feasibility pre-check)."""
        # For S3, [G,G] = A3 has size 3 → 2 cosets total. Asking for weight=3
        # with avoid_same_coset is infeasible: at most 2 elements can occupy
        # distinct cosets. The pre-check should bail out immediately.
        x = random_ring_element(
            gd_s3, 3, rng=np.random.default_rng(0),
            avoid_same_coset=True, max_tries=10_000_000,
        )
        assert x is None   # would hang without the pre-check

    def test_returns_none_when_budget_exhausted(self, gd_c4):
        """With include_identity=False, min_order=4, only the order-4 gens
        qualify. C4 has 2 such generators; weight=3 from a pool of size 2
        is impossible → None immediately."""
        x = random_ring_element(
            gd_c4, 3, rng=np.random.default_rng(0),
            include_identity=False, min_element_order=4,
        )
        assert x is None

    def test_seed_stable(self, gd_s3):
        a = random_ring_element(gd_s3, 3, rng=np.random.default_rng(42))
        b = random_ring_element(gd_s3, 3, rng=np.random.default_rng(42))
        assert a == b


# ─────────────────────────────────────────────────────────────────
# random_ring_matrix
# ─────────────────────────────────────────────────────────────────


class TestRandomRingMatrix:
    def test_shape_matches_weight_matrix(self, gd_s3):
        W = np.array([[2, 1], [3, 2]])
        A = random_ring_matrix(gd_s3, W, rng=np.random.default_rng(0))
        assert A is not None
        assert len(A) == 2
        assert len(A[0]) == 2

    def test_per_entry_weights(self, gd_s3):
        """Multiset of per-entry weights matches the weight matrix
        (positions may be permuted by canonicalize=True)."""
        W = np.array([[2, 1, 3]])
        A = random_ring_matrix(gd_s3, W, rng=np.random.default_rng(7))
        assert A is not None
        actual = sorted(len(A[0][j]) for j in range(3))
        expected = sorted(W[0].tolist())
        assert actual == expected

    def test_per_entry_weights_no_canon(self, gd_s3):
        """canonicalize=False preserves positions exactly."""
        W = np.array([[2, 1, 3]])
        A = random_ring_matrix(
            gd_s3, W, rng=np.random.default_rng(7), canonicalize=False,
        )
        assert A is not None
        for j in range(3):
            assert len(A[0][j]) == int(W[0, j])

    def test_zero_weight_gives_empty_entry(self, gd_s3):
        W = np.array([[0, 2]])
        A = random_ring_matrix(gd_s3, W, rng=np.random.default_rng(0))
        assert A is not None
        assert A[0][0] == ()
        assert len(A[0][1]) == 2

    def test_per_entry_override(self, gd_s3):
        """Matrix default forbids identity; entry (0, 0) overrides to allow."""
        W = np.array([[2, 2]])
        A = random_ring_matrix(
            gd_s3, W, rng=np.random.default_rng(0),
            include_identity=False,
            overrides={(0, 1): {"include_identity": True}},
        )
        assert A is not None
        # (0, 0): matrix default applies → no identity.
        assert gd_s3.identity not in A[0][0]
        # (0, 1): override allows identity but doesn't guarantee it.
        # Just verify no exception.

    def test_unknown_override_knob_raises(self, gd_s3):
        with pytest.raises(ValueError, match="unknown knob"):
            random_ring_matrix(
                gd_s3, np.array([[2]]), rng=np.random.default_rng(0),
                overrides={(0, 0): {"not_a_knob": True}},
            )

    def test_out_of_bounds_override_key_raises(self, gd_s3):
        """overrides={(5, 5): ...} on a 1×2 matrix must NOT be silently dropped."""
        with pytest.raises(ValueError, match="out of bounds"):
            random_ring_matrix(
                gd_s3, np.array([[2, 2]]), rng=np.random.default_rng(0),
                overrides={(5, 5): {"include_identity": False}},
            )

    def test_returns_none_when_entry_fails(self, gd_c4):
        # Force impossible per-entry: weight 3 from pool of 2 generators.
        W = np.array([[3]])
        A = random_ring_matrix(
            gd_c4, W, rng=np.random.default_rng(0),
            include_identity=False, min_element_order=4,
        )
        assert A is None

    def test_invalid_weight_matrix_shape(self, gd_c4):
        with pytest.raises(ValueError, match="2D"):
            random_ring_matrix(
                gd_c4, np.array([1, 2, 3]), rng=np.random.default_rng(0),
            )

    def test_canonicalize_default_true_is_idempotent(self, gd_c4):
        """Default canonicalize=True ⇒ sampled matrix is in canonical form,
        and a second canonical_form_A call returns identity perm."""
        from core.classical_code import canonical_form_A
        W = np.array([[3, 3]])
        rng = np.random.default_rng(0)
        for _ in range(10):
            A = random_ring_matrix(gd_c4, W, rng=rng)
            if A is None:
                continue
            A_can, _, perm, _ = canonical_form_A(A, gd_c4)
            assert A_can == A
            assert perm == list(range(len(A[0])))

    def test_canonicalize_false_preserves_raw_sample(self, gd_c4):
        """canonicalize=False returns the raw sampled matrix (entries sorted
        only by random_ring_element, no block-col permutation)."""
        W = np.array([[3, 3]])
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        A_raw = random_ring_matrix(gd_c4, W, rng=rng_a, canonicalize=False)
        A_canon = random_ring_matrix(gd_c4, W, rng=rng_b, canonicalize=True)
        # Same RNG path → same raw sample. The canon version may have
        # block-cols permuted.
        if A_raw is not None and A_canon is not None:
            # Both contain the same multiset of entries.
            assert sorted(A_raw[0]) == sorted(A_canon[0])
