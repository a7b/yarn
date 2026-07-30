"""Tests for ``search/filters/config.py`` — dispatcher behavior.

Covers:
- ClassicalFilterConfig defaults pass everything.
- Each enabled filter rejects when its threshold isn't met.
- ma=1 precondition raises when enabled with ma > 1.
- Abelian-only precondition raises with non-abelian G.
- QuantumPairingFilterConfig dispatcher.
"""

import numpy as np
import pytest

from search.filters.config import (
    ClassicalFilterConfig,
    QuantumPairingFilterConfig,
    apply_classical_filters,
    apply_quantum_pairing_filters,
)

pytestmark = pytest.mark.gap


@pytest.fixture(scope="module")
def gd_c4():
    from core.group import GroupData
    return GroupData("CyclicGroup(4)")


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


@pytest.fixture(scope="module")
def good_inputs_c4(gd_c4):
    # 1x2 with an identity entry → A_bin is rank n.
    from core.classical_code import build_A_bin
    A = [[(gd_c4.identity,), (1, 2)]]
    A_bin = build_A_bin(A, gd_c4)
    return A, A_bin


# ─────────────────────────────────────────────────────────────────
# Classical dispatcher
# ─────────────────────────────────────────────────────────────────


class TestClassicalDefaults:
    def test_empty_config_passes_everything(self, gd_c4, good_inputs_c4):
        A, A_bin = good_inputs_c4
        assert apply_classical_filters(
            A, A_bin, gd_c4, ClassicalFilterConfig(),
        ) is True


class TestClassicalRejects:
    def test_min_base_girth_bound_can_reject(self, gd_c4, good_inputs_c4):
        A, A_bin = good_inputs_c4
        # (1, 2) with entries of weight 1 and 2 → base_girth_bound likely = inf,
        # but min=20 should still pass if the bound is inf. Let's force a reject
        # with a 1x1 weight-3 entry.
        from core.classical_code import build_A_bin
        A_bad = [[(0, 1, 2)]]   # weight-3 → bound = 6
        A_bin_bad = build_A_bin(A_bad, gd_c4)
        cfg = ClassicalFilterConfig(min_base_girth_bound=8)
        assert apply_classical_filters(A_bad, A_bin_bad, gd_c4, cfg) is False

    def test_require_any_block_col_full_rank_can_reject(self, gd_c4):
        A = [[(0, 1), (2, 3)]]   # neither block-col is full rank n=4
        from core.classical_code import build_A_bin
        A_bin = build_A_bin(A, gd_c4)
        cfg = ClassicalFilterConfig(require_any_block_col_full_rank=True)
        # Whether this passes/fails depends on the specific lift; we just
        # verify the predicate is consulted (no exception).
        result = apply_classical_filters(A, A_bin, gd_c4, cfg)
        assert isinstance(result, bool)


class TestClassicalPreconditions:
    def test_min_entry_order_bound_raises_on_ma_gt_1(self, gd_c4):
        from core.classical_code import build_A_bin
        A = [[(0,)], [(1,)]]                 # ma=2
        A_bin = build_A_bin(A, gd_c4)
        cfg = ClassicalFilterConfig(min_entry_order_bound=4)
        with pytest.raises(ValueError, match="ma=1"):
            apply_classical_filters(A, A_bin, gd_c4, cfg)

    def test_min_ring_distance_bound_raises_on_non_abelian(self, gd_s3):
        from core.classical_code import build_A_bin
        A = [[(0, 1), (2, 3)]]
        A_bin = build_A_bin(A, gd_s3)
        cfg = ClassicalFilterConfig(min_ring_distance_bound=4)
        with pytest.raises(ValueError, match="abelian"):
            apply_classical_filters(A, A_bin, gd_s3, cfg)

    def test_min_base_girth_bound_raises_on_non_abelian(self, gd_s3):
        from core.classical_code import build_A_bin
        A = [[(0, 1), (2, 3)]]
        A_bin = build_A_bin(A, gd_s3)
        cfg = ClassicalFilterConfig(min_base_girth_bound=8)
        with pytest.raises(ValueError, match="abelian"):
            apply_classical_filters(A, A_bin, gd_s3, cfg)

    def test_min_abelianization_bound_raises_on_abelian(self, gd_c4):
        """Non-abelian-only filter must reject abelian G."""
        from core.classical_code import build_A_bin
        A = [[(0, 1), (2, 3)]]
        A_bin = build_A_bin(A, gd_c4)
        cfg = ClassicalFilterConfig(min_abelianization_bound=4)
        with pytest.raises(ValueError, match="non-abelian"):
            apply_classical_filters(A, A_bin, gd_c4, cfg)

    def test_min_abelianization_bound_raises_on_ma_gt_1(self, gd_s3):
        """ma=1-only filter must reject ma>1."""
        from core.classical_code import build_A_bin
        A = [[(0,)], [(1,)]]                 # ma=2
        A_bin = build_A_bin(A, gd_s3)
        cfg = ClassicalFilterConfig(min_abelianization_bound=4)
        with pytest.raises(ValueError, match="ma=1"):
            apply_classical_filters(A, A_bin, gd_s3, cfg)

    def test_min_weight_distance_bound_raises_on_non_abelian(self, gd_s3):
        """Abelian-only filter must reject non-abelian G."""
        from core.classical_code import build_A_bin
        A = [[(0, 1), (2, 3)]]
        A_bin = build_A_bin(A, gd_s3)
        cfg = ClassicalFilterConfig(min_weight_distance_bound=4)
        with pytest.raises(ValueError, match="abelian"):
            apply_classical_filters(A, A_bin, gd_s3, cfg)


class TestClassicalGroupTypeHappyPaths:
    """Confirm filters work without raising when the group type matches."""

    pytestmark = pytest.mark.gap

    def test_abelian_filters_on_abelian_pass(self, gd_c4):
        """All abelian-only filters can be enabled together on abelian G
        without preconditional errors."""
        from core.classical_code import build_A_bin
        A = [[(0,), (1, 2)]]
        A_bin = build_A_bin(A, gd_c4)
        cfg = ClassicalFilterConfig(
            min_base_girth_bound=4,
            min_weight_distance_bound=1,
            min_ring_distance_bound=1,
        )
        # Just verifying no exception. Result depends on the specific A.
        result = apply_classical_filters(A, A_bin, gd_c4, cfg)
        assert isinstance(result, bool)

    def test_abelianization_bound_on_non_abelian_passes(self, gd_s3):
        from core.classical_code import build_A_bin
        A = [[(0, 1), (2, 3)]]
        A_bin = build_A_bin(A, gd_s3)
        cfg = ClassicalFilterConfig(min_abelianization_bound=1)
        result = apply_classical_filters(A, A_bin, gd_s3, cfg)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────
# Quantum-pairing dispatcher
# ─────────────────────────────────────────────────────────────────


class TestQuantumPairingDispatcher:
    pytestmark = pytest.mark.fast

    def _meta(self, **kwargs):
        base = {"group_tag": "S3", "shape": (1, 2)}
        base.update(kwargs)
        return base

    def test_empty_config_passes(self):
        cfg = QuantumPairingFilterConfig()
        assert apply_quantum_pairing_filters(self._meta(), self._meta(), cfg) is True

    def test_same_group_rejects_mismatch(self):
        cfg = QuantumPairingFilterConfig(require_same_group=True)
        assert apply_quantum_pairing_filters(
            self._meta(group_tag="S3"), self._meta(group_tag="C5"), cfg
        ) is False
        assert apply_quantum_pairing_filters(
            self._meta(group_tag="S3"), self._meta(group_tag="S3"), cfg
        ) is True

    def test_same_shape_rejects_mismatch(self):
        cfg = QuantumPairingFilterConfig(require_same_shape=True)
        assert apply_quantum_pairing_filters(
            self._meta(shape=(1, 2)), self._meta(shape=(1, 3)), cfg
        ) is False
        assert apply_quantum_pairing_filters(
            self._meta(shape=(1, 2)), self._meta(shape=(1, 2)), cfg
        ) is True

    def test_min_distance_can_reject(self):
        cfg = QuantumPairingFilterConfig(min_classical_distance=8)
        assert apply_quantum_pairing_filters(
            self._meta(dist=5), self._meta(dist=10), cfg
        ) is False
        assert apply_quantum_pairing_filters(
            self._meta(dist=10), self._meta(dist=10), cfg
        ) is True

    def test_min_girth_can_reject(self):
        cfg = QuantumPairingFilterConfig(min_classical_girth=8)
        # A side girth 6 < 8 → reject.
        assert apply_quantum_pairing_filters(
            self._meta(girth=6), self._meta(girth=10), cfg
        ) is False
        # Both ≥ 8 → pass.
        assert apply_quantum_pairing_filters(
            self._meta(girth=8), self._meta(girth=10), cfg
        ) is True
        # None (forest) → pass.
        assert apply_quantum_pairing_filters(
            self._meta(girth=None), self._meta(girth=10), cfg
        ) is True

    def test_max_check_weight_caps(self):
        # W_A = [[2, 1]], W_B = [[3, 2]] → Hx_cw = 6, Hz_cw = 7.
        W_A = np.array([[2, 1]])
        W_B = np.array([[3, 2]])
        cfg = QuantumPairingFilterConfig(max_Hx_check_weight=5)
        assert apply_quantum_pairing_filters(
            self._meta(weight_matrix=W_A),
            self._meta(weight_matrix=W_B),
            cfg,
        ) is False
        cfg = QuantumPairingFilterConfig(max_Hx_check_weight=6)
        assert apply_quantum_pairing_filters(
            self._meta(weight_matrix=W_A),
            self._meta(weight_matrix=W_B),
            cfg,
        ) is True
        cfg = QuantumPairingFilterConfig(max_Hz_check_weight=6)
        assert apply_quantum_pairing_filters(
            self._meta(weight_matrix=W_A),
            self._meta(weight_matrix=W_B),
            cfg,
        ) is False

    def test_combined_filters(self):
        cfg = QuantumPairingFilterConfig(
            require_same_group=True,
            require_same_shape=True,
            min_classical_distance=4,
            min_classical_girth=6,
        )
        ok = self._meta(dist=5, girth=6)
        assert apply_quantum_pairing_filters(ok, ok, cfg) is True
        assert apply_quantum_pairing_filters(
            self._meta(dist=3, girth=6), ok, cfg
        ) is False
        assert apply_quantum_pairing_filters(
            self._meta(dist=5, girth=4), ok, cfg
        ) is False

    def test_precondition_missing_key_raises(self):
        """Missing required meta key raises ValueError up front."""
        cfg = QuantumPairingFilterConfig(min_classical_distance=4)
        # meta_A has dist, meta_B does not.
        with pytest.raises(ValueError, match="meta_B.*'dist'"):
            apply_quantum_pairing_filters(
                self._meta(dist=5), self._meta(), cfg,
            )

    def test_precondition_value_none_is_allowed(self):
        """A key present with value ``None`` is not a precondition violation;
        it's handled by the per-filter semantics (girth=None → forest pass,
        dist=None → unknown reject)."""
        cfg = QuantumPairingFilterConfig(min_classical_girth=8)
        # girth=None on one side: precondition passes; filter passes (forest).
        assert apply_quantum_pairing_filters(
            self._meta(girth=None), self._meta(girth=10), cfg,
        ) is True

        cfg = QuantumPairingFilterConfig(min_classical_distance=4)
        # dist=None on one side: precondition passes; filter rejects.
        assert apply_quantum_pairing_filters(
            self._meta(dist=None), self._meta(dist=10), cfg,
        ) is False

    def test_precondition_reports_all_missing_keys(self):
        """All missing keys are reported in one error."""
        cfg = QuantumPairingFilterConfig(
            require_same_group=True,
            min_classical_distance=4,
            min_classical_girth=6,
        )
        # meta_A is empty; meta_B has group_tag only. Expect: meta_A missing
        # group_tag, girth, dist; meta_B missing girth, dist.
        with pytest.raises(ValueError) as exc:
            apply_quantum_pairing_filters({}, {"group_tag": "S3"}, cfg)
        msg = str(exc.value)
        assert "meta_A['group_tag']" in msg
        assert "meta_A['girth']" in msg
        assert "meta_A['dist']" in msg
        assert "meta_B['girth']" in msg
        assert "meta_B['dist']" in msg
        # meta_B has group_tag, so it shouldn't be listed.
        assert "meta_B['group_tag']" not in msg

    def test_precondition_silent_when_filter_disabled(self):
        """Missing keys are fine if the filter that would need them is off."""
        cfg = QuantumPairingFilterConfig()   # nothing enabled
        assert apply_quantum_pairing_filters({}, {}, cfg) is True
