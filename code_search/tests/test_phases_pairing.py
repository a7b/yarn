"""Tests for ``search/phases/pairing.py``."""

import json
from pathlib import Path

import numpy as np
import pytest

from search.configs.config import (
    BPOSDConfig,
    ClassicalDistanceConfig,
    ClassicalStageConfig,
    GroupConfig,
    PairingFiltersConfig,
    PairingPoolConfig,
    PairingStageConfig,
    SamplingConfig,
    SearchConfig,
    SqetchVerifyConfig,
)
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    manifest_path,
    quantum_dir,
    tried_pairs_path,
)


def _classical_json(matrix, weight_matrix, *, gap_expr, group_tag, shape,
                    side, dist=4, girth=None):
    return {
        "gap_expr": gap_expr,
        "group_tag": group_tag,
        "side": side,
        "shape": list(shape),
        "weight_matrix": weight_matrix,
        "matrix": matrix,
        "dist": dist,
        "dist_estimator": "bposd",
        "dist_num_trials": 100,
        "dist_n_workers": 1,
        "dist_osd_order": 0,
        "girth_tanner": girth,
        "timestamp": 0,
    }


def _ring_matrix_for_s3() -> tuple:
    """A canonical 1×2 S3 ring matrix where col 1 is full-rank (anchor).

    Built via the pool sampler to guarantee the canonical form. Returns
    (M, W) where W is the weight matrix.
    """
    from core.group import GroupData
    from search.sampling._shared.full_rank_block_pool import (
        build_full_rank_block_pool,
        sample_A_from_pool,
    )
    gd = GroupData("SymmetricGroup(3)")
    pool = build_full_rank_block_pool(
        gd, weight=5, max_pool_size=4, max_tries=500, seed=0,
    )
    rng = np.random.default_rng(0)
    M = sample_A_from_pool(
        gd, pool, shape_a=(1, 2), free_col_weight=2, rng=rng,
    )
    assert M is not None
    W = [[len(entry) for entry in row] for row in M]
    return M, W


def _seed_classical_pool(cfg: SearchConfig, gap_expr: str, group_tag: str,
                         num_per_side: int = 2):
    """Write ``num_per_side`` synthetic classical JSONs into both sides."""
    a_dir = classical_A_dir(cfg) / "w2.5"
    b_dir = classical_B_dir(cfg) / "w2.5"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    M, W = _ring_matrix_for_s3()
    matrix_json = [[list(entry) for entry in row] for row in M]
    for i in range(num_per_side):
        for side, d_dir in (("A", a_dir), ("B", b_dir)):
            data = _classical_json(
                matrix_json, W, gap_expr=gap_expr, group_tag=group_tag,
                shape=cfg.shape, side=side, dist=5,
            )
            (d_dir / f"d5_w2.5_t100_{i}{side}.json").write_text(
                json.dumps(data)
            )


def _cfg(tmp_path: Path, *,
         gap_expr: str = "SymmetricGroup(3)",
         tag: str = "S3",
         pair_mode: str = "full_pool",
         max_pairs=None,
         min_quantum_pool_size: int = 0,
         enabled_sqetch: bool = False) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr=gap_expr, tag=tag),
        run_stages=["pairing"],
        classical=ClassicalStageConfig(
            weight_A=[[2, 5]], weight_B=[[2, 5]],
            sampling=SamplingConfig(total_samples=0, seed=0),
            distance=ClassicalDistanceConfig(
                d_target=4, num_trials=10, n_workers=1, osd_order=0,
            ),
        ),
        pairing=PairingStageConfig(
            bposd=BPOSDConfig(
                d_target=2, num_trials=20, n_workers=1, osd_order=0,
            ),
            sqetch_verify=SqetchVerifyConfig(enabled=enabled_sqetch),
            filters=PairingFiltersConfig(
                require_same_group=True,
                require_same_shape=True,
            ),
            pool=PairingPoolConfig(
                pair_mode=pair_mode,
                max_pairs=max_pairs,
                min_quantum_pool_size=min_quantum_pool_size,
            ),
        ),
        results_dir=tmp_path / "search_results",
    )


class TestPairingMissingConfig:
    pytestmark = pytest.mark.fast

    def test_no_pairing_config_raises(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path)
        cfg.pairing = None
        with pytest.raises(ValueError, match="cfg.pairing"):
            run_pairing(cfg)

    def test_unsupported_pair_mode_raises(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path, pair_mode="new_only")
        with pytest.raises(ValueError, match="pair_mode"):
            run_pairing(cfg)


class TestPairingEmptyPool:
    """No classical files saved → no pairs to try."""

    pytestmark = pytest.mark.fast

    def test_empty_pool_yields_no_quantum(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path)
        out = run_pairing(cfg)
        assert out == {"new_quantum": [], "n_pairs_tried": 0, "n_pairs_passed": 0}
        # tried_pairs.json is created (empty list) even with empty pool.
        # The manifest is NOT updated when nothing passed.
        assert not manifest_path(cfg).exists()


class TestPairingEndToEnd:
    """Seed a small classical pool, run pairing, expect at least one save."""

    pytestmark = pytest.mark.bposd

    def test_writes_quantum_and_tried_pairs(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path)
        _seed_classical_pool(cfg, "SymmetricGroup(3)", "S3", num_per_side=2)

        out = run_pairing(cfg)
        assert out["n_pairs_tried"] > 0
        # If everything passed, the pairs map to saved JSONs.
        for path_str in out["new_quantum"]:
            data = json.loads(Path(path_str).read_text())
            for required in ("k", "dx", "dz", "estimator",
                             "weight_A", "weight_B", "A", "B"):
                assert required in data, f"missing {required}"
            assert data["estimator"].startswith("bposd")
            assert data["group_tag"] == "S3"

        # tried_pairs.json carries every (a, b) attempted.
        tp = json.loads(tried_pairs_path(cfg).read_text())
        assert len(tp) == out["n_pairs_tried"]

    def test_max_pairs_cap(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path, max_pairs=2)
        _seed_classical_pool(cfg, "SymmetricGroup(3)", "S3", num_per_side=3)
        out = run_pairing(cfg)
        assert out["n_pairs_tried"] <= 2


class TestPairingFilterSameGroup:
    """Different group_tags must be rejected by the same_group filter."""

    pytestmark = pytest.mark.fast

    def test_same_group_rejects_mismatch(self, tmp_path):
        from search.phases.pairing import run_pairing
        cfg = _cfg(tmp_path)
        # Seed A side with group_tag S3, B side with group_tag X (mismatch).
        a_dir = classical_A_dir(cfg) / "w2.2"
        b_dir = classical_B_dir(cfg) / "w2.2"
        a_dir.mkdir(parents=True, exist_ok=True)
        b_dir.mkdir(parents=True, exist_ok=True)
        for side, d_dir, tag in (("A", a_dir, "S3"), ("B", b_dir, "OTHER")):
            data = _classical_json(
                [[[0, 1], [2, 3]]], [[2, 2]],
                gap_expr="SymmetricGroup(3)", group_tag=tag,
                shape=cfg.shape, side=side, dist=5,
            )
            (d_dir / f"d5_w2.2_t100_{side}.json").write_text(json.dumps(data))

        out = run_pairing(cfg)
        # All pairs are filtered out (group_tag mismatch); no quantum saved.
        assert out["new_quantum"] == []
        assert out["n_pairs_passed"] == 0
        assert out["n_pairs_tried"] >= 1   # the filter was consulted
