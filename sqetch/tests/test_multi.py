"""Multi-GPU tests for sqetch.estimate_distance_multi.

Skipped automatically when fewer than 2 CUDA devices are visible.
"""

import numpy as np
import pytest


def _steane_inputs():
    H = np.array(
        [
            [1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    L = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    return H, L


def _has_multi_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        return False


def test_detect_gpus_returns_list():
    from sqetch import detect_gpus
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    for g in gpus:
        assert "idx" in g
        assert "cc_str" in g
        assert "shmem_limit" in g


def test_detect_gpus_explicit_device_subset():
    from sqetch import detect_gpus
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("no CUDA")
    gpus = detect_gpus(devices=[0])
    assert len(gpus) == 1
    assert gpus[0]["idx"] == 0


@pytest.mark.skipif(not _has_multi_gpu(),
                    reason="requires >= 2 visible CUDA devices")
def test_steane_multi_gpu_matches_single():
    from sqetch import estimate_distance, estimate_distance_multi
    H, L = _steane_inputs()

    r1 = estimate_distance(H, L, num_trials=400, d_target=3, k_sub=3, seed=0)
    r2 = estimate_distance_multi(H, L, num_trials=400, d_target=3, k_sub=3, seed=0)

    # Both should find dx=3 (Steane code's true distance) with enough trials.
    assert r1.best_weight == 3
    assert r2.best_weight == 3


@pytest.mark.skipif(not _has_multi_gpu(),
                    reason="requires >= 2 visible CUDA devices")
def test_multi_gpu_trials_sum_across_shards():
    from sqetch import estimate_distance_multi
    H, L = _steane_inputs()

    n_trials = 400
    r = estimate_distance_multi(
        H, L, num_trials=n_trials, d_target=None,
        k_sub=3, seed=1,
    )
    # When no early-stop, total trials_run = n_trials.
    assert r.trials_run == n_trials


@pytest.mark.skipif(not _has_multi_gpu(),
                    reason="requires >= 2 visible CUDA devices")
def test_multi_gpu_early_stop():
    from sqetch import estimate_distance_multi
    H, L = _steane_inputs()

    # d_target=4 > true distance 3, so the first weight-3 hit triggers early stop.
    r = estimate_distance_multi(
        H, L, num_trials=2000, d_target=4,
        k_sub=3, seed=2,
    )
    assert r.best_weight == 3
    assert r.found is True
