"""GPU correctness on a small known code -- skipped if CUDA is unavailable."""

import numpy as np
import pytest


def _have_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(not _have_cuda(), reason="CUDA / PyTorch not available")


def _steane_inputs():
    H = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    L = np.ones((1, 7), dtype=np.uint8)
    return H, L


def test_steane_distance_three():
    from sqetch import estimate_distance

    H, L = _steane_inputs()
    result = estimate_distance(
        H,
        L,
        num_trials=500,
        d_target=3,
        k_sub=3,
        seed=0,
    )
    assert result.best_weight == 3
    assert result.trials_run > 0


def test_steane_early_stop_at_lower_target():
    """d_target=4 should early-stop on any weight-3 hit."""
    from sqetch import estimate_distance

    H, L = _steane_inputs()
    result = estimate_distance(
        H,
        L,
        num_trials=2000,
        d_target=4,
        k_sub=3,
        seed=1,
    )
    assert result.best_weight == 3
    assert result.found is True
