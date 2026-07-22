"""Smoke tests that do not require a GPU."""

import numpy as np


def test_version_string():
    import sqetch
    assert isinstance(sqetch.__version__, str)
    assert sqetch.__version__.count(".") >= 1


def test_estimate_distance_callable():
    import sqetch
    assert callable(sqetch.estimate_distance)


def test_distance_result_is_dataclass():
    from dataclasses import is_dataclass, fields
    from sqetch import DistanceResult
    assert is_dataclass(DistanceResult)
    names = {f.name for f in fields(DistanceResult)}
    expected = {
        "best_weight",
        "found",
        "trials_run",
        "elapsed_seconds",
        "raw_iter_per_sec",
        "n",
        "k_null",
        "k_sub",
    }
    assert expected.issubset(names), f"Missing fields: {expected - names}"


def test_null_space_correctness():
    from sqetch._gf2 import gf2_null_space

    H = np.array(
        [
            [1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    W = gf2_null_space(H)
    if W.shape[0] > 0:
        product = (H.astype(np.int32) @ W.T.astype(np.int32)) % 2
        assert np.all(product == 0), "null-space rows must satisfy H @ W.T == 0 mod 2"


def test_input_validation_without_gpu():
    """Validation runs before the GPU is touched."""
    from sqetch import estimate_distance

    H = np.ones((2, 5), dtype=np.uint8)
    L = np.ones((1, 4), dtype=np.uint8)  # wrong width

    try:
        estimate_distance(H, L, num_trials=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched column counts")
