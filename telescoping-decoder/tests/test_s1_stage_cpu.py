from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "system_name,expected_accepted",
    [("original", [True, False]), ("gari", [False, True])],
)
def test_stage1_uses_low_level_device_results_without_correction(
        monkeypatch, system_name, expected_accepted):
    """Stage1 must not upload a host correction to calculate its result."""
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(
                null=SimpleNamespace(synchronize=lambda: None))))
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)

    from telescoping_decoder.s1 import Stage1

    class Decoder:
        def run(self, syndromes, *, n_valid):
            assert syndromes.shape == (2, 3)
            assert n_valid == 2
            # Deliberately omit `correction`: Stage1 should not need it.
            return {
                "converged": np.array([True, False]),
                "relevant_converged": np.array([False, True]),
                "obs_pred": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            }

    stage = Stage1.__new__(Stage1)
    stage.cfg = SimpleNamespace(shots_per_batch=2)
    stage.system_name = system_name
    stage.decoder = Decoder()
    stage.n_obs = 2
    stage.m = 3
    stage.n_det = 3
    stage.init_idx = None

    out = stage.decode(
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        true_obs=np.array([[1, 0], [1, 0]], dtype=np.uint8),
    )

    assert np.array_equal(out["accepted"], expected_accepted)
    assert np.array_equal(out["obs_pred"], [[1, 0], [0, 0]])
    assert np.array_equal(out["le"], [False, True])
    assert out["n_full_conv"] == 1
