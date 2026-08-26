from types import SimpleNamespace

import numpy as np


def test_stage2_uses_low_level_obs_pred_without_correction(monkeypatch):
    """S2 must reuse the coset returned by its relay decoder."""
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(
                null=SimpleNamespace(synchronize=lambda: None))))
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)

    from telescoping_decoder.s2 import Stage2

    class Decoder:
        def run(self, syndromes):
            assert syndromes.shape == (2, 3)
            # Deliberately omit `correction`: Stage2 should not need it.
            return {
                "converged": np.array([True, False]),
                "obs_pred": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            }

    stage = Stage2.__new__(Stage2)
    stage.cfg = SimpleNamespace(shots_per_batch=2, batch_size=2)
    stage.decoder = Decoder()
    stage.n_obs = 2
    stage.m = 3
    stage.n_det = 3
    stage.init_idx = None

    out = stage.decode(
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        true_obs=np.array([[1, 0], [1, 0]], dtype=np.uint8),
    )

    assert np.array_equal(out["accepted"], [True, False])
    assert np.array_equal(out["obs_pred"], [[1, 0], [0, 0]])
    assert np.array_equal(out["le"], [False, False])
