"""Full telescoping-decoder smoke (all stages) — skipped without a GPU."""
import numpy as np
import pytest

from telescoping_decoder import Stage, TelescopeConfig, TelescopingDecoder

from conftest import (DEM_FILE, GARI_NPZ, MATRICES_NPZ, fit_config,
                      needs_gpu)

pytestmark = needs_gpu


def test_full_telescope():
    import stim
    dem = stim.DetectorErrorModel.from_file(DEM_FILE)
    det, obs, _ = dem.compile_sampler(seed=31).sample(
        shots=20_000, bit_packed=False)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    cfg = fit_config(TelescopeConfig())
    with TelescopingDecoder.from_npz(
            gari_npz=GARI_NPZ, matrices_npz=MATRICES_NPZ,
            config=cfg) as dec:
        res = dec.decode(det, true_obs=obs)

        counts = res.counts_by_stage()
        # every shot resolved or explicitly NC-flagged
        assert res.stage.shape == (20_000,)
        assert (res.converged | (res.stage == Stage.NC)).all()
        # S1 takes the bulk at p=0.001
        assert counts.get("S1", 0) > 10_000
        assert np.isfinite(res.ler)

        # Repeat the subset with its corresponding global shot IDs.
        sub = slice(1000, 1500)
        res2 = dec.decode(det[sub], true_obs=obs[sub],
                          shot_ids=np.arange(1000, 1500, dtype=np.uint64))
        assert np.array_equal(res.stage[sub], res2.stage)
        assert np.array_equal(res.obs_pred[sub], res2.obs_pred)
        assert np.array_equal(res.label[sub], res2.label)
