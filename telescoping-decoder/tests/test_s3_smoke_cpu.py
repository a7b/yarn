"""CPU-only end-to-end tests: the full facade with S1 and S2 disabled."""
import numpy as np
import pytest

from telescoping_decoder import Stage, TelescopeConfig, TelescopingDecoder

from conftest import DEM_FILE, GARI_NPZ, MATRICES_NPZ


def _cpu_config(**s3_overrides):
    cfg = TelescopeConfig()
    cfg.s1.enabled = False
    cfg.s2.enabled = False
    cfg.s3.n_procs = 4
    cfg.s4.enabled = False
    for k, v in s3_overrides.items():
        setattr(cfg.s3, k, v)
    return cfg


@pytest.fixture(scope="module")
def decoder():
    dec = TelescopingDecoder.from_npz(
        gari_npz=GARI_NPZ, matrices_npz=MATRICES_NPZ, config=_cpu_config())
    yield dec
    dec.close()


@pytest.fixture(scope="module")
def sampled():
    import stim
    dem = stim.DetectorErrorModel.from_file(DEM_FILE)
    det, obs, _ = dem.compile_sampler(seed=11).sample(
        shots=64, bit_packed=False)
    return det.astype(np.uint8), obs.astype(np.uint8)


def test_decode_with_truth(decoder, sampled):
    det, obs = sampled
    res = decoder.decode(det, true_obs=obs)
    assert res.stage.shape == (64,)
    assert res.converged.all(), "p=0.001 shots must all resolve in S3"
    assert res.le is not None and res.ler == 0.0
    assert set(np.unique(res.stage)) <= {Stage.S3A, Stage.S3B, Stage.S3C}
    assert all(l.startswith("S3") for l in res.label)


def test_decode_without_truth(decoder, sampled):
    det, _ = sampled
    res = decoder.decode(det[:8])
    assert res.le is None
    assert res.converged.all()
    assert res.obs_pred.shape == (8, decoder.system.n_obs)
    with pytest.raises(ValueError):
        _ = res.ler


def test_trivial_syndrome(decoder):
    z = np.zeros((1, decoder.system.n_detectors), dtype=np.uint8)
    res = decoder.decode(z, true_obs=np.zeros((1, decoder.system.n_obs),
                                              dtype=np.uint8))
    assert res.converged[0]
    assert res.label[0].endswith("_trivial")
    assert not res.le[0]


def test_batch_composition_invariance(decoder, sampled):
    det, _ = sampled
    full = decoder.decode(det[:16])
    part = decoder.decode(det[5:16],
                          shot_ids=np.arange(5, 16, dtype=np.uint64))
    assert np.array_equal(full.obs_pred[5:16], part.obs_pred)
    assert np.array_equal(full.stage[5:16], part.stage)
    assert np.array_equal(full.label[5:16], part.label)


def test_input_validation(decoder):
    with pytest.raises(ValueError):
        decoder.decode(np.zeros((2, 7), dtype=np.uint8))
    with pytest.raises(ValueError):
        decoder.decode(
            np.zeros((2, decoder.system.n_detectors), dtype=np.uint8),
            shot_ids=np.array([3, 3], dtype=np.uint64))


def test_no_memory_relay_uses_cached_transpose(monkeypatch):
    """A no-memory shot must not rebuild the sparse transpose."""
    import telescoping_decoder.s3 as s3

    class GuardedH:
        def __matmul__(self, vector):
            return np.array([vector[0] ^ vector[1]], dtype=np.uint8)

        @property
        def T(self):
            raise AssertionError(
                "_relay_bp rebuilt H.T instead of using its cache")

    monkeypatch.setattr(s3, "_G", {
        "H_csr": GuardedH(),
        "H_csr_T": np.array([[1.0], [1.0]], dtype=np.float32),
        "channel_llr": np.array([1.0, 1.0], dtype=np.float32),
        "n_vars": 2,
        "n_checks": 1,
        "synd_pad": 0,
    })
    calls = []

    def fake_call_bp(syndrome, prior, iter_count, alpha, seed):
        calls.append(prior.copy())
        if len(calls) == 1:
            return False, 1, np.array([-1.0, 1.0], dtype=np.float32)
        return True, 0, np.array([-1.0, -1.0], dtype=np.float32)

    monkeypatch.setattr(s3, "_call_bp", fake_call_bp)
    converged, correction, residual, _llr = s3._relay_bp(
        np.array([0], dtype=np.uint8),
        prior_scale=1.0,
        n_runs=2,
        iter_per_run=1,
        alpha=np.array([1.0], dtype=np.float32),
        seed_base=7,
        shot_idx=11,
    )

    assert converged
    assert residual == 0
    assert np.array_equal(correction, [1, 1])
    assert len(calls) == 2


def test_no_ip_label(hard_shots):
    """Shots that survive S3 without gurobipy get the NC_no_ip label."""
    synd, obs = hard_shots
    cfg = _cpu_config(
        # starve S3 so something falls through
        a_iters=4, b_runs=1, b_iters=4, b_gamma0_iters=4,
        c_variants=(("gc1", 0.8, 4, 0.3, 0.99, 1),),
    )
    cfg.s3.a_variants = (("std", 1.0, 0.0, 0),)
    with TelescopingDecoder.from_npz(
            gari_npz=GARI_NPZ, matrices_npz=MATRICES_NPZ,
            config=cfg) as dec:
        res = dec.decode(synd, true_obs=obs)
    nc = res.stage == Stage.NC
    if nc.any():
        assert set(res.label[nc]) == {"NC_no_ip"}
        # NC shots are scored against the zero prediction
        assert np.array_equal(res.le[nc], obs[nc].any(axis=1))
