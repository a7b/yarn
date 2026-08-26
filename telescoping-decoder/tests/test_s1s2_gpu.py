"""S1/S2 stage-builder and repeatability tests (skipped without a CUDA GPU).

Covers the glue around the CUDA kernels rather than the kernels themselves:
the arrays :class:`Stage2` derives for the GARI decoder and repeatability for
an unchanged input order and batching configuration.
"""
import numpy as np
import pytest

from conftest import (DEM_FILE, GARI_NPZ, MATRICES_NPZ, fit_s1, fit_s2,
                      needs_gpu)

pytestmark = needs_gpu


@pytest.fixture(scope="module")
def system():
    from telescoping_decoder.system import DecodingSystem
    return DecodingSystem.from_npz(gari_npz=GARI_NPZ,
                                   matrices_npz=MATRICES_NPZ)


@pytest.fixture(scope="module")
def sampled():
    import stim
    dem = stim.DetectorErrorModel.from_file(DEM_FILE)
    det, obs, _ = dem.compile_sampler(seed=23).sample(
        shots=1000, bit_packed=False)
    return det.astype(np.uint8), obs.astype(np.uint8)


def test_s2_gari_builder_arrays(system):
    """Stage2's GARI arrays must match the npz recomputed independently."""
    from telescoping_decoder.config import S2Config
    from telescoping_decoder.s2 import Stage2

    cfg = S2Config()
    st = Stage2(system, fit_s2(cfg), "gari")
    mats = np.load(GARI_NPZ)

    rel_rows = mats["gari_relevant_rows"].astype(np.int32)
    rel_p = np.clip(mats["gari_relevant_priors"].astype(np.float64),
                    1e-300, 0.5 - 1e-12)
    cbb = mats["gari_col_block_bounds"]
    bi = {"eZ": 0, "eX": 1, "eY": 2, "ebarZ": 3, "ebarX": 4}[
        str(mats["gari_answer_block"])]
    weight_llr = np.zeros(int(mats["h_shape"][1]), dtype=np.float32)
    weight_llr[int(cbb[bi]):int(cbb[bi + 1])] = np.abs(
        np.log((1.0 - rel_p) / rel_p)).astype(np.float32)
    g_layers = mats["gari_layers"]
    det_lo, det_hi = int(g_layers[2][0]), int(g_layers[2][1])
    kk = max(1, min(int(cfg.k), det_hi - det_lo))
    bounds = np.linspace(det_lo, det_hi, kk + 1, dtype=np.int64)
    layers = ([(int(g_layers[0][0]), int(g_layers[0][1])),
               (int(g_layers[1][0]), int(g_layers[1][1]))]
              + [(int(bounds[i]), int(bounds[i + 1]))
                 for i in range(kk) if bounds[i + 1] > bounds[i]])

    got = st.gari_arrays
    assert np.array_equal(got["conv_rows"], rel_rows)
    assert np.array_equal(got["weight_llr"], weight_llr)
    assert got["layers"] == layers


def test_s1_determinism_and_batching(system, sampled):
    import scipy.sparse as sp

    from telescoping_decoder.config import S1Config
    from telescoping_decoder.s1 import Stage1

    det, obs = sampled
    st = Stage1(system, fit_s1(S1Config()), "gari")
    low_batches = []
    low_run = st.decoder.run

    def capture_run(syndromes, **kwargs):
        out = low_run(syndromes, **kwargs)
        low_batches.append(out)
        return out

    st.decoder.run = capture_run
    try:
        a = st.decode(det, obs)
    finally:
        del st.decoder.run

    sizes = [
        min(st.cfg.shots_per_batch, len(det) - start)
        for start in range(0, len(det), st.cfg.shots_per_batch)
    ]
    low_obs = np.concatenate([
        np.asarray(out["obs_pred"][:size], dtype=np.uint8)
        for out, size in zip(low_batches, sizes)
    ])
    low_acc = np.concatenate([
        np.asarray(out["relevant_converged"][:size], dtype=bool)
        for out, size in zip(low_batches, sizes)
    ])
    low_corr = np.concatenate([
        np.asarray(out["correction"][:size], dtype=np.uint8)
        for out, size in zip(low_batches, sizes)
    ])
    mats = np.load(GARI_NPZ)
    L = sp.csr_matrix(
        (mats["l_data"], mats["l_indices"], mats["l_indptr"]),
        shape=tuple(mats["l_shape"]),
    )
    H = sp.csr_matrix(
        (mats["h_data"], mats["h_indices"], mats["h_indptr"]),
        shape=tuple(mats["h_shape"]),
    )
    rel_rows = mats["gari_relevant_rows"].astype(np.int64)
    expected_obs = np.asarray((L @ low_corr.T).T % 2, dtype=np.uint8)
    expected_acc = ~np.any(
        np.asarray((H[rel_rows] @ low_corr.T).T % 2, dtype=np.uint8)
        != det[:, rel_rows],
        axis=1,
    )
    assert np.array_equal(low_obs, expected_obs)
    assert np.array_equal(low_acc, expected_acc)
    assert np.array_equal(a["obs_pred"], low_obs)
    assert np.array_equal(a["accepted"], low_acc)

    del low_batches, low_corr
    b = st.decode(det, obs)
    assert np.array_equal(a["accepted"], b["accepted"])
    assert np.array_equal(a["obs_pred"], b["obs_pred"])
    # different batch composition, same per-shot results (S1 is deterministic)
    c = st.decode(det[100:300], obs[100:300])
    assert np.array_equal(a["accepted"][100:300], c["accepted"])
    assert np.array_equal(a["obs_pred"][100:300], c["obs_pred"])
    # sanity: at p=0.001 the bulk of shots must be accepted
    assert a["accepted"].mean() > 0.5


def test_s1_s2_150_code_1000_shots(system, sampled):
    import scipy.sparse as sp

    from telescoping_decoder.config import S1Config, S2Config
    from telescoping_decoder.s1 import Stage1
    from telescoping_decoder.s2 import Stage2

    det, obs = sampled
    assert det.shape[0] == 1000
    s1 = Stage1(system, fit_s1(S1Config()), "gari")
    r1 = s1.decode(det, obs)
    rej = ~r1["accepted"]
    if not rej.any():
        pytest.skip("no S1 rejects in this sample")
    s2 = Stage2(system, fit_s2(S2Config()), "gari")

    # Capture Stage2's low-level outputs during its normal wrapper call. This
    # tests the whole S1 -> S2 path while running relay BP only once.
    low_batches = []
    low_run = s2.decoder.run

    def capture_run(syndromes):
        out = low_run(syndromes)
        low_batches.append(out)
        return out

    s2.decoder.run = capture_run
    try:
        r2 = s2.decode(det[rej], obs[rej])
    finally:
        # Remove the instance-level closure so it cannot keep the decoder and
        # its CUDA graphs alive in a reference cycle during interpreter exit.
        del s2.decoder.run

    n_rej = int(rej.sum())
    outer_batch = max(s2.cfg.shots_per_batch, s2.cfg.batch_size)
    batch_sizes = [
        min(outer_batch, n_rej - start)
        for start in range(0, n_rej, outer_batch)
    ]
    assert len(low_batches) == len(batch_sizes)
    low_acc = np.concatenate([
        np.asarray(out["converged"][:size], dtype=bool)
        for out, size in zip(low_batches, batch_sizes)
    ])
    low_obs = np.concatenate([
        np.asarray(out["obs_pred"][:size], dtype=np.uint8)
        for out, size in zip(low_batches, batch_sizes)
    ])
    low_corr = np.concatenate([
        np.asarray(out["correction"][:size], dtype=np.uint8)
        for out, size in zip(low_batches, batch_sizes)
    ])

    # The low-level relay decoder computes each accepted shot's agreed coset
    # while collecting quorum votes. Check it against L @ best_corr.
    mats = np.load(GARI_NPZ)
    L = sp.csr_matrix(
        (mats["l_data"], mats["l_indices"], mats["l_indptr"]),
        shape=tuple(mats["l_shape"]),
    )
    expected_obs = np.asarray(
        (L @ low_corr[low_acc].T).T % 2,
        dtype=np.uint8,
    )
    assert np.array_equal(low_obs[low_acc], expected_obs)
    assert not low_obs[~low_acc].any()

    # The wrapper must return those same host-side cosets without another GPU
    # matvec, and logical-error flags are meaningful only for accepted shots.
    assert np.array_equal(r2["accepted"], low_acc)
    assert np.array_equal(r2["obs_pred"], low_obs)
    expected_le = low_acc & np.any(low_obs != obs[rej], axis=1)
    assert np.array_equal(r2["le"], expected_le)
