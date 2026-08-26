from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent / "data"
STEM = DATA / "paper_nonab_150_30_10_xyz_p0.001_r10_xinit_L1_hookfree"

GARI_NPZ = f"{STEM}_gari_matrices.npz"
MATRICES_NPZ = f"{STEM}_matrices.npz"
DEM_FILE = f"{STEM}.dem"


def has_gpu() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not has_gpu(), reason="needs a CUDA GPU")

# Per-shot GPU cost of the bundled artifacts' GARI system (the largest of the
# three), from the README sizing table. The shipped batch defaults assume a
# data-center card; on anything smaller the suite must shrink them or it
# thrashes/OOMs. Batch size is a throughput knob only, so this cannot change
# any test's expected result.
_S1_BYTES_PER_SHOT = int(2.3 * 2**20)
_S2_BYTES_PER_SHOT = int(3.3 * 2**20)


def _free_vram(fraction: float = 0.6) -> int:
    import cupy
    return int(cupy.cuda.runtime.memGetInfo()[0] * fraction)


def fit_s1(cfg):
    """Shrink an S1Config's batch to what the visible GPU can hold."""
    budget = max(64, _free_vram() // _S1_BYTES_PER_SHOT)
    cfg.shots_per_batch = min(cfg.shots_per_batch, budget)
    return cfg


def fit_s2(cfg):
    """Shrink an S2Config's batch to what the visible GPU can hold."""
    budget = max(64, _free_vram() // _S2_BYTES_PER_SHOT)
    cfg.batch_size = min(cfg.batch_size, budget)
    cfg.shots_per_batch = min(cfg.shots_per_batch, cfg.batch_size)
    return cfg


def fit_config(cfg):
    """Same, for a whole TelescopeConfig."""
    fit_s1(cfg.s1)
    fit_s2(cfg.s2)
    return cfg


@pytest.fixture(scope="session")
def hard_shots():
    """Deterministic 'heavy' shots that push past S3-A: sample errors from
    inflated priors on the ORIGINAL system, s = H e mod 2, obs = L e mod 2.
    Syndromes stay in detector space / stim order (the package contract)."""
    import scipy.sparse as sp
    d = np.load(MATRICES_NPZ)
    H = sp.csr_matrix((d["h_data"], d["h_indices"], d["h_indptr"]),
                      shape=tuple(d["h_shape"]))
    L = sp.csr_matrix((d["l_data"], d["l_indices"], d["l_indptr"]),
                      shape=tuple(d["l_shape"]))
    probs = d["probs"]
    rng = np.random.default_rng(20260812)
    n_shots = 24
    errs = (rng.random((n_shots, H.shape[1]))
            < probs[None, :] * 8.0).astype(np.int64)
    synd = (errs @ H.T.astype(np.int64)) % 2
    obs = (errs @ L.T.astype(np.int64)) % 2
    return (np.ascontiguousarray(synd, dtype=np.uint8),
            np.ascontiguousarray(obs, dtype=np.uint8))
