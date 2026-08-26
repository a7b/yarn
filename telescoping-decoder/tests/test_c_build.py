"""Compile cache, env override, and ctypes loadability of the C kernels."""
import ctypes
import os
from pathlib import Path

import pytest

from telescoping_decoder._c.build import KERNELS, ensure_lib


def test_build_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("CHECKSERIAL_BP_SO", raising=False)
    monkeypatch.delenv("RELAY_MEM_BP_SO", raising=False)
    p1 = ensure_lib("checkserial_bp")
    assert Path(p1).is_file()
    mtime = Path(p1).stat().st_mtime_ns
    # second call is a cache hit — same path, not rebuilt
    assert ensure_lib("checkserial_bp") == p1
    assert Path(p1).stat().st_mtime_ns == mtime


@pytest.mark.parametrize("name", KERNELS)
def test_symbols_load(tmp_path, monkeypatch, name):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv(f"{name.upper()}_SO", raising=False)
    lib = ctypes.CDLL(ensure_lib(name))
    symbol = {
        "checkserial_bp": "checkserial_bp_decode_fast",
        "relay_mem_bp": "relay_mem_bp_decode",
    }[name]
    assert getattr(lib, symbol) is not None


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    real = ensure_lib("relay_mem_bp")
    alt = tmp_path / "alt.so"
    alt.write_bytes(Path(real).read_bytes())
    monkeypatch.setenv("RELAY_MEM_BP_SO", str(alt))
    assert ensure_lib("relay_mem_bp") == str(alt)
    monkeypatch.setenv("RELAY_MEM_BP_SO", str(tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError):
        ensure_lib("relay_mem_bp")


def test_unknown_kernel():
    with pytest.raises(ValueError):
        ensure_lib("nope")


def test_c_kernels_reject_oversized_checks_safely(tmp_path, monkeypatch):
    """An oversized row must raise before Python reads the C output LLR."""
    import numpy as np
    import scipy.sparse as sp

    import telescoping_decoder.s3 as s3

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("CHECKSERIAL_BP_SO", raising=False)
    monkeypatch.delenv("RELAY_MEM_BP_SO", raising=False)

    degree = 2050
    H = sp.csr_matrix(np.ones((1, degree), dtype=np.uint8))
    monkeypatch.setattr(s3, "_G", {
        "lib": s3._load_lib(),
        "relay_lib": s3._load_relay_lib(),
        "H_rel": H,
        "rel_rows": np.array([0], dtype=np.int64),
        "h_indptr": np.ascontiguousarray(H.indptr, dtype=np.int32),
        "h_indices": np.ascontiguousarray(H.indices, dtype=np.int32),
        "channel_llr": np.ones(degree, dtype=np.float32),
        "n_checks": 1,
        "n_vars": degree,
        "nnz": H.nnz,
        "synd_pad": 0,
    })
    monkeypatch.setattr(s3, "_B", {})
    s3._pool_init()

    syndrome = np.zeros(1, dtype=np.uint8)
    prior = np.ones(degree, dtype=np.float32)
    alpha = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="at most 2048"):
        s3._call_bp(syndrome, prior, 1, alpha, seed=1)
    with pytest.raises(ValueError, match="at most 2048"):
        s3._call_relay_bp(
            syndrome, prior, 1, alpha, seed=1,
            gamma=np.zeros(degree, dtype=np.float32), m_init=None)

    assert s3._B["decoding"].shape == (degree,)
