"""S3 — the CPU relay stages (S3-A, S3-B, S3-C) of the telescoping decoder.

Each stage decodes one shot at a time through the C belief-propagation
kernels in ``_c/``, and is run across a process pool by
:class:`~telescoping_decoder.decoder.TelescopingDecoder`:

  S3-A   an ensemble of standalone BP variants; first converged wins.
  S3-B   relay-BP with per-iteration memory; converged legs vote on their
         logical coset and a coset must reach quorum to be accepted.
  S3-C   a sequential sweep of independent BP configurations; the first
         converged attempt wins.

Every BP call tests acceptance over the *relevant* check rows (the
``_call_bp`` / ``_call_relay_bp`` wrappers re-test kernel-NC results against
those rows; residuals are relevant-row unsat counts). On a non-GARI system
the relevant rows cover every row, so this reduces to full-row acceptance.
Shots that no stage accepts are handed to S4 for exact certification.

Per-shot decode is a pure function of ``(syndrome, cfg, base_seed, gid)``:
every seed derives from :func:`_seed32`, so results never depend on batch
composition, pool size, or shot order. Worker processes call
``_install_cfg(flat_cfg)`` and ``_ensure_init(npz_path)`` once, then
``_pool_init()`` to allocate the per-process scratch buffers.

Seed compatibility: the string literals fed to :func:`_seed32` in this module
("phase0", "phase2", "phase2_mem", "phase2_noise", "phase3_single") are
opaque hash inputs, not descriptions. They predate the S1..S4 stage naming
and are deliberately left alone: renaming one changes every seed it feeds,
so decoding stays correct but stops reproducing anything decoded before the
rename. Sites carrying them are marked ``frozen seed tag``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import traceback

import numpy as np
import scipy.sparse as sp

from ._c.build import ensure_lib

# ---------------------------------------------------------------------------
# Process-local state. The single source of truth for per-task parameters is
# _G['cfg'] (a flat namespace: TelescopeConfig._flatten()). Pool workers
# populate _G via their initializer.
# ---------------------------------------------------------------------------

_G: dict = {}   # cfg, matrices, C libs, alpha schedules
_B: dict = {}   # per-process mutable buffers (llr, c2v, ...)


def _cfg():
    """Accessor for the active flat config. Callers must _install_cfg first."""
    return _G['cfg']


def _install_cfg(cfg) -> None:
    """Stash cfg in the module's _G so helpers can read it."""
    _G['cfg'] = cfg


def _schedule_alpha(max_iter, alpha_max, tau):
    return np.array([alpha_max * (1.0 - np.exp(-(t + 1) / tau))
                     for t in range(max_iter)], dtype=np.float32)


def _seed32(*parts) -> int:
    """Deterministic 32-bit seed from a tuple of ints/strings.

    Uses BLAKE2 as specified by RFC 7693. Use for any seed that must fit in
    uint32 -- np.random.RandomState (which rejects seeds >= 2**32) and the
    C BP kernel (which receives the seed via ctypes.c_uint). Hashing rather
    than arithmetic on the shot index (e.g. ``seed_base + shot_idx * 31``) is
    deliberate: that form overflows RandomState past shot_idx ~1.4e8 and
    collides modulo 2**32 at the C layer past ~4.3e7.
    """
    b = b"|".join(str(p).encode() for p in parts)
    return int.from_bytes(hashlib.blake2b(b, digest_size=4).digest(), "big")


def _load_lib():
    # Resolved via the runtime build cache; the CHECKSERIAL_BP_SO env
    # override is honored inside ensure_lib().
    so_path = ensure_lib("checkserial_bp")
    lib = ctypes.CDLL(so_path)
    lib.checkserial_bp_decode_fast.restype = None
    lib.checkserial_bp_decode_fast.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    return lib


def _load_relay_lib():
    """Load the standalone relay_mem_bp.so (relay-BP-with-memory kernel).

    Separate .so from checkserial_bp.so; checkserial_bp.c is untouched. Symbol:
    relay_mem_bp_decode (checkserial args + gamma, m_init, m_prev, lambda_prev
    inserted before the output pointers)."""
    # Resolved via the runtime build cache; the RELAY_MEM_BP_SO env
    # override is honored inside ensure_lib().
    so_path = ensure_lib("relay_mem_bp")
    lib = ctypes.CDLL(so_path)
    lib.relay_mem_bp_decode.restype = None
    lib.relay_mem_bp_decode.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        # relay-memory: gamma, m_init, m_prev, lambda_prev
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    return lib


def _ensure_init(matrices_npz_path: str) -> None:
    """Load matrices + C lib once per process. Subsequent calls are no-ops."""
    if _G.get('_ready'):
        return
    saved_cfg = _G.get('cfg')
    if saved_cfg is None:
        raise RuntimeError(
            "_ensure_init called before _G['cfg'] was set; call "
            "_install_cfg(cfg) before any helper.")
    _G.clear()
    _G['cfg'] = saved_cfg

    d = np.load(matrices_npz_path)
    required = {"h_data", "h_indices", "h_indptr", "h_shape",
                "probs", "l_data", "l_indices", "l_indptr", "l_shape"}
    missing = required - set(d.files)
    assert not missing, f"matrices npz missing keys: {missing}"

    # matrices npz is saved as CSR (h_indptr length = n_rows+1).
    H_csr = sp.csr_matrix(
        (d['h_data'].astype(np.uint8), d['h_indices'].astype(np.int32),
         d['h_indptr'].astype(np.int32)), shape=tuple(d['h_shape']))
    probs = d['probs'].astype(np.float64)
    channel_llr = np.log((1.0 - probs) / probs).astype(np.float32)
    L_csr = sp.csr_matrix(
        (d['l_data'].astype(np.uint8), d['l_indices'].astype(np.int32),
         d['l_indptr'].astype(np.int32)), shape=tuple(d['l_shape']))
    n_checks, n_vars = H_csr.shape

    cfg = saved_cfg
    # GARI npz: detector rows come first, then zero-syndrome consistency
    # rows. Stage artifacts stay n_det_rows wide; _pad_synd() zero-extends
    # at the decode boundary.
    #
    # The GARI bundle is optional. A plain matrices NPZ for the original or
    # init_dets system sets rel_rows to all rows. Relevant-row and full-row
    # residuals are then identical; weights use |channel_llr| and no padding
    # is required.
    gari_keys = {"gari_n_detectors", "gari_relevant_rows",
                 "gari_relevant_priors", "gari_col_block_bounds",
                 "gari_answer_block"}
    if gari_keys <= set(d.files):
        n_det_rows = int(d["gari_n_detectors"])

        # ---- relevant-half acceptance bundle (the GARI accept rule) ------
        # accept  <=> H[rel_rows] @ corr == syndrome[rel_rows] (D_X·ē_Z = s_X)
        # weight  = sum over the answer block of |combined-prior llr| — the
        #           auxiliary blocks of a half-accepted correction are not
        #           guaranteed consistent, so cost/min-weight must never
        #           read them.
        rel_rows = d["gari_relevant_rows"].astype(np.int64)
        H_rel = H_csr[rel_rows]
        rel_p = np.clip(d["gari_relevant_priors"].astype(np.float64),
                        1e-300, 0.5 - 1e-12)
        cb = d["gari_col_block_bounds"]
        answer_block = str(d["gari_answer_block"])
        block_idx = {"eZ": 0, "eX": 1, "eY": 2,
                     "ebarZ": 3, "ebarX": 4}[answer_block]
        ans_lo, ans_hi = int(cb[block_idx]), int(cb[block_idx + 1])
        weight_llr = np.zeros(n_vars, dtype=np.float32)
        weight_llr[ans_lo:ans_hi] = np.abs(
            np.log((1.0 - rel_p) / rel_p)).astype(np.float32)
    else:
        n_det_rows = n_checks
        rel_rows = np.arange(n_checks, dtype=np.int64)
        H_rel = H_csr
        weight_llr = np.abs(channel_llr).astype(np.float32)

    _G.update({
        'lib': _load_lib(),
        'relay_lib': _load_relay_lib(),
        'H_csr': H_csr, 'L_csr': L_csr,
        'H_csr_T': H_csr.T.tocsr(),     # relay BP "near unsat-check" mask
        'h_indptr': np.ascontiguousarray(H_csr.indptr, dtype=np.int32),
        'h_indices': np.ascontiguousarray(H_csr.indices, dtype=np.int32),
        'channel_llr': np.ascontiguousarray(channel_llr),
        'n_checks': n_checks, 'n_vars': n_vars, 'nnz': H_csr.nnz,
        'n_det_rows': n_det_rows, 'synd_pad': n_checks - n_det_rows,
        'rel_rows': rel_rows, 'H_rel': H_rel, 'weight_llr': weight_llr,
        'alpha_s3a': _schedule_alpha(
            cfg.s3a_iters, cfg.s3a_alpha_max, cfg.s3a_tau),
        'alpha_s3b': _schedule_alpha(
            cfg.s3b_iters, cfg.s3b_alpha_max, cfg.s3b_tau),
        '_ready': True,
    })


def _init_buffers() -> None:
    """Allocate per-process mutable buffers. Called once per subprocess."""
    nv, nnz, nc = _G['n_vars'], _G['nnz'], _G['n_checks']
    _B.update({
        'llr': np.empty(nv, dtype=np.float32),
        'c2v': np.empty(nnz, dtype=np.float32),
        'check_order': np.empty(nc, dtype=np.int32),
        'decoding': np.empty(nv, dtype=np.uint8),
        # relay-memory scratch (relay_mem_bp_decode)
        'm_prev': np.empty(nv, dtype=np.float32),
        'lambda_prev': np.empty(nv, dtype=np.float32),
    })


def _pool_init():
    """Subprocess initializer: clear and re-allocate mutable buffers."""
    _B.clear()
    _init_buffers()


def _pad_synd(syndrome):
    """Zero-extend a detector-space syndrome to the decode width (GARI
    consistency rows expect zero syndrome bits). Idempotent: no-op when the
    syndrome already has n_checks bits."""
    if not _G['synd_pad'] or syndrome.shape[-1] == _G['n_checks']:
        return syndrome
    return np.ascontiguousarray(np.concatenate(
        [syndrome, np.zeros(_G['synd_pad'], dtype=syndrome.dtype)]))


def _rel_residual(llr, syndrome):
    """Relevant-half unsat count of the hard decision of `llr` against the
    (padded) syndrome: |{r in rel_rows : (H @ corr)_r != s_r}|. 0 means the
    correction is accepted under the GARI relevant-half rule (the answer
    block explains the answer-side detectors; the auxiliary consistency
    rows and the unused half need not be settled). Cost: one ~61k-nnz
    matvec — ~0.2% of a BP leg."""
    corr = (llr < 0).astype(np.int32)
    pred = np.asarray(_G['H_rel'] @ corr).ravel() & 1
    return int(np.count_nonzero(
        pred != syndrome[_G['rel_rows']].astype(np.int32)))


def _call_bp(syndrome, prior, iter_count, alpha, seed):
    """Wrap the C checkserial_bp_decode_fast kernel.

    GARI half-acceptance: the kernel's internal stop is full-row, so on a
    kernel-NC result we re-test the final hard decision against the
    relevant rows only and accept if they are satisfied. The returned
    residual is the relevant-row unsatisfied count used to rank NC results."""
    lib = _G['lib']
    syndrome = _pad_synd(syndrome)
    nc, nv = _G['n_checks'], _G['n_vars']
    oi = ctypes.c_int(0)
    oc = ctypes.c_int(0)
    orr = ctypes.c_int(0)
    lib.checkserial_bp_decode_fast(
        _G['h_indptr'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _G['h_indices'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nc, nv, _G['nnz'],
        syndrome.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        prior.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        iter_count, alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(0.0), 1, ctypes.c_uint(seed),
        _B['llr'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _B['c2v'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _B['check_order'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _B['decoding'].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(oi), ctypes.byref(oc), ctypes.byref(orr),
    )
    if oi.value < 0:
        raise ValueError(
            "S3 CPU BP supports at most 2048 variables per check")
    if oc.value:
        return 1, 0, _B['llr'].copy()
    # GARI half-acceptance on the final hard decision (see _rel_residual).
    rel_res = _rel_residual(_B['llr'], syndrome)
    return (1 if rel_res == 0 else 0), rel_res, _B['llr'].copy()


def _call_relay_bp(syndrome, prior, iter_count, alpha, seed, gamma, m_init):
    """Wrap the relay_mem_bp_decode kernel (relay-BP-with-memory).

    `prior` is Lambda0 (the scaled channel prior, immutable bias). `gamma` is a
    per-variable float32 array (fixed for this leg); `m_init` is the warm-start
    marginal (float32 array) or None for a cold leg. Messages (c2v) are reset
    each call — only marginals warm-start, via m_init."""
    lib = _G['relay_lib']
    syndrome = _pad_synd(syndrome)
    nc, nv = _G['n_checks'], _G['n_vars']
    P_F = ctypes.POINTER(ctypes.c_float)
    oi = ctypes.c_int(0)
    oc = ctypes.c_int(0)
    orr = ctypes.c_int(0)
    g_ptr = gamma.ctypes.data_as(P_F) if gamma is not None else None
    mi_ptr = m_init.ctypes.data_as(P_F) if m_init is not None else None
    lib.relay_mem_bp_decode(
        _G['h_indptr'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _G['h_indices'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nc, nv, _G['nnz'],
        syndrome.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        prior.ctypes.data_as(P_F),
        iter_count, alpha.ctypes.data_as(P_F),
        ctypes.c_float(0.0), 1, ctypes.c_uint(seed),
        _B['llr'].ctypes.data_as(P_F),
        _B['c2v'].ctypes.data_as(P_F),
        _B['check_order'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        g_ptr, mi_ptr,
        _B['m_prev'].ctypes.data_as(P_F),
        _B['lambda_prev'].ctypes.data_as(P_F),
        _B['decoding'].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(oi), ctypes.byref(oc), ctypes.byref(orr),
    )
    if oi.value < 0:
        raise ValueError(
            "S3 CPU relay BP supports at most 2048 variables per check")
    if oc.value:
        return 1, 0, _B['llr'].copy()
    # GARI half-acceptance on the final hard decision (see _rel_residual).
    rel_res = _rel_residual(_B['llr'], syndrome)
    return (1 if rel_res == 0 else 0), rel_res, _B['llr'].copy()


def _relay_bp_mem(syndrome, prior_scale, num_sets, set_iters, alpha,
                  gamma0_iters, alpha0, gamma0, gamma_center, gamma_width,
                  seed_base, shot_idx, base_llr=None, stop_nconv=5,
                  votes=None, quorum=1, tag=None):
    """Relay-BP with per-iteration memory + marginal warm-start (CPU).

    Leg 0: ordered scalar gamma0 over gamma0_iters. Legs 1..num_sets: per-variable
    gamma ~ U[center +/- width/2] (signed), fixed for the leg; marginals warm-start
    from the previous leg. Collects converged corrections; stops after stop_nconv
    converged legs.

    Online first-to-quorum (votes is not None): each converged leg votes for its
    coset in the caller-shared `votes` dict ({coset tuple -> [n_votes,
    first_tag]}, tallied across variants); the moment a coset reaches `quorum`
    votes the function returns early with that coset as `hit_coset`. `tag`
    labels this variant's votes.

    Returns (converged_list, best_residual, best_llr, hit_coset) where
    converged_list is a list of (corr uint8, cost float, coset tuple) for every
    converged leg (for the quorum=1 min-weight selection), best_llr is the
    lowest-residual NC posterior (for the NC fallback), and hit_coset is the
    quorum-winning coset tuple or None."""
    nv = _G['n_vars']
    channel_llr = _G['channel_llr']
    Lambda0 = np.ascontiguousarray(
        base_llr if base_llr is not None else channel_llr * prior_scale,
        dtype=np.float32)
    rng = np.random.RandomState(_seed32(seed_base, shot_idx, "gamma"))
    lo = gamma_center - gamma_width / 2.0
    hi = gamma_center + gamma_width / 2.0

    converged = []
    best_res, best_llr = 999999, None
    m_init = None  # leg 0 cold start
    n_conv = 0
    for leg in range(num_sets + 1):
        if leg == 0:
            gamma = np.full(nv, np.float32(gamma0))
            iters, a = gamma0_iters, alpha0
        else:
            gamma = rng.uniform(lo, hi, size=nv).astype(np.float32)
            iters, a = set_iters, alpha
        seed = _seed32(seed_base, shot_idx, leg)
        conv, res, llr = _call_relay_bp(syndrome, Lambda0, iters, a, seed,
                                        gamma, m_init)
        if conv:
            corr = (llr < 0).astype(np.uint8)
            coset = tuple(_predict_obs(corr))
            converged.append((corr, _correction_cost(corr), coset))
            if votes is not None:
                v = votes.setdefault(coset, [0, tag])
                v[0] += 1
                if v[0] >= quorum:
                    return converged, best_res, best_llr, coset
            n_conv += 1
            if n_conv >= stop_nconv:
                break
        elif res < best_res:
            best_res, best_llr = res, llr.copy()
        m_init = llr.copy()  # marginal warm-start for the next leg
    return converged, best_res, best_llr, None


def _relay_bp(syndrome, prior_scale, n_runs, iter_per_run, alpha,
              seed_base, shot_idx, base_llr=None):
    """Multi-leg relay BP: each leg uses the previous leg's posterior re-mixed
    with the channel prior via per-variable random gamma weights.

    Returns (converged: bool, correction or None, best_residual: int, best_llr).
    """
    nv = _G['n_vars']
    H_csr, H_csr_T = _G['H_csr'], _G['H_csr_T']
    channel_llr = _G['channel_llr']
    syndrome = _pad_synd(syndrome)  # near-unsat mask compares over all rows

    scaled_llr = base_llr if base_llr is not None else channel_llr * prior_scale
    current_prior = scaled_llr.copy()
    rng = np.random.RandomState(_seed32(seed_base, shot_idx, "gamma"))
    best_res, best_llr = 999999, None

    for run in range(n_runs):
        seed = _seed32(seed_base, shot_idx, run)
        if run == 0:
            run_prior = current_prior.copy()
        else:
            pred_dec = (current_prior < 0).astype(np.uint8)
            pred_synd = np.asarray(
                (H_csr @ pred_dec.astype(np.int32)) % 2).ravel().astype(np.uint8)
            unsat = (pred_synd != syndrome).astype(np.float32)
            near_unsat = np.asarray(H_csr_T @ unsat).ravel()
            near_mask = (near_unsat > 0.5)
            gamma_wide = rng.uniform(0.0, 6.0, size=nv).astype(np.float32)
            gamma_narrow = rng.uniform(0.5, 1.5, size=nv).astype(np.float32)
            gamma = np.where(near_mask, gamma_wide, gamma_narrow)
            clipped_prior = np.clip(current_prior, -1e30, 1e30)
            run_prior = scaled_llr + (clipped_prior - scaled_llr) * gamma

        conv, res, llr = _call_bp(syndrome, run_prior, iter_per_run, alpha, seed)
        if conv:
            return True, (llr < 0).astype(np.uint8), 0, llr
        if res < best_res:
            best_res, best_llr = res, llr.copy()
        current_prior = llr.copy()

    return False, None, best_res, best_llr


def _predict_obs(correction):
    """Predicted logical-flip vector L @ c % 2 as uint8."""
    L = _G['L_csr']
    return np.asarray(
        L @ correction.astype(np.int32) % 2).ravel().astype(np.uint8)


def _correction_cost(correction):
    """LLR-weighted Hamming cost over the GARI answer block:
    sum(weight_llr[i]) over bits where correction[i] == 1. weight_llr is the
    |combined ē-block prior llr| at the answer columns and 0 elsewhere
    (built in _ensure_init) — under half-acceptance the auxiliary blocks of
    a correction are not guaranteed consistent, so they must not be costed."""
    return float(
        (_G['weight_llr'].astype(np.float64) * correction).sum())


def _check_obs(correction, true_obs):
    """Return True iff predicted observables (L @ c % 2) differ from
    `true_obs`."""
    obs_pred = _predict_obs(correction)
    return bool(np.any(obs_pred != true_obs))


def _le_split_from_obs(obs_pred, true_obs, n_meas):
    """Per-shot LE classification for the measured/unmeasured observable split.

    Returns ``(le_measured, le_unmeasured)`` bools: did any observable in rows
    ``[:n_meas]`` (measured/time-like) disagree with truth, and did any in rows
    ``[n_meas:]`` (surviving/unmeasured) disagree. ``le_measured or
    le_unmeasured`` equals the usual total-LE bool. Caller guards on
    ``n_meas > 0``."""
    diff = np.asarray(obs_pred, np.uint8) != np.asarray(true_obs, np.uint8)
    return bool(diff[:n_meas].any()), bool(diff[n_meas:].any())


def _classify_obs(obs_pred, true_obs):
    """Return ``(le_bool, le_class)`` from a predicted observable vector.

    ``le_class`` is a 2-bit mask: bit0 = a measured (row [:n]) observable
    failed, bit1 = an unmeasured (row [n:]) observable failed, where
    ``n = cfg.n_measured_observables``. When n == 0 the class is always 0
    (no split requested) and only the total ``le_bool`` is meaningful.

    ``true_obs=None`` (decoding without ground truth) returns (None, 0) —
    nothing downstream of the accept decision reads le.
    """
    if true_obs is None:
        return None, 0
    obs_pred = np.asarray(obs_pred, np.uint8)
    true_obs = np.asarray(true_obs, np.uint8)
    le = bool(np.any(obs_pred != true_obs))
    n_meas = int(_cfg().n_measured_observables)
    if n_meas <= 0:
        return le, 0
    m, u = _le_split_from_obs(obs_pred, true_obs, n_meas)
    return le, (int(m) | (int(u) << 1))


def _check_obs_classified(correction, true_obs):
    """``(le_bool, le_class, obs_pred)`` for a correction (predicts obs first).

    Also returns obs_pred so stage steps can surface the predicted
    observables without ground truth."""
    obs_pred = _predict_obs(correction)
    le, le_class = _classify_obs(obs_pred, true_obs)
    return le, le_class, obs_pred


# ---------------------------------------------------------------------------
# S3 relay cascade of BP stages (A → B → C), one in-memory function per shot.
#
#   S3-A: 5 standalone variants × 100 iterations
#   S3-B: 6 variants × relay-memory legs, coset quorum
#   S3-C: 12 independent BP configurations × {1, 10} seeds
#
# Every BP call decodes the loaded system with relevant-half acceptance (the
# _call_bp/_call_relay_bp wrappers re-test kernel-NC results against the
# relevant rows; residuals are relevant-row unsat counts). On a non-GARI
# system rel_rows covers every row, so this reduces to full-row acceptance.
# S3-C NC shots go to S4 (IP) — no posterior payloads.
# ---------------------------------------------------------------------------

def _stage_a_step(syndrome, true_obs, gid, nv, channel_llr, cfg):
    """S3-A: an ensemble of standalone BP variants, ``s3a_iters`` each.

    Returns ('solved', label, le, le_class, obs_pred) on the first converged
    variant, or ('nc', None, None, 0, None) if every variant fails."""
    alpha0 = _G['alpha_s3a']
    iters0 = cfg.s3a_iters
    for vi, (label, scale, noise_std, seed_offset) in enumerate(cfg.s3a_variants):
        # frozen seed tag — see the module docstring before touching it
        seed = _seed32(cfg.base_seed, "phase0", seed_offset, gid, vi)
        if noise_std > 0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, noise_std, size=nv).astype(np.float32)
            prior = np.ascontiguousarray(channel_llr * scale * (1.0 + noise))
        else:
            prior = np.ascontiguousarray(channel_llr * scale)
        conv, _residual, llr = _call_bp(syndrome, prior, iters0, alpha0, seed)
        if conv:
            correction = (llr < 0).astype(np.uint8)
            le, le_class, obs_pred = _check_obs_classified(correction, true_obs)
            return ('solved', f"S3A_{label}", le, le_class, obs_pred)
    return ('nc', None, None, 0, None)


def _stage_b_step(syndrome, true_obs, gid, nv, channel_llr, cfg):
    """S3-B: relay-BP with per-iteration memory and coset-quorum acceptance.

    ``s3b_use_memory=True`` (default) routes the S3-B variants through the
    relay-BP-with-memory kernel (:func:`_relay_bp_mem`).

      ``s3b_quorum > 1`` (default 3) — online first-to-quorum: converged legs
      vote for their coset in one tally shared across legs AND variants; the
      first coset to collect ``quorum`` votes is accepted immediately (later
      legs and variants never run) and the logical-error verdict is read off
      the winning coset. If no coset reaches quorum the shot is deferred to
      S3-C/S4 rather than committing a possibly-wrong coset.
      ``s3b_stop_nconv`` remains the per-variant convergence cap, bounding
      the work an ambiguous shot can consume.

      ``s3b_quorum == 1`` — pool every converged leg and accept the
      minimum-weight correction, so a heavier wrong-coset solution never
      wins over a lighter one present in the ensemble.

    ``s3b_use_memory=False`` selects the plain relay path in
    :func:`_stage_b_step_no_memory` (first-converged or min-cost acceptance).
    """
    if not getattr(cfg, 's3b_use_memory', False):
        return _stage_b_step_no_memory(syndrome, true_obs, gid, nv, channel_llr, cfg)

    alpha2 = _G['alpha_s3b']
    alpha0 = _schedule_alpha(cfg.s3b_gamma0_iters,
                             cfg.s3b_alpha_max, cfg.s3b_tau)
    quorum = max(1, int(cfg.s3b_quorum))
    stop_nconv = max(int(cfg.s3b_stop_nconv), quorum)

    def _variant_prior(pi, ps, noise_std):
        if noise_std > 0:
            rng_thm = np.random.default_rng(
                _seed32(cfg.base_seed, "phase2_noise", pi, gid))  # frozen tag
            noise = rng_thm.normal(0, noise_std, size=nv).astype(np.float32)
            return np.ascontiguousarray(channel_llr * ps * (1.0 + noise))
        return None

    if quorum > 1:
        # Online first-to-quorum across legs and variants.
        votes: dict = {}   # coset tuple -> [n_votes, first_tag]
        for pi, (ps, noise_std) in enumerate(cfg.s3b_variants):
            seed_base = _seed32(cfg.base_seed, "phase2_mem", pi)  # frozen tag
            bl = _variant_prior(pi, ps, noise_std)
            tag = f"S3B_p{ps}" if noise_std == 0 else f"S3B_p{ps}_t{noise_std}"
            _conv, _res, _llr, hit = _relay_bp_mem(
                syndrome, ps, cfg.s3b_runs, cfg.s3b_iters, alpha2,
                cfg.s3b_gamma0_iters, alpha0, cfg.s3b_gamma0,
                cfg.s3b_gamma_center, cfg.s3b_gamma_width,
                seed_base, gid, base_llr=bl, stop_nconv=stop_nconv,
                votes=votes, quorum=quorum, tag=tag)
            if hit is not None:
                # `hit` is the winning coset == the predicted observable vector.
                obs_pred = np.asarray(hit, dtype=np.uint8)
                le, le_class = _classify_obs(obs_pred, true_obs)
                return ('solved', votes[hit][1], le, le_class, obs_pred)
        return ('nc', None, None, 0, None)

    # quorum == 1 (b_quorum=1): pool every converged leg across all variants
    # ((cost, corr, coset, tag)), min-weight wins, accept unconditionally.
    pool = []
    for pi, (ps, noise_std) in enumerate(cfg.s3b_variants):
        seed_base = _seed32(cfg.base_seed, "phase2_mem", pi)  # frozen tag
        bl = _variant_prior(pi, ps, noise_std)
        converged, _res, _llr, _hit = _relay_bp_mem(
            syndrome, ps, cfg.s3b_runs, cfg.s3b_iters, alpha2,
            cfg.s3b_gamma0_iters, alpha0, cfg.s3b_gamma0,
            cfg.s3b_gamma_center, cfg.s3b_gamma_width,
            seed_base, gid, base_llr=bl, stop_nconv=stop_nconv)
        tag = f"S3B_p{ps}" if noise_std == 0 else f"S3B_p{ps}_t{noise_std}"
        for (corr, cost, coset) in converged:
            pool.append((cost, corr, coset, tag))

    if not pool:
        return ('nc', None, None, 0, None)

    pool.sort(key=lambda x: x[0])  # min-weight first
    best_cost, best_corr, best_coset, best_tag = pool[0]
    obs_pred = np.asarray(best_coset, dtype=np.uint8)
    le, le_class = _classify_obs(obs_pred, true_obs)
    return ('solved', best_tag, le, le_class, obs_pred)


def _stage_b_step_no_memory(syndrome, true_obs, gid, nv, channel_llr, cfg):
    """S3-B without relay memory: ``s3b_runs`` relay legs × ``s3b_iters``.

    Default: first-converged-variant wins. With ``s3b_min_cost_select``, runs
    every variant and accepts the minimum-LLR-cost converged correction.
    """
    alpha2 = _G['alpha_s3b']
    min_cost = bool(getattr(cfg, 's3b_min_cost_select', False))
    best_cost, best_corr, best_tag = np.inf, None, None
    for pi, (ps, noise_std) in enumerate(cfg.s3b_variants):
        seed_base = _seed32(cfg.base_seed, "phase2", pi)  # frozen tag
        if noise_std > 0:
            rng_thm = np.random.default_rng(
                _seed32(cfg.base_seed, "phase2_noise", pi, gid))  # frozen tag
            noise = rng_thm.normal(0, noise_std, size=nv).astype(np.float32)
            bl = np.ascontiguousarray(channel_llr * ps * (1.0 + noise))
        else:
            bl = None
        conv, correction, _res, _llr = _relay_bp(
            syndrome, ps, cfg.s3b_runs, cfg.s3b_iters,
            alpha2, seed_base, gid, base_llr=bl)
        if conv:
            tag = (f"S3B_p{ps}" if noise_std == 0
                   else f"S3B_p{ps}_t{noise_std}")
            if not min_cost:
                # Legacy: first-converged-variant wins.
                le, le_class, obs_pred = _check_obs_classified(
                    correction, true_obs)
                return ('solved', tag, le, le_class, obs_pred)
            cost = _correction_cost(correction)
            if cost < best_cost:
                best_cost, best_corr, best_tag = cost, correction, tag
            # no early return: evaluate all variants, keep min-cost
    if best_corr is not None:
        le, le_class, obs_pred = _check_obs_classified(best_corr, true_obs)
        return ('solved', best_tag, le, le_class, obs_pred)
    return ('nc', None, None, 0, None)


def _stage_c_step(syndrome, true_obs, gid, nv, channel_llr, cfg):
    """Try independent BP configurations sequentially and accept the first
    converged result.

    Noisy configurations use up to 10 seeds; a noise-free configuration uses
    one. Each attempt starts independently, without relay memory or a quorum
    requirement.

    Returns ('solved', label, le, le_class, obs_pred) or
    ('nc', None, None, 0, None).

    First-converged-wins, with no coset quorum. A quorum gate was tested
    here and removed: the one wrong-coset acceptance at this depth drew two
    independent agreeing wrong votes, so quorum=2 accepted it anyway, while
    55 correctly-solved lone-converger shots were needlessly deferred to S4.
    Corroborated-wrong shots this deep are only catchable by the exact
    solver. Deferred shots carry no posterior payload; S4 reads syndromes
    directly."""
    for vi, variant in enumerate(cfg.s3c_variants):
        label, ps, iters, noise_std, alpha_max, n_seeds_override = variant
        if n_seeds_override is not None:
            n_seeds = n_seeds_override
        else:
            n_seeds = cfg.s3c_n_seeds_default if noise_std > 0 else 1
        alpha = _schedule_alpha(iters, alpha_max, cfg.s3c_tau)
        for si in range(n_seeds):
            seed = _seed32(cfg.base_seed, "phase3_single", vi, si, gid)
            if noise_std > 0:
                rng = np.random.default_rng(seed)
                noise = rng.normal(0, noise_std, size=nv).astype(np.float32)
                prior = np.ascontiguousarray(channel_llr * ps * (1.0 + noise))
            else:
                prior = np.ascontiguousarray(channel_llr * ps)
            conv, _residual, llr = _call_bp(syndrome, prior, iters, alpha, seed)
            if conv:
                correction = (llr < 0).astype(np.uint8)
                le, le_class, obs_pred = _check_obs_classified(
                    correction, true_obs)
                return ('solved', f"S3C_{label}_s{si}", le, le_class, obs_pred)
    return ('nc', None, None, 0, None)


# ---------------------------------------------------------------------------
# Per-stage per-shot workers — each calls exactly one stage step. Pool
# workers call one of these. All return conv/le/label/obs_pred; S4 reads
# syndromes directly, so no posterior payloads are carried forward.
# ---------------------------------------------------------------------------

def _stage_one_shot(args, stage_letter):
    """Shared body for the S3-A/B/C single-shot workers.

    args = (syndrome_u8 (n_det,), true_obs_u8 (n_obs,) or None, global_idx).
    Returns one of:
      {global_idx, conv=True,  stage="S3A"|"S3B"|"S3C", label, le, le_class,
       obs_pred}
      {global_idx, conv=False, stage=...}               (NC; goes forward)
    """
    syndrome_u8, true_obs_u8, global_idx_i = args
    stage_name = f"S3{stage_letter.upper()}"
    try:
        if not _B:
            _init_buffers()
        syndrome = np.ascontiguousarray(np.asarray(syndrome_u8, dtype=np.uint8))
        true_obs = (np.asarray(true_obs_u8, dtype=np.uint8)
                    if true_obs_u8 is not None else None)
        gid = int(global_idx_i)

        if not syndrome.any():
            # Trivial syndrome only ever reaches S3-A in practice (samplers
            # don't filter), but handle it uniformly anyway. The zero
            # correction predicts all-zero observables, so the LE class is
            # read straight off true_obs (any set bit = a flip we failed to
            # catch).
            obs_pred_t = np.zeros(int(_G['L_csr'].shape[0]), dtype=np.uint8)
            _le_t, _cls_t = _classify_obs(obs_pred_t, true_obs)
            return {"global_idx": gid, "conv": True,
                    "stage": stage_name, "label": f"{stage_name}_trivial",
                    "le": _le_t, "le_class": int(_cls_t),
                    "obs_pred": obs_pred_t}

        nv = _G['n_vars']
        channel_llr = _G['channel_llr']
        cfg = _cfg()

        if stage_letter == 'a':
            status, label, le, le_class, obs_pred = _stage_a_step(
                syndrome, true_obs, gid, nv, channel_llr, cfg)
        elif stage_letter == 'b':
            status, label, le, le_class, obs_pred = _stage_b_step(
                syndrome, true_obs, gid, nv, channel_llr, cfg)
        else:  # 'c'
            status, label, le, le_class, obs_pred = _stage_c_step(
                syndrome, true_obs, gid, nv, channel_llr, cfg)

        if status == 'solved':
            return {"global_idx": gid, "conv": True,
                    "stage": stage_name, "label": label,
                    "le": (bool(le) if le is not None else None),
                    "le_class": int(le_class), "obs_pred": obs_pred}
        return {"global_idx": gid, "conv": False, "stage": stage_name}
    except Exception as e:
        print(f"[{stage_name} shot {global_idx_i}] crashed: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return {"global_idx": int(global_idx_i), "conv": False,
                "stage": stage_name, "label": f"{stage_name}_exception"}


def _s3_a_one_shot(args): return _stage_one_shot(args, 'a')
def _s3_b_one_shot(args): return _stage_one_shot(args, 'b')
def _s3_c_one_shot(args): return _stage_one_shot(args, 'c')
