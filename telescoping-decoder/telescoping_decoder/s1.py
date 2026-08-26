"""S1: fixed-iteration layered BP on the GPU.

Decodes every shot. Three decode systems (S1Config.system):
  gari      — GARI H with bottom-first row order and relevant-half acceptance
  original  — the correlated XYZ DEM, full-row convergence acceptance
  init_dets — the init-basis-detectors-only system, full-row acceptance

Input syndromes are always ``(B, n_det)`` uint8 in stim detector order;
system-specific padding, row permutation, and column gathers happen
at the decode boundary.
"""
from __future__ import annotations

import numpy as np


class Stage1:
    """Build the S1 decoder and apply its acceptance rules."""

    def __init__(self, system, cfg, resolved_system: str):
        """system: DecodingSystem; cfg: S1Config; resolved_system: one of
        "gari" | "original" | "init_dets" (already resolved, not "auto").

        Untouched S1 knobs are swapped for ``resolved_system``'s preset here
        rather than only in the facade, so a direct ``Stage1(...)`` build
        cannot pair the shipped init-dets knobs with the GARI graph (where
        hybrid_sp_iters=5 oscillates). Idempotent: the facade's swap leaves
        nothing for this one to do."""
        import warnings

        import scipy.sparse as sp

        from .config import s1_config_for
        from .kernels import S1LayeredBP

        cfg, note = s1_config_for(cfg, resolved_system)
        if note is not None:
            warnings.warn(note, stacklevel=2)

        self.cfg = cfg
        self.system_name = resolved_system
        init_idx = None
        if resolved_system == "init_dets":
            # Init-basis-detectors-only system (derive_init_det_system:
            # masked rows, columns merged by (detector, observable)
            # signature, XOR-combined priors). Acceptance = full-row
            # convergence, like the original system.
            (H, L, priors, init_idx, n_det, dc_pad, dv_pad) = \
                system.ensure_init_dets()
            m = int(H.shape[0])
            rel_rows = None
            H_rel = None
            acceptance_syndrome_cols = None
            H_decode = H
        else:
            npz = system.npz_path_for(resolved_system)
            mats = np.load(npz)
            H = sp.csr_matrix(
                (mats["h_data"], mats["h_indices"], mats["h_indptr"]),
                shape=tuple(mats["h_shape"]),
            )
            L = sp.csr_matrix(
                (mats["l_data"], mats["l_indices"], mats["l_indptr"]),
                shape=tuple(mats["l_shape"]),
            )
            priors = mats["probs"]
            dc_pad = int(mats["dc_pad"])
            dv_pad = int(mats["dv_pad"])
            m = int(H.shape[0])
            if resolved_system == "gari":
                # GARI: detector rows come first; the consistency rows are
                # decoded against zero syndrome bits. Input syndromes stay
                # n_det wide; only the decode input is padded.
                n_det = int(mats["gari_n_detectors"])
                # Relevant-half acceptance (the GARI accept rule): accept iff
                # H[rel_rows] @ corr == s[rel_rows]  (D_X·ē_Z = s_X) — NOT the
                # decoder's full-row convergence, which additionally waits for
                # the auxiliary consistency rows and the unused half (kept
                # only as the full_conv diagnostic stat). rel_rows ⊂ detector
                # rows, so the syndrome bits come from the unpadded batch.
                rel_rows = mats["gari_relevant_rows"].astype(np.int64)
                H_rel = sp.csr_matrix(H[rel_rows])

                # Bottom-first row order for S1's contiguous layered
                # blocking: [U | V | detectors]. The consistency rows
                # propagate the e-block priors into the ē variables before
                # the detector rows are processed (the npz stores
                # [detectors | U | V]; detector syndrome bits therefore go
                # in the trailing m-n_det..m columns of s_full). Corrections
                # are column-space, so the acceptance SpMM (H_rel) and L̄
                # are unaffected.
                row_perm = np.concatenate(
                    [np.arange(n_det, m), np.arange(0, n_det)]).astype(np.int64)
                H_decode = sp.csr_matrix(H[row_perm])
                acceptance_syndrome_cols = m - n_det + rel_rows
            else:
                # Original system: decode
                # H as-is, acceptance = full-row convergence.
                n_det = m
                rel_rows = None
                H_rel = None
                acceptance_syndrome_cols = None
                H_decode = H
        channel_llr = np.log((1.0 - priors) / priors).astype(np.float32)

        decoder = S1LayeredBP(
            H_decode, channel_llr,
            schedule="layered",
            n_iters=cfg.n_iters,
            alpha_schedule="ramp",
            alpha_max=cfg.alpha_max,
            alpha_tau=cfg.alpha_tau,
            K=cfg.k,
            algorithm=cfg.algorithm,
            hybrid_sp_iters=cfg.hybrid_sp_iters,
            fast_var_resync=True,
            var_nwy=cfg.var_nwy,
            n_legs=1,
            dc_pad=dc_pad, dv_pad=dv_pad,
            obs_matrix=L,
            acceptance_matrix=H_rel,
            acceptance_syndrome_cols=acceptance_syndrome_cols,
        )

        self.decoder = decoder
        self.n_obs = int(L.shape[0])
        self.m = m
        self.n_det = n_det
        self.init_idx = init_idx
        self.rel_rows = rel_rows

    def decode(self, syndromes: np.ndarray, true_obs=None) -> dict:
        """Decode a batch. syndromes: (B, n_det_full) uint8, stim detector
        order (full sampled width — for init_dets the gather happens here).

        Returns dict of per-shot arrays: accepted (B,) bool, obs_pred
        (B, n_obs) uint8 (meaningful where accepted), le (B,) bool or None,
        n_full_conv int (diagnostic), decode_s float.
        """
        import time
        import cupy as cp

        cfg = self.cfg
        m, n_det = self.m, self.n_det
        B_total = int(syndromes.shape[0])
        accepted = np.zeros(B_total, dtype=bool)
        obs_pred_all = np.zeros((B_total, self.n_obs), dtype=np.uint8)
        le_all = np.zeros(B_total, dtype=bool) if true_obs is not None else None
        n_full_conv = 0
        total_decode_s = 0.0

        bs = int(cfg.shots_per_batch)
        for batch_start in range(0, B_total, bs):
            batch_end = min(batch_start + bs, B_total)
            B_actual = batch_end - batch_start
            s_actual = np.ascontiguousarray(
                syndromes[batch_start:batch_end], dtype=np.uint8)
            # Constant decode-batch shape (bs rows) keeps the decoder on one
            # cached CUDA graph; the tail batch is zero-padded rows.
            s_full = np.zeros((bs, m), dtype=np.uint8)
            if self.init_idx is not None:
                # init-dets-only: decode input = init-family column gather
                # of the full-width syndrome.
                s_full[:B_actual] = s_actual[:, self.init_idx]
            else:
                # GARI bottom-first: detector rows are the last n_det rows
                # of the permuted H, so their bits go in the trailing columns
                # and the leading consistency-row bits are zero. Original
                # mode: n_det == m, so the slice is the whole row — same
                # line works.
                s_full[:B_actual, m - n_det:] = s_actual

            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            out = self.decoder.run(s_full, n_valid=B_actual)
            cp.cuda.Stream.null.synchronize()
            total_decode_s += time.perf_counter() - t0

            full_conv = np.asarray(out["converged"], dtype=bool)[:B_actual]
            n_full_conv += int(full_conv.sum())

            obs_pred = np.asarray(
                out["obs_pred"][:B_actual], dtype=np.uint8)

            if self.system_name == "gari":
                # The low-level decoder evaluates GARI's relevant-half rule
                # directly from its device correction.
                conv = np.asarray(
                    out["relevant_converged"][:B_actual], dtype=bool)
            else:
                # Original / init-dets-only system: acceptance = full-row
                # convergence on the decoded H.
                conv = full_conv

            accepted[batch_start:batch_end] = conv
            obs_pred_all[batch_start:batch_end] = obs_pred
            if true_obs is not None:
                o_b = np.asarray(
                    true_obs[batch_start:batch_end], dtype=np.uint8)
                le_all[batch_start:batch_end] = np.any(
                    obs_pred != o_b, axis=1)

        return {
            "accepted": accepted,
            "obs_pred": obs_pred_all,
            "le": le_all,
            "n_full_conv": n_full_conv,
            "decode_s": total_decode_s,
        }


__all__ = ["Stage1"]
