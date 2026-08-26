"""S2: relay-BP with coset-quorum acceptance on the GPU.

Decodes the S1 rejects. Three decode systems (S2Config.system):
  gari      — the GARI relay decoder with relevant-row convergence,
              answer-block weights, and bottom-first [U][V][detector] layers
  original  — the plain relay decoder on the correlated XYZ DEM, full-row
              convergence (the bundled benchmark uses quorum 2 here)
  init_dets — the plain relay decoder on the init-detectors-only system

The decoder's "converged" output represents coset-quorum acceptance:
accept iff at least ``quorum`` (half-)converged legs collected identical
cosets.
"""
from __future__ import annotations

import numpy as np


class Stage2:
    """Build the S2 decoder and extract its observable predictions."""

    def __init__(self, system, cfg, resolved_system: str):
        """system: DecodingSystem; cfg: S2Config; resolved_system: one of
        "gari" | "original" | "init_dets" (already resolved, not "auto")."""
        import scipy.sparse as sp

        self.cfg = cfg
        self.system_name = resolved_system
        t2_init_idx = None
        mats = None
        if resolved_system == "init_dets":
            # Init-basis-detectors-only system, derived from the original
            # matrices (same derive_init_det_system as S1's init_dets mode:
            # masked rows, columns merged by (detector, observable)
            # signature, XOR-combined priors). Full-row convergence on
            # the small system; the full-width syndrome is column-gathered
            # to the init family before decode. n_det stays the full
            # sampled-DEM detector width; decode width m == len(init_idx).
            (H, L, priors, t2_init_idx, n_det, dc_pad, dv_pad) = \
                system.ensure_init_dets()
            channel_llr = np.log((1.0 - priors) / priors).astype(np.float32)
            m = int(H.shape[0])
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
            channel_llr = np.log((1.0 - priors) / priors).astype(np.float32)
            m = int(H.shape[0])
        if resolved_system == "gari":
            # GARI: input syndromes are n_det wide; the decode input is
            # padded to m with zeros for the consistency rows.
            n_det = int(mats["gari_n_detectors"])
            # GARI-only inputs to S2RelayBPGari:
            #   conv_rows  — per-leg convergence over the relevant rows only
            #   weight_llr — min-weight selection over the answer block only
            #   layers     — bottom-first [U][V][detector-row blocks]
            rel_rows = mats["gari_relevant_rows"].astype(np.int32)
            rel_p = np.clip(mats["gari_relevant_priors"].astype(np.float64),
                            1e-300, 0.5 - 1e-12)
            cbb = mats["gari_col_block_bounds"]
            _bi = {"eZ": 0, "eX": 1, "eY": 2, "ebarZ": 3, "ebarX": 4}[
                str(mats["gari_answer_block"])]
            weight_llr = np.zeros(int(H.shape[1]), dtype=np.float32)
            weight_llr[int(cbb[_bi]):int(cbb[_bi + 1])] = np.abs(
                np.log((1.0 - rel_p) / rel_p)).astype(np.float32)
            # gari_layers rows: [U-range, V-range, detector-range] in
            # processing order; split the detector range into k blocks.
            g_layers = mats["gari_layers"]
            det_lo, det_hi = int(g_layers[2][0]), int(g_layers[2][1])
            n_det_rows_range = det_hi - det_lo
            kk = max(1, min(int(cfg.k), n_det_rows_range))
            bounds = np.linspace(det_lo, det_hi, kk + 1, dtype=np.int64)
            layers = ([(int(g_layers[0][0]), int(g_layers[0][1])),
                       (int(g_layers[1][0]), int(g_layers[1][1]))]
                      + [(int(bounds[i]), int(bounds[i + 1]))
                         for i in range(kk) if bounds[i + 1] > bounds[i]])
        elif resolved_system == "original":
            # Original system: full-row convergence, no padding
            # (n_det == m).
            n_det = m
        # init_dets: n_det (full sampled-DEM width) already set above by
        # ensure_init_dets; decode width m == len(init_idx) < n_det.

        if resolved_system == "gari":
            from .kernels import S2RelayBPGari, S2RelayPhaseGari
            phase = S2RelayPhaseGari(
                "S2_gari", list(cfg.priors) or [(1.0, 0.0)],
                num_sets=cfg.num_sets,
                leg_max_iter=cfg.leg_max_iter,
                gamma0=cfg.gamma0, gamma0_iters=cfg.gamma0_iters,
                gamma_center=cfg.gamma_center,
                gamma_width=cfg.gamma_width,
                alpha=cfg.alpha, alpha_const=cfg.alpha_const,
                tau=cfg.tau, quorum=cfg.quorum,
            )
            decoder = S2RelayBPGari(
                H, channel_llr, obs_matrix=L, K=cfg.k, phase=phase,
                batch_size=cfg.batch_size,
                num_slots=cfg.num_slots,
                prebuild_bundles=False, dc_pad=dc_pad, dv_pad=dv_pad,
                conv_rows=rel_rows, weight_llr=weight_llr, layers=layers,
            )
        else:
            from .kernels import S2RelayBP, S2RelayPhase
            phase = S2RelayPhase(
                "S2", list(cfg.priors) or [(1.0, 0.0)],
                num_sets=cfg.num_sets,
                leg_max_iter=cfg.leg_max_iter,
                gamma0=cfg.gamma0, gamma0_iters=cfg.gamma0_iters,
                gamma_center=cfg.gamma_center,
                gamma_width=cfg.gamma_width,
                alpha=cfg.alpha, alpha_const=cfg.alpha_const,
                tau=cfg.tau, quorum=cfg.quorum,
            )
            decoder = S2RelayBP(
                H, channel_llr, obs_matrix=L, K=cfg.k, phase=phase,
                batch_size=cfg.batch_size,
                num_slots=cfg.num_slots,
                prebuild_bundles=False, dc_pad=dc_pad, dv_pad=dv_pad,
            )

        self.decoder = decoder
        self.n_obs = int(L.shape[0])
        self.m = m
        self.n_det = n_det
        self.init_idx = t2_init_idx
        # introspection handle for tests (GARI builder arrays)
        self.gari_arrays = (
            {"conv_rows": rel_rows, "weight_llr": weight_llr,
             "layers": layers}
            if resolved_system == "gari" else None)

    def decode(self, syndromes: np.ndarray, true_obs=None) -> dict:
        """Decode a batch of S1 rejects. syndromes: (B, n_det_full) uint8,
        stim detector order (full sampled width).

        Returns dict of per-shot arrays: accepted (B,) bool, obs_pred
        (B, n_obs) uint8 (meaningful where accepted), le (B,) bool or None,
        decode_s float.
        """
        import time
        import cupy as cp

        cfg = self.cfg
        m, n_det = self.m, self.n_det
        n_in = int(syndromes.shape[0])
        accepted = np.zeros(n_in, dtype=bool)
        obs_pred_all = np.zeros((n_in, self.n_obs), dtype=np.uint8)
        le_all = np.zeros(n_in, dtype=bool) if true_obs is not None else None

        # The driver feeds the decoder slices of `outer_batch`, which is
        # then re-chunked internally by the decoder's own batch size. If the
        # inner batch exceeds the outer slice it is silently clamped to the
        # slice. Widen the outer slice to at least the configured inner
        # batch so the requested batch size is actually realized.
        outer_batch = max(int(cfg.shots_per_batch), int(cfg.batch_size))

        t_loop = time.perf_counter()
        i = 0
        while i < n_in:
            j = min(i + outer_batch, n_in)
            batch_synd = np.ascontiguousarray(
                syndromes[i:j], dtype=np.uint8)

            B_actual = batch_synd.shape[0]
            if self.init_idx is not None:
                # init-dets-only: decode input = init-family column gather
                # of the full-width syndrome (m == len(init_idx) < n_det).
                synd_p = np.zeros((outer_batch, m), dtype=np.uint8)
                synd_p[:B_actual] = batch_synd[:, self.init_idx]
            elif B_actual < outer_batch or n_det < m:
                # zero rows pad the batch; zero columns are the GARI
                # consistency-row syndrome bits
                synd_p = np.zeros((outer_batch, m), dtype=np.uint8)
                synd_p[:B_actual, :n_det] = batch_synd
            else:
                synd_p = batch_synd

            cp.cuda.Stream.null.synchronize()
            out = self.decoder.run(synd_p)
            cp.cuda.Stream.null.synchronize()
            conv = np.asarray(out["converged"], dtype=bool)[:B_actual]
            obs_pred = np.asarray(
                out["obs_pred"][:B_actual], dtype=np.uint8)

            accepted[i:j] = conv
            obs_pred_all[i:j] = obs_pred
            if true_obs is not None:
                batch_obs = np.asarray(true_obs[i:j], dtype=np.uint8)
                le_all[i:j] = conv & np.any(
                    obs_pred != batch_obs, axis=1)
            i = j

        return {
            "accepted": accepted,
            "obs_pred": obs_pred_all,
            "le": le_all,
            "decode_s": time.perf_counter() - t_loop,
        }


__all__ = ["Stage2"]
