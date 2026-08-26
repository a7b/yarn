"""Public interface for staged decoding.

Routes each shot through the telescoping stages S1 (GPU flooding BP) →
S2 (GPU relay-BP, coset quorum) → S3-A/B/C (CPU relay stages) → S4 (Gurobi
IP certification), each stage decoding only the previous stage's rejects.

Typical use::

    from telescoping_decoder import TelescopingDecoder

    dec = TelescopingDecoder.from_stim_circuit(circuit, init_basis="X")
    result = dec.decode(syndromes)          # (B, n_det) uint8, stim order
    print(result.summary())

S1 is deterministic. S3 derives its random seeds from ``base_seed`` and
``shot_id``, so its results do not depend on pool scheduling. S2 derives its
random schedule from the input batch position; repeatability therefore also
requires the same input order and S2 batching configuration.
"""
from __future__ import annotations

import dataclasses
import enum
import multiprocessing as mp
import os
import time
import warnings
from typing import Optional

import numpy as np

from . import s3 as s3mod
from .config import TelescopeConfig, s1_config_for
from .s4_ip import _ip_pool_init, _solve_one_ip_shot
from .system import DecodingSystem

# s1/s2 are imported lazily inside the stage getters: importing them pulls in
# cupy, which must not be required for a CPU-only install.


class Stage(enum.IntEnum):
    """Which stage accepted a shot. NC = no stage accepted (deferred)."""
    NC = 0
    S1 = 1
    S2 = 2
    S3A = 3
    S3B = 4
    S3C = 5
    S4 = 6


def _s3_worker_init(flat_cfg, npz_path):
    from . import s3
    s3._install_cfg(flat_cfg)
    s3._ensure_init(npz_path)
    s3._pool_init()


@dataclasses.dataclass
class DecodeResult:
    """Struct-of-arrays result for one decode() call (B shots).

    ``obs_pred`` is the zero vector for NC shots (the deferred shot's
    implicit prediction). ``le`` is present only when ``true_obs`` was
    given; NC shots are scored against the zero prediction.
    """
    obs_pred: np.ndarray          # (B, n_obs) uint8
    stage: np.ndarray             # (B,) int8 of Stage codes
    label: np.ndarray             # (B,) object str: winning variant / status
    converged: np.ndarray         # (B,) bool == (stage != NC)
    le: Optional[np.ndarray]      # (B,) bool, or None if no true_obs
    diagnostics: dict

    @property
    def ler(self) -> float:
        """Logical error rate over these shots (needs ``true_obs``)."""
        if self.le is None:
            raise ValueError("ler needs true_obs passed to decode()")
        return float(self.le.mean()) if self.le.size else 0.0

    def counts_by_stage(self) -> dict:
        """``{stage name: shots accepted}``, NC included."""
        return {Stage(s).name: int(n) for s, n in
                zip(*np.unique(self.stage, return_counts=True))}

    def summary(self) -> str:
        """Human-readable per-stage acceptance table (+ LER when scored)."""
        B = self.stage.shape[0]
        lines = [f"shots: {B}"]
        counts = self.counts_by_stage()
        for name in ("S1", "S2", "S3A", "S3B", "S3C", "S4", "NC"):
            if name in counts:
                n = counts[name]
                lines.append(f"  {name:<4} accepted {n:>10}  "
                             f"({n / max(B, 1):8.3%})")
        if self.le is not None:
            lines.append(f"logical errors: {int(self.le.sum())}  "
                         f"(LER {self.ler:.3e})")
        return "\n".join(lines)


class TelescopingDecoder:
    """Route syndrome batches through the configured decoder stages."""

    def __init__(self, system: DecodingSystem,
                 config: Optional[TelescopeConfig] = None):
        self.system = system
        self.config = config or TelescopeConfig()
        self._prepared = False
        self._flat_cfg = None
        self._resolved: dict = {}
        self._s3_npz = None
        self._s4_npz = None
        self._s4_enabled = False
        self._warned_no_ip = False
        self._warned_no_license = False
        self._ctx = None
        self._s1_cfg = None       # config.s1, or its per-system preset
        self._s1 = None
        self._s2 = None
        self._s3_pool = None
        self._s4_pool = None

    # ------------------------------------------------------------------ #
    # constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_stim_circuit(cls, circuit, *, init_basis: str,
                          config: Optional[TelescopeConfig] = None,
                          verify: bool = True, verify_samples: int = 1024,
                          workdir=None) -> "TelescopingDecoder":
        """Construct from a Stim circuit and infer detector types from it."""
        config = config or TelescopeConfig()
        system = DecodingSystem.from_stim_circuit(
            circuit, init_basis=init_basis, use_gari=config.use_gari,
            verify=verify, verify_samples=verify_samples, workdir=workdir)
        return cls(system, config)

    @classmethod
    def from_dem(cls, dem, *, is_x_detector=None, canonical_layout=None,
                 init_basis: Optional[str] = None,
                 config: Optional[TelescopeConfig] = None,
                 verify: bool = True, verify_samples: int = 1024,
                 workdir=None) -> "TelescopingDecoder":
        """Construct from a Stim detector error model.

        Load a file with ``stim.DetectorErrorModel.from_file`` before calling
        this method. GARI also requires ``is_x_detector`` or
        ``canonical_layout``.
        """
        config = config or TelescopeConfig()
        system = DecodingSystem.from_dem(
            dem, is_x_detector=is_x_detector,
            canonical_layout=canonical_layout, init_basis=init_basis,
            use_gari=config.use_gari, verify=verify,
            verify_samples=verify_samples, workdir=workdir)
        return cls(system, config)

    @classmethod
    def from_matrices(cls, H, L, priors, *, is_x_detector=None,
                      init_basis: Optional[str] = None,
                      config: Optional[TelescopeConfig] = None,
                      workdir=None) -> "TelescopingDecoder":
        """Raw check/observable matrices: the original-system stages
        (no GARI — it needs a stim DEM)."""
        system = DecodingSystem.from_matrices(
            H, L, priors, is_x_detector=is_x_detector,
            init_basis=init_basis, workdir=workdir)
        return cls(system, config)

    @classmethod
    def from_npz(cls, *, gari_npz=None, matrices_npz=None,
                 config: Optional[TelescopeConfig] = None,
                 workdir=None) -> "TelescopingDecoder":
        """Construct from NPZ artifacts materialized by ``prepare()``.

        Skips the DEM parse and the GARI transform entirely."""
        system = DecodingSystem.from_npz(
            gari_npz=gari_npz, matrices_npz=matrices_npz, workdir=workdir)
        return cls(system, config)

    # ------------------------------------------------------------------ #
    # preparation                                                         #
    # ------------------------------------------------------------------ #

    def _resolve_system(self, requested: str, stage: str) -> str:
        if requested == "gari" and not self.config.use_gari:
            raise ValueError(
                f"{stage}.system='gari' but config.use_gari=False")
        return self.system.resolve(requested, stage,
                                   use_gari=self.config.use_gari)

    def prepare(self) -> "TelescopingDecoder":
        """Prepare stage systems, artifacts, worker support, and C kernels.

        This method is idempotent. GPU decoders are constructed by the first
        call to :meth:`decode`.
        """
        if self._prepared:
            return self
        cfg = self.config

        # Pre-spawn the multiprocessing context BEFORE any CUDA work so
        # forkserver children never inherit a CUDA context.
        self._ctx = mp.get_context(cfg.s3.mp_start_method)
        if cfg.s3.mp_start_method == "forkserver":
            try:
                from multiprocessing import forkserver as _fs
                _fs._forkserver.ensure_running()
            except Exception:
                pass

        self._resolved = {}
        if cfg.s1.enabled:
            resolved_s1 = self._resolve_system(cfg.s1.system, "s1")
            self._resolved["s1"] = resolved_s1
            # S1Config's knob defaults belong to one system; if S1 landed on
            # another, take that system's preset instead of a mixed pairing.
            self._s1_cfg, note = s1_config_for(cfg.s1, resolved_s1)
            if note is not None:
                warnings.warn(note, stacklevel=2)
        if cfg.s2.enabled:
            self._resolved["s2"] = self._resolve_system(cfg.s2.system, "s2")
        if cfg.s3.enabled:
            s3_sys = self._resolve_system(cfg.s3.system, "s3")
            self._resolved["s3"] = s3_sys
            self._s3_npz = self.system.npz_path_for(s3_sys)
            # compile the C kernels now so failures surface before decoding
            from ._c.build import ensure_lib
            ensure_lib("checkserial_bp")
            ensure_lib("relay_mem_bp")

        # S4 availability
        s4 = cfg.s4
        if s4.enabled in (True, "auto"):
            import importlib.util
            have_grb = importlib.util.find_spec("gurobipy") is not None
            if s4.enabled is True and not have_grb:
                raise ImportError(
                    "s4.enabled=True but the core gurobipy dependency is "
                    "unavailable; reinstall telescoping-decoder (and provide "
                    "a full Gurobi license), or set s4.enabled='auto'/False")
            self._s4_enabled = have_grb
        else:
            self._s4_enabled = False
        if self._s4_enabled:
            if not self.system.has_original:
                raise ValueError(
                    "S4 (IP) solves the original DEM; construct with "
                    "matrices_npz=/from_dem/from_matrices, or disable s4")
            self._s4_npz = self.system.npz_path_for(
                "init_dets" if s4.init_dets_only else "original")

        self._flat_cfg = cfg._flatten()
        self._prepared = True
        return self

    # ------------------------------------------------------------------ #
    # stage lazies                                                        #
    # ------------------------------------------------------------------ #

    def _get_s1(self):
        if self._s1 is None:
            from .s1 import Stage1
            self._s1 = self._build_gpu_stage(
                Stage1, self._s1_cfg or self.config.s1, self._resolved["s1"])
        return self._s1

    def _get_s2(self):
        if self._s2 is None:
            from .s2 import Stage2
            self._s2 = self._build_gpu_stage(
                Stage2, self.config.s2, self._resolved["s2"])
        return self._s2

    def _build_gpu_stage(self, klass, stage_cfg, resolved):
        try:
            return klass(self.system, stage_cfg, resolved)
        except ImportError as e:
            raise RuntimeError(
                "S1/S2 need a CUDA GPU and the package's [gpu] extra; "
                "install from the checkout with pip install -e '.[gpu]'. "
                "To run CPU-only, set "
                "config.s1.enabled=False and config.s2.enabled=False"
            ) from e

    def _get_s3_pool(self):
        if self._s3_pool is None:
            n = self.config.s3.n_procs or os.cpu_count() or 1
            self._s3_pool = self._ctx.Pool(
                n, initializer=_s3_worker_init,
                initargs=(self._flat_cfg, self._s3_npz))
        return self._s3_pool

    def _get_s4_pool(self):
        if self._s4_pool is None:
            n = self.config.s4.n_procs or os.cpu_count() or 1
            self._s4_pool = self._ctx.Pool(
                n, initializer=_ip_pool_init,
                initargs=(self._flat_cfg, self._s4_npz,
                          self.config.s4.license_file))
        return self._s4_pool

    # ------------------------------------------------------------------ #
    # decode                                                              #
    # ------------------------------------------------------------------ #

    def decode(self, syndromes, true_obs=None, shot_ids=None) -> DecodeResult:
        """Decode a batch of syndromes through the telescoping stages.

        syndromes: (B, n_det) uint8/bool, stim detector order.
        true_obs:  optional (B, n_obs) uint8 ground truth — enables le/LER;
                   never influences decoding.
        shot_ids:  optional (B,) unique uint64 identifiers used to seed S3
                   variants (default arange(B)). S2 uses batch-position seeds
                   instead; see the README reproducibility notes.
        """
        self.prepare()
        syndromes = np.ascontiguousarray(np.asarray(syndromes), dtype=None)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.system.n_detectors:
            raise ValueError(
                f"syndromes must be (B, {self.system.n_detectors}) in stim "
                f"detector order; got {syndromes.shape}")
        B = int(syndromes.shape[0])
        n_obs = self.system.n_obs
        if true_obs is not None:
            true_obs = np.asarray(true_obs, dtype=np.uint8)
            if true_obs.shape != (B, n_obs):
                raise ValueError(
                    f"true_obs must be (B={B}, {n_obs}); got {true_obs.shape}")
        if shot_ids is None:
            shot_ids = np.arange(B, dtype=np.uint64)
        else:
            shot_ids = np.asarray(shot_ids, dtype=np.uint64)
            if shot_ids.shape != (B,):
                raise ValueError(f"shot_ids must be ({B},)")
            if np.unique(shot_ids).size != B:
                raise ValueError("shot_ids must be unique")

        stage = np.zeros(B, dtype=np.int8)
        label = np.array([""] * B, dtype=object)
        obs_pred = np.zeros((B, n_obs), dtype=np.uint8)
        le = np.zeros(B, dtype=bool) if true_obs is not None else None
        diagnostics: dict = {"n_shots": B}
        remaining = np.arange(B)

        # ---- S1 / S2 (GPU) --------------------------------------------- #
        for name, getter, code in (("s1", self._get_s1, Stage.S1),
                                   ("s2", self._get_s2, Stage.S2)):
            if not getattr(self.config, name).enabled or remaining.size == 0:
                continue
            st = getter()
            r = st.decode(syndromes[remaining],
                          true_obs[remaining] if true_obs is not None
                          else None)
            acc = r["accepted"]
            acc_rows = remaining[acc]
            stage[acc_rows] = code
            label[acc_rows] = code.name
            obs_pred[acc_rows] = r["obs_pred"][acc]
            if le is not None:
                le[acc_rows] = r["le"][acc]
            diagnostics[name] = {
                "system": self._resolved[name],
                "n_in": int(remaining.size),
                "n_accepted": int(acc.sum()),
                "decode_s": r["decode_s"],
            }
            if name == "s1":
                diagnostics[name]["n_full_conv"] = r["n_full_conv"]
            remaining = remaining[~acc]

        # ---- S3-A/B/C (CPU pool) ---------------------------------------- #
        if self.config.s3.enabled and remaining.size > 0:
            pool = self._get_s3_pool()
            row_of = {int(shot_ids[i]): i for i in remaining}
            items = [
                (syndromes[i],
                 true_obs[i] if true_obs is not None else None,
                 int(shot_ids[i]))
                for i in remaining
            ]
            for key, fn, code, chunksize in (
                    ("s3a", s3mod._s3_a_one_shot, Stage.S3A, 16),
                    ("s3b", s3mod._s3_b_one_shot, Stage.S3B, 1),
                    ("s3c", s3mod._s3_c_one_shot, Stage.S3C, 1)):
                if not items:
                    break
                t0 = time.perf_counter()
                n_in = len(items)
                for r in pool.imap_unordered(fn, items,
                                             chunksize=chunksize):
                    row = row_of[int(r["global_idx"])]
                    if r["conv"]:
                        stage[row] = code
                        label[row] = r.get("label", "")
                        if "obs_pred" in r:
                            obs_pred[row] = r["obs_pred"]
                        if le is not None and r.get("le") is not None:
                            le[row] = r["le"]
                # keep only this stage's NC shots, in input order (the next
                # stage's per-shot decode is order-independent anyway)
                items = [it for it in items
                         if stage[row_of[it[2]]] == Stage.NC]
                diagnostics[key] = {
                    "n_in": n_in,
                    "n_accepted": n_in - len(items),
                    "wall_s": time.perf_counter() - t0,
                }
            remaining = np.array(
                [row_of[it[2]] for it in items], dtype=int)
            if "s3a" in diagnostics:
                diagnostics["s3"] = {"system": self._resolved["s3"]}

        # ---- S4 (IP) ----------------------------------------------------- #
        if remaining.size > 0 and self._s4_enabled:
            pool = self._get_s4_pool()
            row_of = {int(shot_ids[i]): i for i in remaining}
            items = [
                (syndromes[i],
                 true_obs[i] if true_obs is not None else None,
                 int(shot_ids[i]))
                for i in remaining
            ]
            t0 = time.perf_counter()
            status_hist: dict = {}
            n_ok = 0
            still_nc = []
            for r in pool.imap_unordered(_solve_one_ip_shot, items,
                                         chunksize=1):
                row = row_of[int(r["global_idx"])]
                status_hist[str(r.get("status"))] = (
                    status_hist.get(str(r.get("status")), 0) + 1)
                if r["ok"]:
                    n_ok += 1
                    stage[row] = Stage.S4
                    label[row] = f"IP_{r['stage']}_{r['status']}"
                    if r.get("obs_pred") is not None:
                        obs_pred[row] = r["obs_pred"]
                    if le is not None and r.get("le") is not None:
                        le[row] = r["le"]
                else:
                    status = str(r.get("status"))
                    label[row] = ("IP_no_license" if status.startswith("LICENSE_")
                                  else "IP_giveup_uncertified")
                    still_nc.append(row)
            n_license = sum(n for s, n in status_hist.items()
                            if s.startswith("LICENSE_"))
            if n_license and not self._warned_no_license:
                self._warned_no_license = True
                warnings.warn(
                    f"S4 could not run on {n_license} shot(s): Gurobi "
                    f"reported {sorted(s for s in status_hist if s.startswith('LICENSE_'))}. "
                    "Those shots are uncertified deferrals (NC), not proven "
                    "logical errors — a shot the IP never solved is scored "
                    "against the zero prediction. Point config.s4.license_file "
                    "at a full Gurobi license (the pip-bundled restricted "
                    "license caps models at 2000 variables) or set "
                    "s4.enabled=False to stop attempting S4.", stacklevel=2)
            diagnostics["s4"] = {
                "n_in": len(items),
                "n_ok": n_ok,
                "n_giveup_uncertified": len(still_nc) - n_license,
                "n_no_license": n_license,
                "status_hist": status_hist,
                "wall_s": time.perf_counter() - t0,
            }
            remaining = np.array(still_nc, dtype=int)
        elif remaining.size > 0:
            label[remaining] = "NC_no_ip"
            if not self._warned_no_ip and self.config.s4.enabled == "auto":
                self._warned_no_ip = True
                warnings.warn(
                    f"{remaining.size} shot(s) deferred without the S4 IP "
                    "stage because the core gurobipy dependency is "
                    "unavailable; reinstall telescoping-decoder and provide "
                    "a full Gurobi license to certify them", stacklevel=2)

        # NC shots keep the zero prediction; score them against it.
        if le is not None and remaining.size > 0:
            le[remaining] = true_obs[remaining].any(axis=1)
        diagnostics["n_nc"] = int(remaining.size)

        converged = stage != Stage.NC
        return DecodeResult(obs_pred=obs_pred, stage=stage, label=label,
                            converged=converged, le=le,
                            diagnostics=diagnostics)

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Shut down the S3 and S4 worker pools.

        The method is idempotent. A later call to :meth:`decode` creates new
        pools as needed.
        """
        for pool_attr in ("_s3_pool", "_s4_pool"):
            pool = getattr(self, pool_attr)
            if pool is not None:
                pool.terminate()
                pool.join()
                setattr(self, pool_attr, None)

    def __enter__(self) -> "TelescopingDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["TelescopingDecoder", "DecodeResult", "Stage"]
