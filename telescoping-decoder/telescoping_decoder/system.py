"""Construction and storage of the supported decoding systems.

The container holds the original correlated XYZ system (H, L, priors) plus, when
available, the GARI-transformed system (a :class:`GariModel`) and the
init-basis-detectors-only system derived on demand. ``materialize()``
writes each system to an npz in the workdir, in the
``<stem>_matrices.npz`` / ``<stem>_gari_matrices.npz`` format documented in
the README, because the stage builders and the S3/S4 worker processes
initialize by loading those files (``s3._ensure_init(path)``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .gari import GariModel, gari_transform, load_gari_npz, save_gari_npz, \
    xz_detector_type_mask
from ._fingerprint import source_fingerprint
from .stim_mask import circuit_to_dem, circuit_xz_detector_mask

SYSTEMS = ("gari", "original", "init_dets")

# What ``system="auto"`` walks, per stage. S1 prefers the init-basis-detector
# system: it is far smaller than GARI (660x9600 vs 19575x94620 on the bundled
# [[150,30,10]] artifacts) and accepts more shots, so the expensive stages see
# less traffic. S2/S3 prefer GARI, whose relevant-half acceptance is what
# their quorum rules are tuned against. Candidates the decoder cannot build
# are skipped; an explicitly requested system is never skipped, it raises.
_AUTO_CHAINS = {
    "s1": ("init_dets", "gari", "original"),
    "s2": ("gari", "original"),
    "s3": ("gari", "original"),
}
_DEFAULT_AUTO_CHAIN = ("gari", "original")

_MASK_HELP = (
    "GARI needs every detector classified as X- or Z-type, and a DEM alone "
    "does not carry this. Either (a) pass is_x_detector= (bool array, one "
    "entry per detector, True = X-type), (b) pass canonical_layout=(n_x, "
    "n_z, num_rounds) if your circuit emits X-ancilla then Z-ancilla "
    "detectors per round (the canonical order), (c) construct from the stim "
    "circuit via from_stim_circuit (the mask is derived empirically from "
    "measurement bases), or (d) pass use_gari=False to run the telescoping "
    "decoder on the original DEM without the GARI transform.")


def _save_matrices_npz(path, H, L, priors, *, source_id="") -> None:
    """Write an original-system npz with the pipeline's exact key set."""
    H = H.tocsr()
    L = L.tocsr()
    dc_pad = int(np.diff(H.indptr).max())
    dv_pad = int(np.diff(H.tocsc().indptr).max())
    payload = dict(
        h_data=H.data.astype(np.uint8),
        h_indices=H.indices.astype(np.int32),
        h_indptr=H.indptr.astype(np.int32),
        h_shape=np.array(H.shape, dtype=np.int32),
        l_data=L.data.astype(np.uint8),
        l_indices=L.indices.astype(np.int32),
        l_indptr=L.indptr.astype(np.int32),
        l_shape=np.array(L.shape, dtype=np.int32),
        probs=np.asarray(priors, dtype=np.float64),
        dc_pad=np.int32(dc_pad),
        dv_pad=np.int32(dv_pad),
    )
    if source_id:
        payload["source_fingerprint"] = source_id
    np.savez(path, **payload)


def _load_matrices_npz(path):
    """Read an original-system npz -> (H csr, L csr, priors, dc_pad, dv_pad)."""
    d = np.load(path)
    H = sp.csr_matrix(
        (d["h_data"].astype(np.uint8), d["h_indices"].astype(np.int32),
         d["h_indptr"].astype(np.int32)), shape=tuple(d["h_shape"]))
    L = sp.csr_matrix(
        (d["l_data"].astype(np.uint8), d["l_indices"].astype(np.int32),
         d["l_indptr"].astype(np.int32)), shape=tuple(d["l_shape"]))
    source_id = (str(d["source_fingerprint"])
                 if "source_fingerprint" in d.files else "")
    return (H, L, d["probs"].astype(np.float64),
            int(d["dc_pad"]), int(d["dv_pad"]), source_id)


def derive_init_det_system(H, L, probs, is_x_detector, init_basis):
    """Derive the init-basis-detectors-only decode system.

    Take H restricted to the init-basis-family detector rows, merge columns
    with identical (detector, observable) signatures and XOR-combine their
    priors using p <- p(1-q) + q(1-p). This matches the independent-flip
    combination used when a Stim DEM emits only one detector family.

    Mechanisms whose signature is empty in both H[init rows] and L are
    dropped (invisible to this system); empty-H/nonempty-L columns are kept
    to match stim. Merged column order is first-occurrence order —
    deterministic.

    Returns (H_init csr, L_init csr, priors, init_idx, det_width,
    dc_pad, dv_pad).
    """
    H = H.tocsr()
    L = L.tocsr()
    probs = np.asarray(probs, dtype=np.float64)
    det_width = int(H.shape[0])

    mask = np.asarray(is_x_detector, dtype=bool)
    assert mask.shape[0] == det_width, (
        f"is_x_detector length {mask.shape[0]} != original H rows "
        f"{det_width}")
    init_idx = np.where(mask if init_basis == "X" else ~mask)[0].astype(
        np.int64)

    H_rows = H[init_idx, :].tocsc()
    H_rows.sort_indices()
    L_csc = L.tocsc()
    L_csc.sort_indices()
    hi, hp = H_rows.indices, H_rows.indptr
    li, lp = L_csc.indices, L_csc.indptr

    groups: dict = {}        # (h_sig, l_sig) -> merged column index
    merged_p: list = []
    merged_h: list = []
    merged_l: list = []
    for j in range(H.shape[1]):
        h_sig = tuple(hi[hp[j]:hp[j + 1]])
        l_sig = tuple(li[lp[j]:lp[j + 1]])
        if not h_sig and not l_sig:
            continue
        k = groups.get((h_sig, l_sig))
        p = float(probs[j])
        if k is None:
            groups[(h_sig, l_sig)] = len(merged_p)
            merged_p.append(p)
            merged_h.append(h_sig)
            merged_l.append(l_sig)
        else:
            q = merged_p[k]
            merged_p[k] = q * (1.0 - p) + p * (1.0 - q)

    n_merged = len(merged_p)

    def _csc_from_sigs(sigs, n_rows):
        lens = np.fromiter((len(s) for s in sigs), dtype=np.int64,
                           count=n_merged)
        indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        indices = np.fromiter(
            (r for s in sigs for r in s), dtype=np.int32, count=int(lens.sum()))
        data = np.ones(int(lens.sum()), dtype=np.uint8)
        return sp.csc_matrix((data, indices, indptr),
                             shape=(n_rows, n_merged)).tocsr()

    H_init = _csc_from_sigs(merged_h, len(init_idx))
    L_init = _csc_from_sigs(merged_l, int(L.shape[0]))
    priors = np.asarray(merged_p, dtype=np.float64)
    dc_pad = int(np.diff(H_init.indptr).max())
    dv_pad = int(np.diff(H_init.tocsc().indptr).max())
    return H_init, L_init, priors, init_idx, det_width, dc_pad, dv_pad


def _default_workdir(key_bytes: bytes) -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    h = hashlib.sha256(key_bytes).hexdigest()[:16]
    return Path(base) / "telescoping_decoder" / "systems" / h


@dataclasses.dataclass
class DecodingSystem:
    """The decode systems + their materialized npz artifacts."""

    # Original correlated XYZ system (None only when constructed from a GARI
    # npz alone — then only the "gari" system is available).
    H: Optional[sp.csr_matrix]
    L: Optional[sp.csr_matrix]
    priors: Optional[np.ndarray]
    dc_pad: Optional[int] = None
    dv_pad: Optional[int] = None
    n_detectors: int = 0            # syndrome width (stim detector order)
    n_obs: int = 0
    dem = None                      # stim.DetectorErrorModel or None
    is_x_detector: Optional[np.ndarray] = None
    init_basis: Optional[str] = None
    gari: Optional[GariModel] = None
    workdir: Optional[Path] = None
    # materialized/known npz paths, keyed by system name
    paths: dict = dataclasses.field(default_factory=dict)
    # init-det derived system, filled by ensure_init_dets()
    init_det: Optional[tuple] = None   # (H, L, priors, init_idx, det_width,
                                       #  dc_pad, dv_pad)
    source_fingerprint: str = ""

    # ------------------------------------------------------------------ #
    # constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_stim_circuit(cls, circuit, *, init_basis: str,
                          use_gari: bool = True, verify: bool = True,
                          verify_samples: int = 1024,
                          workdir=None) -> "DecodingSystem":
        """Build from a stim circuit, deriving the X/Z detector mask from the
        circuit's measurement bases (so every system, GARI included, is
        available). ``init_basis`` is "X" or "Z", the basis the memory
        experiment prepares and measures in."""
        mask = circuit_xz_detector_mask(circuit)
        dem = circuit_to_dem(circuit)
        return cls.from_dem(dem, is_x_detector=mask, init_basis=init_basis,
                            use_gari=use_gari, verify=verify,
                            verify_samples=verify_samples, workdir=workdir)

    @classmethod
    def from_dem(cls, dem, *, is_x_detector=None, canonical_layout=None,
                 init_basis: Optional[str] = None, use_gari: bool = True,
                 verify: bool = True, verify_samples: int = 1024,
                 workdir=None) -> "DecodingSystem":
        """Build from a stim DetectorErrorModel.

        A DEM does not record which detectors are X- and which are Z-type, so
        pass ``is_x_detector`` (bool per detector) or ``canonical_layout=
        (n_x, n_z, num_rounds)``; without either, only the original system is
        available. ``verify`` re-samples the DEM and checks the parsed
        matrices reproduce stim's syndromes."""
        from .dem_utils import dem_to_sparse_matrices
        if is_x_detector is None and canonical_layout is not None:
            if init_basis is None:
                raise ValueError("canonical_layout requires init_basis")
            n_x, n_z, num_rounds = canonical_layout
            is_x_detector = xz_detector_type_mask(
                n_x, n_z, num_rounds, init_basis=init_basis)
        if use_gari and is_x_detector is None:
            raise ValueError("from_dem with use_gari=True: " + _MASK_HELP)
        if use_gari and init_basis is None:
            raise ValueError(
                "from_dem with use_gari=True requires init_basis='X' or 'Z' "
                "(the data prep/readout basis; not inferrable from a DEM)")

        H, L, priors = dem_to_sparse_matrices(dem)
        H = H.tocsr()
        L = L.tocsr()
        gari = None
        if use_gari:
            gari = gari_transform(
                dem, np.asarray(is_x_detector, dtype=bool),
                init_basis=init_basis, verify=verify,
                verify_samples=verify_samples)
        obj = cls(
            H=H, L=L, priors=np.asarray(priors, dtype=np.float64),
            dc_pad=int(np.diff(H.indptr).max()),
            dv_pad=int(np.diff(H.tocsc().indptr).max()),
            n_detectors=int(H.shape[0]), n_obs=int(L.shape[0]),
            is_x_detector=(np.asarray(is_x_detector, dtype=bool)
                           if is_x_detector is not None else None),
            init_basis=init_basis, gari=gari,
            workdir=Path(workdir) if workdir is not None else None)
        obj.source_fingerprint = source_fingerprint(
            H, L, obj.priors, is_x_detector=obj.is_x_detector,
            init_basis=init_basis)
        # gari_transform independently fingerprints the same physical system;
        # keep this assertion close to construction so parser drift cannot
        # create artifacts that later appear unrelated.
        if gari is not None:
            if gari.source_fingerprint != obj.source_fingerprint:
                raise ValueError(
                    "internal error: original and GARI source fingerprints "
                    "differ")
        obj.dem = dem
        return obj

    @classmethod
    def from_matrices(cls, H, L, priors, *, is_x_detector=None,
                      init_basis: Optional[str] = None,
                      workdir=None) -> "DecodingSystem":
        """Raw-matrices construction: the original-system stages (non-GARI).

        The GARI transform needs a stim DEM, so ``system="gari"`` is
        unavailable here; ``"init_dets"`` is available iff ``is_x_detector``
        and ``init_basis`` are given.
        """
        H = sp.csr_matrix(H).astype(np.uint8)
        L = sp.csr_matrix(L).astype(np.uint8)
        priors = np.asarray(priors, dtype=np.float64)
        if H.shape[1] != L.shape[1]:
            raise ValueError(
                f"H has {H.shape[1]} columns but L has {L.shape[1]}; both "
                "must be indexed by the same error mechanisms")
        if priors.shape[0] != H.shape[1]:
            raise ValueError(
                f"priors has {priors.shape[0]} entries != H columns "
                f"{H.shape[1]}")
        source_id = source_fingerprint(
            H, L, priors, is_x_detector=is_x_detector,
            init_basis=init_basis)
        return cls(
            H=H, L=L, priors=priors,
            dc_pad=int(np.diff(H.indptr).max()),
            dv_pad=int(np.diff(H.tocsc().indptr).max()),
            n_detectors=int(H.shape[0]), n_obs=int(L.shape[0]),
            is_x_detector=(np.asarray(is_x_detector, dtype=bool)
                           if is_x_detector is not None else None),
            init_basis=init_basis,
            workdir=Path(workdir) if workdir is not None else None,
            source_fingerprint=source_id)

    @classmethod
    def from_npz(cls, *, gari_npz=None, matrices_npz=None,
                 workdir=None) -> "DecodingSystem":
        """Load previously materialized npz artifacts."""
        if gari_npz is None and matrices_npz is None:
            raise ValueError("from_npz needs gari_npz and/or matrices_npz")
        H = L = priors = None
        dc_pad = dv_pad = None
        n_det = n_obs = 0
        gari = None
        paths = {}
        mask = None
        init_basis = None
        original_source_id = ""
        gari_source_id = ""
        if matrices_npz is not None:
            (H, L, priors, dc_pad, dv_pad,
             original_source_id) = _load_matrices_npz(matrices_npz)
            n_det, n_obs = int(H.shape[0]), int(L.shape[0])
            paths["original"] = str(matrices_npz)
        if gari_npz is not None:
            gari = load_gari_npz(gari_npz)
            gari_source_id = gari.source_fingerprint
            paths["gari"] = str(gari_npz)
            mask = gari.is_x_detector
            init_basis = gari.init_basis
            if matrices_npz is not None and not original_source_id:
                # Legacy original NPZs predate embedded identities, but the
                # paired GARI artifact supplies the missing mask and basis.
                original_source_id = source_fingerprint(
                    H, L, priors, is_x_detector=mask,
                    init_basis=init_basis)
            if matrices_npz is None:
                n_det = int(gari.n_detectors)
                n_obs = int(gari.l.shape[0])
            else:
                if original_source_id != gari_source_id:
                    raise ValueError(
                        "matrices_npz and gari_npz come from different "
                        "decoding problems (source_fingerprint mismatch)")
        return cls(
            H=H, L=L, priors=priors, dc_pad=dc_pad, dv_pad=dv_pad,
            n_detectors=n_det, n_obs=n_obs, is_x_detector=mask,
            init_basis=init_basis, gari=gari, paths=paths,
            workdir=Path(workdir) if workdir is not None else None,
            source_fingerprint=original_source_id or gari_source_id)

    # ------------------------------------------------------------------ #
    # availability / resolution                                           #
    # ------------------------------------------------------------------ #

    @property
    def has_original(self) -> bool:
        return self.H is not None or "original" in self.paths

    @property
    def has_gari(self) -> bool:
        return self.gari is not None or "gari" in self.paths

    @property
    def can_init_dets(self) -> bool:
        return (self.has_original and self.is_x_detector is not None
                and self.init_basis in ("X", "Z"))

    def _available(self, system: str, use_gari: bool) -> bool:
        if system == "init_dets":
            return self.can_init_dets
        if system == "gari":
            return use_gari and self.has_gari
        return self.has_original

    def resolve(self, requested: str, stage: str, *,
                use_gari: bool = True) -> str:
        """Resolve a stage's ``system`` field against availability.

        ``"auto"`` walks that stage's chain (``_AUTO_CHAINS``) and returns the
        first system this decoder can build; anything else is strict and
        raises when unavailable. ``use_gari=False`` drops GARI from the chain
        (``TelescopeConfig.use_gari``)."""
        if requested == "auto":
            chain = _AUTO_CHAINS.get(stage, _DEFAULT_AUTO_CHAIN)
            for candidate in chain:
                if self._available(candidate, use_gari):
                    return candidate
            raise ValueError(
                f"{stage}.system='auto' but none of {chain} is available on "
                "this decoder; construct it with matrices_npz=/from_dem/"
                "from_matrices so at least the original system exists")
        if requested == "gari" and not self.has_gari:
            raise ValueError(
                f"{stage}.system='gari' but no GARI system is available. "
                + _MASK_HELP)
        if requested == "original" and not self.has_original:
            raise ValueError(
                f"{stage}.system='original' but the original matrices were "
                "not provided (constructed from a GARI npz alone); pass "
                "matrices_npz= too")
        if requested == "init_dets" and not self.can_init_dets:
            raise ValueError(
                f"{stage}.system='init_dets' needs the original matrices "
                "plus the X/Z detector mask and init_basis")
        if requested not in SYSTEMS:
            raise ValueError(f"unknown system {requested!r}")
        return requested

    # ------------------------------------------------------------------ #
    # derivation / materialization                                        #
    # ------------------------------------------------------------------ #

    def ensure_init_dets(self):
        if self.init_det is None:
            if not self.can_init_dets:
                raise ValueError(
                    "init_dets system needs the original matrices + X/Z "
                    "detector mask + init_basis")
            if self.H is None:
                self._load_original()
            self.init_det = derive_init_det_system(
                self.H, self.L, self.priors, self.is_x_detector,
                self.init_basis)
        return self.init_det

    def _load_original(self) -> None:
        (self.H, self.L, self.priors,
         self.dc_pad, self.dv_pad, loaded_source_id) = _load_matrices_npz(
            self.paths["original"])
        if self.source_fingerprint and loaded_source_id != self.source_fingerprint:
            raise ValueError("original artifact source_fingerprint mismatch")
        self.source_fingerprint = loaded_source_id
        self.n_detectors = int(self.H.shape[0])
        self.n_obs = int(self.L.shape[0])

    def _ensure_workdir(self) -> Path:
        if self.workdir is None:
            if self.source_fingerprint:
                key_bytes = self.source_fingerprint.encode("ascii")
            elif self.H is not None:
                # Compatibility for an in-memory object produced outside the
                # constructors. Include the complete source semantics.
                key_bytes = source_fingerprint(
                    self.H, self.L, self.priors,
                    is_x_detector=self.is_x_detector,
                    init_basis=self.init_basis).encode("ascii")
            else:
                key_bytes = str(sorted(self.paths.items())).encode()
            self.workdir = _default_workdir(key_bytes)
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir

    def npz_path_for(self, system: str) -> str:
        """Materialized npz for a resolved system name (writes on demand)."""
        if system in self.paths and os.path.isfile(self.paths[system]):
            return self.paths[system]
        wd = self._ensure_workdir()
        suffix = (f"-{self.source_fingerprint[:16]}"
                  if self.source_fingerprint else "")
        if system == "original":
            if self.H is None:
                raise ValueError("original system not available")
            path = wd / f"matrices{suffix}.npz"
            if not path.is_file() or not self._artifact_matches(path):
                _save_matrices_npz(
                    path, self.H, self.L, self.priors,
                    source_id=self.source_fingerprint)
        elif system == "gari":
            if self.gari is None:
                raise ValueError("GARI system not available")
            path = wd / f"gari_matrices{suffix}.npz"
            if not path.is_file() or not self._artifact_matches(path):
                save_gari_npz(path, self.gari)
        elif system == "init_dets":
            H_i, L_i, p_i, init_idx, _dw, _dc, _dv = self.ensure_init_dets()
            path = wd / f"init_dets_matrices{suffix}.npz"
            if not path.is_file() or not self._artifact_matches(path):
                _save_matrices_npz(
                    path, H_i, L_i, p_i,
                    source_id=self.source_fingerprint)
                # Record the selected detector rows for diagnostics and S4.
                d = dict(np.load(path))
                d["init_idx"] = init_idx
                np.savez(path, **d)
        else:
            raise ValueError(f"unknown system {system!r}")
        self.paths[system] = str(path)
        return self.paths[system]

    def _artifact_matches(self, path) -> bool:
        """Whether a cached artifact belongs to this physical system."""
        if not self.source_fingerprint:
            return False
        try:
            with np.load(path) as d:
                return ("source_fingerprint" in d.files
                        and str(d["source_fingerprint"])
                        == self.source_fingerprint)
        except (OSError, ValueError, KeyError):
            return False


__all__ = ["DecodingSystem", "derive_init_det_system", "SYSTEMS"]
