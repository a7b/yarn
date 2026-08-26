"""GARI (graph augmentation and rewiring for inference) transform.

Implements the decoding-matrix transform of Maan, Garcia-Herrero, Paler &
Savin, "Decoding Correlated Errors in Quantum LDPC Codes" (arXiv:2510.14060),
Eq. (5), for the correlated XYZ detector error models produced by our XZ
memory-experiment circuits.

Given the correlated DEM matrix (rows = detectors, columns = merged error
mechanisms), every column is classified by which detector *types* its
support touches:

  - support only on X-type detectors -> "eZ"-type column (the D_X block),
  - support only on Z-type detectors -> "eX"-type column (the D_Z block),
  - support on both                  -> "eY"-type correlated column.

For depolarizing circuit noise, every eY column's X-detector restriction
(together with its observable signature) equals exactly one eZ column, and
its Z-detector restriction equals exactly one eX column; these matchings
define the column-weight-1 matrices U and V of the paper. The change of
variables ``ebar_Z = e_Z + U e_Y``, ``ebar_X = e_X + V e_Y`` then yields the
GARI matrix

           eZ  eX  eY  ebarZ ebarX        syndrome
  Hbar = [  0   0   0   D_X    0    |       s
            0   0   0    0    D_Z   |       s
            I   0   U    I     0    |       0
            0   I   V    0     I    |       0  ]

with row order ``[detector rows (stim order) | U-rows | V-rows]`` and column
order ``[eZ | eX | eY | ebarZ | ebarX]``, so detector syndromes are extended
by zero-padding on the right. The transform removes the Y-induced 4-cycles
targeted by the construction. It generally reduces the edge count for the
correlated DEMs used here, but small or atypical inputs can gain edges.

``gari_transform`` validates the matching property and raises ``ValueError``
when a mixed column lacks a pure partner. With ``verify=True`` it also checks
the decoding and observable equivalences on random error vectors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import stim
from scipy import sparse

from ._fingerprint import source_fingerprint


# ---------------------------------------------------------------------------
# Detector typing
# ---------------------------------------------------------------------------

def xz_detector_type_mask(
    n_x: int,
    n_z: int,
    num_rounds: int,
    init_basis: str = "X",
) -> np.ndarray:
    """Boolean mask over detectors: True = X-type, False = Z-type.

    Assumes the canonical XZ memory-experiment detector emission order:
    each round emits X-detectors then Z-detectors, the first round skips the
    family whose stabilizer values are random under the chosen init basis,
    and the destructive final readout emits detectors of the init-basis
    family only.

      init_basis="X": [R1: n_x X][R2..R_r: n_x X, n_z Z][final MX: n_x X]
                      -> n_det = n_x*(r+1) + n_z*(r-1)
      init_basis="Z": [R1: n_z Z][R2..R_r: n_x X, n_z Z][final M:  n_z Z]
                      -> n_det = n_x*(r-1) + n_z*(r+1)

    DEM detector coordinates describe circuit locations or rounds, not X/Z
    type, and are not used for this classification.
    """
    if init_basis not in {"X", "Z"}:
        raise ValueError(f"Unsupported init basis: {init_basis}")
    x_block = np.ones(n_x, dtype=bool)
    z_block = np.zeros(n_z, dtype=bool)
    blocks = [x_block if init_basis == "X" else z_block]
    for _ in range(num_rounds - 1):
        blocks.append(x_block)
        blocks.append(z_block)
    blocks.append(x_block if init_basis == "X" else z_block)
    return np.concatenate(blocks)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GariModel:
    """GARI decoding system + structure metadata.

    ``h`` / ``l`` / ``priors`` define the decoding problem exactly like the
    original (H, L, probs) triple: decode the zero-padded syndrome against
    ``h``, predict observables as ``l @ correction % 2``. Auxiliary ebar
    columns carry prior 0.5 (channel LLR 0).
    """

    h: sparse.csr_matrix            # (n_det + nZ + nX) x (nC + nZ + nX), uint8
    l: sparse.csr_matrix            # (num_obs) x same cols, uint8
    priors: np.ndarray              # float64, 0.5 at ebar columns
    n_detectors: int                # detector rows come first, in stim order
    is_x_detector: np.ndarray       # (n_detectors,) bool
    col_blocks: dict                # name -> slice; eZ|eX|eY|ebarZ|ebarX
    row_blocks: dict                # name -> slice; detectors|U|V
    answer_block: str               # "ebarZ" (xinit) or "ebarX" (zinit)
    relevant_rows: np.ndarray       # detector rows of the answer-side type
    relevant_priors: np.ndarray     # combined priors of the answer block
    layers: np.ndarray              # (3, 2) row ranges in processing order:
                                    # [U-rows][V-rows][detector rows]
    u_map: np.ndarray               # (nY,) eY index -> eZ-block index
    v_map: np.ndarray               # (nY,) eY index -> eX-block index
    init_basis: str
    source_fingerprint: str = ""    # identity shared with the original NPZ

    @property
    def n_rows(self) -> int:
        return self.h.shape[0]

    @property
    def n_cols(self) -> int:
        return self.h.shape[1]


# ---------------------------------------------------------------------------
# DEM parsing
# ---------------------------------------------------------------------------

def _parse_dem_columns(dem: stim.DetectorErrorModel) -> dict:
    """Merge error instructions into columns keyed on (dets, obs) jointly.

    Same XOR-of-targets and prior-combination (p∘q = p+q-2pq) semantics as
    ``dem_utils.dem_to_sparse_matrices``. Mechanisms with equal detector
    support but different observable support are rejected: together they form
    a detector-silent, logically nontrivial error. Insertion order is kept for
    determinism.

    Returns an ordered dict {(dets_tuple, obs_tuple): prob} with sorted
    tuples of int detector / observable indices.
    """
    cols: dict = {}
    detector_observables: dict[tuple[int, ...], tuple[int, ...]] = {}
    for ins in dem.flattened():
        if ins.type != "error":
            if ins.type in ("detector", "logical_observable"):
                continue
            raise NotImplementedError(f"Unsupported DEM instruction: {ins.type}")
        dets: set = set()
        obs: set = set()
        prob = ins.args_copy()[0]
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                dets.symmetric_difference_update((t.val,))
            elif t.is_logical_observable_id():
                obs.symmetric_difference_update((t.val,))
        detector_key = tuple(sorted(dets))
        observable_key = tuple(sorted(obs))
        previous_observables = detector_observables.get(detector_key)
        if (
            previous_observables is not None
            and previous_observables != observable_key
        ):
            raise ValueError(
                "DEM contains error mechanisms with identical detector "
                f"support {list(detector_key)} but different logical-"
                f"observable support {list(previous_observables)} and "
                f"{list(observable_key)}. XORing them gives a detector-silent, "
                "logically nontrivial error, so the circuit-level distance is "
                "at most 2. Refusing to merge them."
            )
        detector_observables[detector_key] = observable_key
        key = (detector_key, observable_key)
        prev = cols.get(key, 0.0)
        cols[key] = prev * (1.0 - prob) + prob * (1.0 - prev)
    return cols


def _combine(p: float, q: float) -> float:
    """XOR-combine two independent flip probabilities."""
    return p + q - 2.0 * p * q


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def gari_transform(
    dem: stim.DetectorErrorModel,
    is_x_detector: np.ndarray,
    *,
    init_basis: str,
    verify: bool = True,
    verify_samples: int = 1024,
    seed: int = 0,
) -> GariModel:
    """Build the GARI decoding system for a correlated XYZ DEM.

    ``is_x_detector`` is the boolean detector-type mask (True = X-type), see
    ``xz_detector_type_mask``. ``init_basis`` selects which side carries the
    tracked observables: "X" (|+> prep, logical-X observables, flipped by
    Z-components, detected by X-detectors, answer block ebarZ) or "Z"
    (mirrored, answer block ebarX).

    Raises ValueError on any structural violation (det-empty column,
    unmatched mixed column, observables on the wrong pure side).
    """
    if init_basis not in {"X", "Z"}:
        raise ValueError(f"Unsupported init basis: {init_basis}")
    is_x_detector = np.asarray(is_x_detector, dtype=bool)
    n_det = int(is_x_detector.shape[0])
    if n_det != dem.num_detectors:
        raise ValueError(
            f"is_x_detector has {n_det} entries but DEM has "
            f"{dem.num_detectors} detectors")

    cols = _parse_dem_columns(dem)

    # Fingerprint the physical system before changing its column order or
    # adding auxiliary variables. Original and GARI artifacts made from this
    # DEM will consequently carry exactly the same identity.
    source_h_rows: list[int] = []
    source_h_cols: list[int] = []
    source_l_rows: list[int] = []
    source_l_cols: list[int] = []
    source_priors: list[float] = []
    for j, ((dets, obs), p) in enumerate(cols.items()):
        source_h_rows.extend(dets)
        source_h_cols.extend([j] * len(dets))
        source_l_rows.extend(obs)
        source_l_cols.extend([j] * len(obs))
        source_priors.append(p)
    source_h = sparse.csr_matrix(
        (np.ones(len(source_h_rows), dtype=np.uint8),
         (source_h_rows, source_h_cols)),
        shape=(dem.num_detectors, len(cols)),
    )
    source_l = sparse.csr_matrix(
        (np.ones(len(source_l_rows), dtype=np.uint8),
         (source_l_rows, source_l_cols)),
        shape=(dem.num_observables, len(cols)),
    )
    source_id = source_fingerprint(
        source_h, source_l, np.asarray(source_priors, dtype=np.float64),
        is_x_detector=is_x_detector, init_basis=init_basis,
    )

    # --- classify columns by detector-type support -------------------------
    # eZ-type: support only on X-dets (the D_X block); eX-type: only Z-dets
    # (D_Z); eY-type: both. Only pure columns on the initialized-basis side,
    # and mixed columns, may carry tracked observables.
    eZ_dets: list = []; eZ_obs: list = []; eZ_p: list = []
    eX_dets: list = []; eX_obs: list = []; eX_p: list = []
    eY_x: list = []; eY_z: list = []; eY_obs: list = []; eY_p: list = []
    for (dets, obs), p in cols.items():
        if not dets:
            raise ValueError(
                f"DEM column with empty detector support (obs={obs}): "
                "undetectable error mechanisms are not supported")
        dx = tuple(d for d in dets if is_x_detector[d])
        dz = tuple(d for d in dets if not is_x_detector[d])
        if dx and dz:
            eY_x.append(dx); eY_z.append(dz); eY_obs.append(obs); eY_p.append(p)
        elif dx:
            eZ_dets.append(dx); eZ_obs.append(obs); eZ_p.append(p)
        else:
            eX_dets.append(dz); eX_obs.append(obs); eX_p.append(p)

    nZ, nX, nY = len(eZ_dets), len(eX_dets), len(eY_x)

    # Pure columns on the non-observable side must have empty obs signatures
    # (e.g. xinit tracks logical-X, which pure X-components never flip).
    off_obs, off_name = (eX_obs, "eX") if init_basis == "X" else (eZ_obs, "eZ")
    n_bad = sum(1 for o in off_obs if o)
    if n_bad:
        raise ValueError(
            f"{n_bad}/{len(off_obs)} {off_name}-type columns carry observables;"
            f" inconsistent with init_basis={init_basis!r}")

    # --- U/V matching (dictionary lookups) ---------------------------------
    # Mixed column j matches the eZ column equal to (X-restriction, obs-if-
    # xinit) and the eX column equal to (Z-restriction, obs-if-zinit).
    eZ_pos = {(d, o): i for i, (d, o) in enumerate(zip(eZ_dets, eZ_obs))}
    eX_pos = {(d, o): i for i, (d, o) in enumerate(zip(eX_dets, eX_obs))}
    empty: tuple = ()
    u_map = np.empty(nY, dtype=np.int64)
    v_map = np.empty(nY, dtype=np.int64)
    n_un_u = n_un_v = 0
    for j in range(nY):
        o = eY_obs[j]
        u_key = (eY_x[j], o if init_basis == "X" else empty)
        v_key = (eY_z[j], o if init_basis == "Z" else empty)
        ui = eZ_pos.get(u_key)
        vi = eX_pos.get(v_key)
        if ui is None:
            n_un_u += 1
        else:
            u_map[j] = ui
        if vi is None:
            n_un_v += 1
        else:
            v_map[j] = vi
    if n_un_u or n_un_v:
        raise ValueError(
            f"GARI matching failed: {n_un_u}/{nY} mixed columns without an "
            f"eZ partner, {n_un_v}/{nY} without an eX partner. The DEM does "
            "not have the depolarizing pure-component structure GARI requires.")

    # --- assemble Hbar ------------------------------------------------------
    # Row order: [detectors 0..n_det-1 | U-rows (one per eZ col) | V-rows];
    # column order: [eZ | eX | eY | ebarZ | ebarX].
    c_eZ, c_eX, c_eY = 0, nZ, nZ + nX
    c_ebarZ, c_ebarX = nZ + nX + nY, nZ + nX + nY + nZ
    n_cols = nZ + nX + nY + nZ + nX
    r_U, r_V = n_det, n_det + nZ
    n_rows = n_det + nZ + nX

    rows_parts: list = []; cols_parts: list = []

    # ebarZ block: each eZ column's detector support, moved onto ebarZ.
    eZ_lens = np.fromiter((len(d) for d in eZ_dets), dtype=np.int64, count=nZ)
    rows_parts.append(np.fromiter(
        (d for ds in eZ_dets for d in ds), dtype=np.int64, count=int(eZ_lens.sum())))
    cols_parts.append(np.repeat(np.arange(c_ebarZ, c_ebarZ + nZ), eZ_lens))
    # ebarX block likewise.
    eX_lens = np.fromiter((len(d) for d in eX_dets), dtype=np.int64, count=nX)
    rows_parts.append(np.fromiter(
        (d for ds in eX_dets for d in ds), dtype=np.int64, count=int(eX_lens.sum())))
    cols_parts.append(np.repeat(np.arange(c_ebarX, c_ebarX + nX), eX_lens))
    # Identity edges on the consistency rows: ebarZ/ebarX and eZ/eX columns.
    rows_parts.append(np.arange(r_U, r_U + nZ)); cols_parts.append(np.arange(c_ebarZ, c_ebarZ + nZ))
    rows_parts.append(np.arange(r_V, r_V + nX)); cols_parts.append(np.arange(c_ebarX, c_ebarX + nX))
    rows_parts.append(np.arange(r_U, r_U + nZ)); cols_parts.append(np.arange(c_eZ, c_eZ + nZ))
    rows_parts.append(np.arange(r_V, r_V + nX)); cols_parts.append(np.arange(c_eX, c_eX + nX))
    # eY columns: degree 2, one edge per consistency row of each partner.
    if nY:
        rows_parts.append(r_U + u_map); cols_parts.append(np.arange(c_eY, c_eY + nY))
        rows_parts.append(r_V + v_map); cols_parts.append(np.arange(c_eY, c_eY + nY))

    rr = np.concatenate(rows_parts); cc = np.concatenate(cols_parts)
    h = sparse.csr_matrix(
        (np.ones(rr.size, dtype=np.uint8), (rr, cc)), shape=(n_rows, n_cols))

    # --- observable matrix Lbar --------------------------------------------
    # Store tracked observables on the answer block only (see module docs):
    # the obs-side pure columns' signatures, placed at their ebar positions.
    if init_basis == "X":
        obs_src, obs_col0 = eZ_obs, c_ebarZ
    else:
        obs_src, obs_col0 = eX_obs, c_ebarX
    l_rows: list = []; l_cols: list = []
    for i, o in enumerate(obs_src):
        for ob in o:
            l_rows.append(ob); l_cols.append(obs_col0 + i)
    l = sparse.csr_matrix(
        (np.ones(len(l_rows), dtype=np.uint8),
         (np.asarray(l_rows, dtype=np.int64), np.asarray(l_cols, dtype=np.int64))),
        shape=(dem.num_observables, n_cols))

    # --- priors -------------------------------------------------------------
    priors = np.concatenate([
        np.asarray(eZ_p, dtype=np.float64),
        np.asarray(eX_p, dtype=np.float64),
        np.asarray(eY_p, dtype=np.float64),
        np.full(nZ + nX, 0.5, dtype=np.float64),   # ebar: no prior knowledge
    ])

    # Combined answer-block priors (for future masked-stop weighting): the
    # ebar variable flips iff an odd number of {pure column, its mapped eY
    # columns} flip.
    if init_basis == "X":
        relevant_priors = np.asarray(eZ_p, dtype=np.float64).copy()
        rel_map = u_map
    else:
        relevant_priors = np.asarray(eX_p, dtype=np.float64).copy()
        rel_map = v_map
    for j in range(nY):
        i = rel_map[j]
        relevant_priors[i] = _combine(relevant_priors[i], eY_p[j])

    col_blocks = {
        "eZ": slice(c_eZ, c_eZ + nZ),
        "eX": slice(c_eX, c_eX + nX),
        "eY": slice(c_eY, c_eY + nY),
        "ebarZ": slice(c_ebarZ, c_ebarZ + nZ),
        "ebarX": slice(c_ebarX, c_ebarX + nX),
    }
    row_blocks = {
        "detectors": slice(0, n_det),
        "U": slice(r_U, r_U + nZ),
        "V": slice(r_V, r_V + nX),
    }
    relevant_rows = (np.where(is_x_detector)[0] if init_basis == "X"
                     else np.where(~is_x_detector)[0])
    layers = np.array([[r_U, r_U + nZ], [r_V, r_V + nX], [0, n_det]],
                      dtype=np.int64)

    gm = GariModel(
        h=h, l=l, priors=priors,
        n_detectors=n_det, is_x_detector=is_x_detector,
        col_blocks=col_blocks, row_blocks=row_blocks,
        answer_block="ebarZ" if init_basis == "X" else "ebarX",
        relevant_rows=relevant_rows, relevant_priors=relevant_priors,
        layers=layers, u_map=u_map, v_map=v_map, init_basis=init_basis,
        source_fingerprint=source_id,
    )

    if verify:
        _verify_gari(gm, cols, n_samples=verify_samples, seed=seed)
    return gm


# ---------------------------------------------------------------------------
# Built-in verification
# ---------------------------------------------------------------------------

def _verify_gari(gm: GariModel, cols: dict, *, n_samples: int, seed: int) -> None:
    """Check the GARI system against the original on random error vectors.

    Builds the original (H, L) in GARI column order [eZ | eX | eY] and
    asserts, for random e:  Hbar @ (e, ebar(e)) == [H @ e ; 0]  and
    Lbar @ (e, ebar(e)) == L @ e,  where ebar is given by the change of
    variables. Also checks nnz reduction and warns about dets-only merge
    collisions (which would make the *original* `_matrices.npz` suspect).
    """
    nZ = gm.col_blocks["eZ"].stop - gm.col_blocks["eZ"].start
    nX = gm.col_blocks["eX"].stop - gm.col_blocks["eX"].start
    nY = gm.col_blocks["eY"].stop - gm.col_blocks["eY"].start
    nC = nZ + nX + nY
    n_det = gm.n_detectors

    # Original H/L in GARI column order. _parse_dem_columns preserves
    # insertion order, and gari_transform's classification scans that order,
    # so re-scanning here reproduces the eZ/eX/eY ordering exactly.
    is_x = gm.is_x_detector
    h_rows: list = []; h_cols: list = []
    l_rows: list = []; l_cols: list = []
    iZ = iX = iY = 0
    for (dets, obs), _p in cols.items():
        dx = [d for d in dets if is_x[d]]
        dz = [d for d in dets if not is_x[d]]
        if dx and dz:
            ci = nZ + nX + iY; iY += 1
        elif dx:
            ci = iZ; iZ += 1
        else:
            ci = nZ + iX; iX += 1
        for d in dets:
            h_rows.append(d); h_cols.append(ci)
        for ob in obs:
            l_rows.append(ob); l_cols.append(ci)
    assert (iZ, iX, iY) == (nZ, nX, nY)
    h_orig = sparse.csr_matrix(
        (np.ones(len(h_rows), np.uint8), (h_rows, h_cols)), shape=(n_det, nC))
    l_orig = sparse.csr_matrix(
        (np.ones(len(l_rows), np.uint8), (l_rows, l_cols)),
        shape=(gm.l.shape[0], nC))

    if not gm.h.nnz < h_orig.nnz:
        # Not a correctness condition: GARI adds one edge per bare 4-cycle
        # and only reduces edges for larger bicliques (paper Sec. II A 2).
        # Real correlated DEMs have large bicliques and always shrink.
        warnings.warn(
            f"GARI did not reduce nonzeros ({gm.h.nnz} >= {h_orig.nnz}); "
            "expected only for toy DEMs with tiny bicliques.", stacklevel=3)

    n_dets_only = len({d for (d, _o) in cols})
    if n_dets_only != len(cols):
        warnings.warn(
            f"{len(cols) - n_dets_only} (dets, obs) column pairs share "
            "identical detector signatures; the dets-only merge used by "
            "dem_to_sparse_matrices conflates them — the existing "
            "_matrices.npz for this DEM is suspect.", stacklevel=3)

    # U/V as sparse matrices for the change of variables.
    U = sparse.csr_matrix(
        (np.ones(nY, np.uint8), (gm.u_map, np.arange(nY))), shape=(nZ, nY))
    V = sparse.csr_matrix(
        (np.ones(nY, np.uint8), (gm.v_map, np.arange(nY))), shape=(nX, nY))

    rng = np.random.default_rng(seed)
    chunk = 64
    p_flip = min(0.5, 64.0 / nC)   # ~64 flipped columns per sample
    done = 0
    while done < n_samples:
        t = min(chunk, n_samples - done)
        # int32: row degrees can exceed uint8 range in the matmuls below
        e = (rng.random((nC, t)) < p_flip).astype(np.int32)
        e_Z, e_X, e_Y = e[:nZ], e[nZ:nZ + nX], e[nZ + nX:]
        ebar_Z = (e_Z + U @ e_Y) % 2
        ebar_X = (e_X + V @ e_Y) % 2
        e_hat = np.concatenate([e, ebar_Z, ebar_X], axis=0)
        lhs = np.asarray(gm.h @ e_hat) % 2
        rhs = np.zeros_like(lhs)
        rhs[:n_det] = np.asarray(h_orig @ e) % 2
        if not np.array_equal(lhs, rhs):
            raise ValueError("GARI verification failed: Hbar @ (e, ebar) != [H @ e; 0]")
        if not np.array_equal(np.asarray(gm.l @ e_hat) % 2,
                              np.asarray(l_orig @ e) % 2):
            raise ValueError("GARI verification failed: Lbar @ (e, ebar) != L @ e")
        done += t


# ---------------------------------------------------------------------------
# Syndrome adapter
# ---------------------------------------------------------------------------

def extend_syndromes(synd: np.ndarray, n_rows: int) -> np.ndarray:
    """Zero-pad detector-space syndromes (1-D or batch (B, n_det)) on the
    right to the GARI row count. No-op if already the right width."""
    synd = np.asarray(synd)
    pad = n_rows - synd.shape[-1]
    if pad < 0:
        raise ValueError(
            f"syndrome width {synd.shape[-1]} exceeds GARI rows {n_rows}")
    if pad == 0:
        return synd
    pad_width = [(0, 0)] * (synd.ndim - 1) + [(0, pad)]
    return np.pad(synd, pad_width)


# ---------------------------------------------------------------------------
# npz I/O (pipeline-compatible schema + gari_* extras)
# ---------------------------------------------------------------------------

_BLOCK_COL_ORDER = ("eZ", "eX", "eY", "ebarZ", "ebarX")
_BLOCK_ROW_ORDER = ("detectors", "U", "V")


def save_gari_npz(path, gm: GariModel) -> None:
    """Write the GARI system with the same required keys as the pipeline's
    `_matrices.npz` (h_*, l_*, probs, dc_pad, dv_pad) plus gari_* extras, so
    the pipeline can load it as a drop-in matrices file."""
    h = gm.h.tocsr()
    l = gm.l.tocsr()
    dc_pad = int(np.diff(h.indptr).max())
    dv_pad = int(np.diff(h.tocsc().indptr).max())
    col_bounds = np.array(
        [gm.col_blocks[name].start for name in _BLOCK_COL_ORDER]
        + [gm.col_blocks[_BLOCK_COL_ORDER[-1]].stop], dtype=np.int64)
    row_bounds = np.array(
        [gm.row_blocks[name].start for name in _BLOCK_ROW_ORDER]
        + [gm.row_blocks[_BLOCK_ROW_ORDER[-1]].stop], dtype=np.int64)
    payload = dict(
        h_data=h.data.astype(np.uint8),
        h_indices=h.indices.astype(np.int32),
        h_indptr=h.indptr.astype(np.int32),
        h_shape=np.array(h.shape, dtype=np.int32),
        l_data=l.data.astype(np.uint8),
        l_indices=l.indices.astype(np.int32),
        l_indptr=l.indptr.astype(np.int32),
        l_shape=np.array(l.shape, dtype=np.int32),
        probs=gm.priors.astype(np.float64),
        dc_pad=np.int32(dc_pad),
        dv_pad=np.int32(dv_pad),
        gari_n_detectors=np.int64(gm.n_detectors),
        gari_is_x_detector=gm.is_x_detector,
        gari_col_block_bounds=col_bounds,
        gari_row_block_bounds=row_bounds,
        gari_relevant_rows=gm.relevant_rows.astype(np.int64),
        gari_relevant_priors=gm.relevant_priors.astype(np.float64),
        gari_layers=gm.layers.astype(np.int64),
        gari_u_map=gm.u_map.astype(np.int64),
        gari_v_map=gm.v_map.astype(np.int64),
        gari_init_basis=gm.init_basis,
        gari_answer_block=gm.answer_block,
    )
    if gm.source_fingerprint:
        payload["source_fingerprint"] = gm.source_fingerprint
    np.savez(path, **payload)


def load_gari_npz(path) -> GariModel:
    """Reload a GariModel written by ``save_gari_npz``."""
    d = np.load(path)
    h = sparse.csr_matrix(
        (d["h_data"].astype(np.uint8), d["h_indices"].astype(np.int32),
         d["h_indptr"].astype(np.int32)), shape=tuple(d["h_shape"]))
    l = sparse.csr_matrix(
        (d["l_data"].astype(np.uint8), d["l_indices"].astype(np.int32),
         d["l_indptr"].astype(np.int32)), shape=tuple(d["l_shape"]))
    cb = d["gari_col_block_bounds"]
    rb = d["gari_row_block_bounds"]
    col_blocks = {name: slice(int(cb[i]), int(cb[i + 1]))
                  for i, name in enumerate(_BLOCK_COL_ORDER)}
    row_blocks = {name: slice(int(rb[i]), int(rb[i + 1]))
                  for i, name in enumerate(_BLOCK_ROW_ORDER)}
    gm = GariModel(
        h=h, l=l, priors=d["probs"].astype(np.float64),
        n_detectors=int(d["gari_n_detectors"]),
        is_x_detector=d["gari_is_x_detector"].astype(bool),
        col_blocks=col_blocks, row_blocks=row_blocks,
        answer_block=str(d["gari_answer_block"]),
        relevant_rows=d["gari_relevant_rows"].astype(np.int64),
        relevant_priors=d["gari_relevant_priors"].astype(np.float64),
        layers=d["gari_layers"].astype(np.int64),
        u_map=d["gari_u_map"].astype(np.int64),
        v_map=d["gari_v_map"].astype(np.int64),
        init_basis=str(d["gari_init_basis"]),
        source_fingerprint=(str(d["source_fingerprint"])
                            if "source_fingerprint" in d.files else ""),
    )
    if not gm.source_fingerprint:
        gm.source_fingerprint = _reconstruct_source_fingerprint(gm)
    return gm


def _reconstruct_source_fingerprint(gm: GariModel) -> str:
    """Recover the physical-system identity from a legacy GARI artifact."""
    det = gm.row_blocks["detectors"]
    eZ = gm.col_blocks["eZ"]
    eX = gm.col_blocks["eX"]
    eY = gm.col_blocks["eY"]
    ebarZ = gm.col_blocks["ebarZ"]
    ebarX = gm.col_blocks["ebarX"]

    h_z = gm.h[det, ebarZ].tocsr()
    h_x = gm.h[det, ebarX].tocsr()
    h_y = (h_z[:, gm.u_map] + h_x[:, gm.v_map]).astype(np.uint8)
    source_h = sparse.hstack((h_z, h_x, h_y), format="csr")

    l_z = gm.l[:, ebarZ].tocsr()
    l_x = gm.l[:, ebarX].tocsr()
    l_y = (l_z[:, gm.u_map] if gm.init_basis == "X"
           else l_x[:, gm.v_map])
    source_l = sparse.hstack((l_z, l_x, l_y), format="csr")
    n_physical = eY.stop
    return source_fingerprint(
        source_h, source_l, gm.priors[:n_physical],
        is_x_detector=gm.is_x_detector, init_basis=gm.init_basis,
    )
