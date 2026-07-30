# -*- coding: utf-8 -*-
"""Distance estimation for GENERAL (non-CSS) stabilizer codes, symplectic form.

Random information-set sampling in the style of QDistRnd's ``DistRandStab``:
per trial, scramble the code by a symplectic-weight-preserving local map
(random per-qubit X/Z column swaps and shears) plus a random elimination
column order, row-reduce a basis of the CENTRALIZER, and harvest low-weight
rows that are not stabilizers. The minimum symplectic weight seen over all
trials upper-bounds the code distance; with enough trials it converges onto
it. This is the non-CSS analogue of the sqetch/BP+OSD CSS estimators and is
the authoritative distance record for the Y-surgery merged codes (which are
necessarily non-CSS — see ``surgery/cheeger_bridge/Y_GADGET_DESIGN.html``).

Also provides the affine variant (:func:`estimate_coset_min_weight`) for the
"masked" measurement-fault cosets ``X̄·⟨S, Init, L⟩`` / ``Z̄·⟨S, Init, L⟩``.

Conventions (match ``core/dist``):
    - matrices ``np.uint8`` with entries {0,1};
    - a check / operator is a symplectic pair of length-``W`` vectors
      ``(x, z)`` meaning the Pauli ``X(x)·Z(z)`` up to phase;
    - two operators commute iff ``x·z' + z·x' = 0 (mod 2)``;
    - symplectic (qubit) weight = ``|supp(x) ∪ supp(z)|``;
    - **strict <** early-stop: only a weight ``< d_target`` triggers early
      return, never ``== d_target``;
    - every result dict records ``num_trials`` actually run and the ``seed``.

Pure numpy (needs numpy >= 2.0 for ``np.bitwise_count``). No GPU, no GAP.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = [
    "symplectic_weight",
    "stab_k",
    "centralizer_basis",
    "pure_sector_css",
    "estimate_stab_distance",
    "estimate_coset_min_weight",
]


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _f2(M) -> np.ndarray:
    return np.asarray(M, dtype=np.uint8) & 1


def symplectic_weight(x: np.ndarray, z: np.ndarray) -> int:
    """Qubit weight of the Pauli with X-part ``x`` and Z-part ``z``:
    ``|supp(x) ∪ supp(z)|``. Both ``(W,)`` uint8."""
    return int((_f2(x) | _f2(z)).sum())


def _rref_u8(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Plain uint8 RREF (natural column order). Returns ``(R, pivots)`` where
    ``R`` keeps only the nonzero rows. Used once per call on small inputs."""
    M = _f2(M).copy()
    m, n = M.shape
    piv: list[int] = []
    r = 0
    for c in range(n):
        rows = np.flatnonzero(M[r:, c])
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        M[[r, p]] = M[[p, r]]
        mask = M[:, c].astype(bool).copy()
        mask[r] = False
        M[mask] ^= M[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    return M[:r], piv


def _f2_rank(M: np.ndarray) -> int:
    M = _f2(M)
    if M.size == 0 or M.shape[0] == 0:
        return 0
    return _rref_u8(M)[0].shape[0]


def stab_k(HX: np.ndarray, HZ: np.ndarray) -> int:
    """Logical count ``k = W - rank([HX | HZ])`` of a stabilizer code given
    the X/Z parts of its (possibly redundant) generator list."""
    HX, HZ = _f2(HX), _f2(HZ)
    return HX.shape[1] - _f2_rank(np.hstack([HX, HZ]))


def centralizer_basis(HX: np.ndarray, HZ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Basis of the centralizer ``C(S) = {v : Ω(v, s) = 0 ∀ s}``.

    ``v = (x, z)`` commutes with every generator iff ``[HZ | HX] · (x, z)ᵀ = 0``
    — the null space of the Ω-swapped generator matrix. Returns ``(CX, CZ)``,
    each ``(dim, W)`` uint8, with ``dim = 2W - rank([HX|HZ]) = W + k``.
    """
    HX, HZ = _f2(HX), _f2(HZ)
    W = HX.shape[1]
    K = np.hstack([HZ, HX])  # Ω-swap
    R, piv = _rref_u8(K)
    piv_set = set(piv)
    free = [c for c in range(2 * W) if c not in piv_set]
    basis = np.zeros((len(free), 2 * W), dtype=np.uint8)
    for i, fc in enumerate(free):
        v = basis[i]
        v[fc] = 1
        # back-substitute pivots (R is in RREF: each pivot row has a single
        # pivot column, eliminated everywhere else)
        for row_idx, pc in enumerate(piv):
            s = int((R[row_idx] @ v) % 2)
            if s:
                v[pc] ^= 1
    return basis[:, :W], basis[:, W:]


def _pack_bits(M: np.ndarray) -> np.ndarray:
    """Pack a (r, n) uint8 matrix into (r, ceil(n/64)) uint64 words."""
    M = _f2(M)
    r, n = M.shape
    nw = (n + 63) // 64
    pad = np.zeros((r, nw * 64), dtype=np.uint8)
    pad[:, :n] = M
    bits = pad.reshape(r, nw, 64)
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    return (bits.astype(np.uint64) * weights).sum(axis=2)


def _unpack_bits(P: np.ndarray, n: int) -> np.ndarray:
    r, nw = P.shape
    bits = ((P[:, :, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1))
    return bits.reshape(r, nw * 64)[:, :n].astype(np.uint8)


def _flat_rref(P: np.ndarray, ncols: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Packed RREF over flat columns 0..ncols-1 in natural order."""
    m = P.shape[0]
    r = 0
    pivots: list[tuple[int, int]] = []
    for c in range(ncols):
        word, bit = c >> 6, np.uint64(c & 63)
        col = (P[:, word] >> bit) & np.uint64(1)
        nz = np.flatnonzero(col[r:])
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            P[[r, p]] = P[[p, r]]
        col = (P[:, word] >> bit) & np.uint64(1)
        mask = col.astype(bool)
        mask[r] = False
        if mask.any():
            P[mask] ^= P[r]
        pivots.append((c, r))
        r += 1
        if r == m:
            break
    return P, pivots


def _nullspace_u8(M: np.ndarray) -> np.ndarray:
    """Right null space of a (m, n) uint8 matrix, bit-packed [Mᵀ | I] method:
    rows of the reduced augmented matrix whose Mᵀ-part is zero give the basis
    in their I-part. Returns (dim, n) uint8."""
    M = _f2(M)
    m, n = M.shape
    if m == 0 or M.sum() == 0:
        return np.eye(n, dtype=np.uint8)
    aug = np.zeros((n, m + n), dtype=np.uint8)
    aug[:, :m] = M.T
    aug[:, m:] = np.eye(n, dtype=np.uint8)
    P = _pack_bits(aug)
    P, _ = _flat_rref(P, m)
    U = _unpack_bits(P, m + n)
    zero_head = ~U[:, :m].any(axis=1)
    return U[zero_head][:, m:]


# --------------------------------------------------------------------------- #
# Bit-packed core.
# --------------------------------------------------------------------------- #
def _pack_pair(X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack (r, W) X/Z uint8 halves into one (r, 2*wx) uint64 array: words
    ``[:wx]`` = X half, ``[wx:]`` = Z half, both with the SAME qubit->bit
    layout so ``weight = popcount(Px | Pz)``. Returns ``(P, wx)``."""
    X, Z = _f2(X), _f2(Z)
    r, W = X.shape
    wx = (W + 63) // 64
    P = np.zeros((r, 2 * wx), dtype=np.uint64)
    for half, M in ((0, X), (1, Z)):
        # pad columns to wx*64, pack little-bit-order per 64-bit word
        pad = np.zeros((r, wx * 64), dtype=np.uint8)
        pad[:, :W] = M
        bits = pad.reshape(r, wx, 64)
        weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
        P[:, half * wx:(half + 1) * wx] = (bits.astype(np.uint64) * weights).sum(axis=2)
    return P, wx


def _unpack_pair(P: np.ndarray, wx: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`_pack_pair` for a (r, 2*wx) packed array."""
    r = P.shape[0]
    out = []
    for half in (0, 1):
        words = P[:, half * wx:(half + 1) * wx]
        bits = ((words[:, :, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1))
        out.append(bits.reshape(r, wx * 64)[:, :W].astype(np.uint8))
    return out[0], out[1]


def _row_weights(P: np.ndarray, wx: int) -> np.ndarray:
    """Symplectic (qubit) weights of packed rows."""
    return np.bitwise_count(P[:, :wx] | P[:, wx:]).sum(axis=1).astype(np.int64)


def _packed_rref(P: np.ndarray, col_order: np.ndarray, wx: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """In-place full RREF of packed rows ``P`` over virtual columns visited in
    ``col_order`` (values in [0, 2W)). Column ``c`` maps to word
    ``c//64 (+wx for the Z half)`` / bit ``c%64``. Returns ``(P, pivots)`` with
    ``pivots = [(col, row), ...]`` in elimination order."""
    m = P.shape[0]
    r = 0
    pivots: list[tuple[int, int]] = []
    for c in col_order:
        c = int(c)
        half, cc = (0, c) if c < wx * 64 else (1, c - wx * 64)
        word = half * wx + (cc >> 6)
        bit = np.uint64(cc & 63)
        col = (P[:, word] >> bit) & np.uint64(1)
        nz = np.flatnonzero(col[r:])
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            P[[r, p]] = P[[p, r]]
        col = (P[:, word] >> bit) & np.uint64(1)
        mask = col.astype(bool)
        mask[r] = False
        if mask.any():
            P[mask] ^= P[r]
        pivots.append((c, r))
        r += 1
        if r == m:
            break
    return P, pivots


def _reduce_vec(v: np.ndarray, R: np.ndarray, pivots: list[tuple[int, int]], wx: int) -> np.ndarray:
    """Reduce packed vector ``v`` (shape (2*wx,)) against packed RREF ``R``."""
    v = v.copy()
    for c, row in pivots:
        half, cc = (0, c) if c < wx * 64 else (1, c - wx * 64)
        word = half * wx + (cc >> 6)
        bit = np.uint64(cc & 63)
        if (v[word] >> bit) & np.uint64(1):
            v ^= R[row]
    return v


class _Scramble:
    """Per-trial symplectic-weight-preserving scramble: for a random subset of
    qubits swap the X/Z columns (Hadamard-like), for another subset shear
    ``z_i ^= x_i`` (S-like). Applied on uint8 halves; invertible."""

    def __init__(self, W: int, rng: np.random.Generator):
        self.swap = rng.random(W) < 0.5
        self.shear = rng.random(W) < 0.5

    def apply(self, X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, Z = X.copy(), Z.copy()
        Z[:, self.shear] ^= X[:, self.shear]          # S: (x, z) -> (x, z^x)
        Xs, Zs = X.copy(), Z.copy()
        Xs[:, self.swap], Zs[:, self.swap] = Z[:, self.swap], X[:, self.swap]  # H
        return Xs, Zs

    def undo(self, X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, Z = X.copy(), Z.copy()
        X[:, self.swap], Z[:, self.swap] = Z[:, self.swap].copy(), X[:, self.swap].copy()
        Z[:, self.shear] ^= X[:, self.shear]
        return X, Z


# --------------------------------------------------------------------------- #
# Pure-sector CSS reduction.
# --------------------------------------------------------------------------- #
def pure_sector_css(
    HX: np.ndarray, HZ: np.ndarray, sector: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSS-shaped reduction of the pure-``sector`` subproblem of a general
    stabilizer code.

    For ``sector='x'``: a pure-X operator ``(x, 0)`` is in the centralizer iff
    ``x ⊥`` every generator's Z-part, and is a stabilizer iff it equals
    ``c·HX`` for some combo with ``c·HZ = 0``. Hence the minimum weight of a
    pure-X nontrivial logical equals the **dx** of the CSS pair
    ``(Hx = S_pure, Hz = HZ)`` — certifiable with any CSS estimator (sqetch).
    Dually for ``sector='z'`` (use the returned triple as
    ``(Hz = S_pure, Hx = HX)`` and read **dz**).

    Args:
        HX, HZ: ``(m, W)`` uint8 generator parts.
        sector: ``'x'`` or ``'z'``.

    Returns:
        ``(S_pure, H_opp, reps)``: the pure-sector stabilizer basis (r, W),
        the opposite-part constraint matrix (m, W), and a basis of the
        pure-sector logical representatives (may be empty ``(0, W)``).
    """
    if sector not in ("x", "z"):
        raise ValueError(f"sector must be 'x' or 'z'; got {sector!r}")
    HX, HZ = _f2(HX), _f2(HZ)
    W = HX.shape[1]
    opp = HZ if sector == "x" else HX
    own = HX if sector == "x" else HZ
    cent = _nullspace_u8(opp)                # pure-sector centralizer basis
    combos = _nullspace_u8(opp.T)            # row combos with zero opp-part
    S_pure = ((combos.astype(np.int64) @ own) % 2).astype(np.uint8) \
        if combos.shape[0] else np.zeros((0, W), np.uint8)
    # reps: cent rows independent modulo S_pure (incremental packed RREF)
    SP, wx = _pack_pair(S_pure, np.zeros_like(S_pure))
    SP, piv = _packed_rref(SP, np.arange(2 * wx * 64), wx)
    acc, pivs = SP[: len(piv)].copy(), list(piv)
    reps: list[np.ndarray] = []
    centP, _ = _pack_pair(cent, np.zeros_like(cent))
    for i in range(centP.shape[0]):
        red = _reduce_vec(centP[i], acc, pivs, wx)
        if red.any():
            reps.append(cent[i])
            w = int(np.flatnonzero(red)[0])
            c = w * 64 + (int(red[w]).bit_length() - 1)
            m2 = ((acc[:, c >> 6] >> np.uint64(c & 63)) & np.uint64(1)).astype(bool)
            if m2.any():
                acc[m2] ^= red
            acc = np.vstack([acc, red[None, :]])
            pivs.append((c, acc.shape[0] - 1))
    R = np.array(reps, np.uint8) if reps else np.zeros((0, W), np.uint8)
    return S_pure, opp, R


# --------------------------------------------------------------------------- #
# Public estimators.
# --------------------------------------------------------------------------- #
def estimate_stab_distance(
    HX: np.ndarray,
    HZ: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    seed: Optional[int] = None,
    pair_passes: int = 2,
    verbose: bool = False,
) -> dict:
    """Estimate the distance of the stabilizer code with generators
    ``X(HX[i])·Z(HZ[i])`` by random information-set sampling.

    Per trial: scramble (qubit-local, weight-preserving), pack, full RREF of a
    centralizer basis over a random column order, then take the minimum
    symplectic weight over reduced rows that are NOT in the stabilizer span,
    improved by ``pair_passes`` greedy row-combination sweeps. The global
    minimum over trials is returned as ``d_est`` (an upper bound on the true
    distance that converges with trials).

    Args:
        HX, HZ: ``(m, W)`` uint8 — X/Z parts of the stabilizer generators
            (redundant rows fine).
        num_trials: information-set trials to run.
        d_target: strict-``<`` early stop — return as soon as a nontrivial
            logical of weight ``< d_target`` is seen (distance collapse
            witness). ``None`` disables.
        seed: RNG seed (recorded in the result).
        pair_passes: greedy improvement sweeps combining the current best row
            with every other row (0 disables).
        verbose: print progress every ~10% of trials.

    Returns:
        dict with ``d_est`` (int), ``witness_x``/``witness_z`` (uint8 vectors
        achieving it, ORIGINAL coordinates), ``trials_run``, ``num_trials``,
        ``seed``, ``early_stopped`` (bool), ``k`` (logical count).

    Raises:
        ValueError on shape mismatch or a k=0 code (no logicals to weigh).
    """
    HX, HZ = _f2(HX), _f2(HZ)
    if HX.shape != HZ.shape:
        raise ValueError(f"HX {HX.shape} and HZ {HZ.shape} must match.")
    W = HX.shape[1]
    k = stab_k(HX, HZ)
    if k <= 0:
        raise ValueError(f"code has k={k}; distance undefined (no logicals).")

    # Split the centralizer into stabilizer rows + 2k logical representatives
    # (independent mod S). During elimination each row carries a TAG block
    # recording which logical reps it contains: tag != 0  <=>  the row is a
    # nontrivial logical. This replaces per-candidate membership tests — for
    # LDPC codes the lightest centralizer rows are overwhelmingly stabilizers,
    # so candidates MUST be selected in the quotient.
    CX, CZ = centralizer_basis(HX, HZ)
    SP0, wx = _pack_pair(HX, HZ)
    SP0, spiv0 = _packed_rref(SP0, np.arange(2 * wx * 64), wx)
    CP, _ = _pack_pair(CX, CZ)
    lreps: list[int] = []
    acc = SP0[: len(spiv0)].copy()   # nonzero RREF rows only
    piv = list(spiv0)
    for i in range(CP.shape[0]):
        red = _reduce_vec(CP[i], acc, piv, wx)
        if red.any():
            lreps.append(i)
            # incremental RREF insert: pick any set bit of red as its pivot,
            # eliminate that bit from every existing row, then append.
            w = int(np.flatnonzero(red)[0])
            c = w * 64 + (int(red[w]).bit_length() - 1)
            bit = np.uint64(c & 63)
            word = c >> 6
            mask = ((acc[:, word] >> bit) & np.uint64(1)).astype(bool)
            if mask.any():
                acc[mask] ^= red
            acc = np.vstack([acc, red[None, :]])
            piv.append((c, acc.shape[0] - 1))
    if len(lreps) != 2 * k:
        raise RuntimeError(f"logical rep extraction found {len(lreps)} != 2k={2 * k}.")
    Lx_reps, LZ_reps = CX[lreps], CZ[lreps]

    GX = np.vstack([HX, Lx_reps])
    GZ = np.vstack([HZ, LZ_reps])
    n_rows = GX.shape[0]
    n_tag = 2 * k
    wt = (n_tag + 63) // 64
    TAGS = np.zeros((n_rows, wt), dtype=np.uint64)
    for j in range(n_tag):
        TAGS[HX.shape[0] + j, j >> 6] |= np.uint64(1) << np.uint64(j & 63)

    # Pure-sector ensembles: for X (dually Z), the pure-X centralizer is
    # ker(all Z-parts); the pure-X stabilizer subgroup is {c·GX : c·GZ = 0}.
    # Sector reps = centralizer mod that subgroup, tagged like the full
    # ensemble. For CSS(-ish) codes the minimum-weight logical is (near-)pure,
    # and the classical single-half information sets converge far faster than
    # the full symplectic ones — the full ensemble stays in the mix for
    # genuinely mixed logicals.
    ensembles = [dict(GX=GX, GZ=GZ, TAGS=TAGS, mode="sym")]
    for mode in ("x", "z"):
        Spure, _, R = pure_sector_css(HX, HZ, mode)
        if R.shape[0] == 0:
            continue
        stack = np.vstack([Spure, R])
        gx_s = stack if mode == "x" else np.zeros_like(stack)
        gz_s = np.zeros_like(stack) if mode == "x" else stack
        nt_s = R.shape[0]
        wt_s = (nt_s + 63) // 64
        T_s = np.zeros((stack.shape[0], wt_s), np.uint64)
        for j in range(nt_s):
            T_s[Spure.shape[0] + j, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
        ensembles.append(dict(GX=gx_s, GZ=gz_s, TAGS=T_s, mode=mode))

    rng = np.random.default_rng(seed)
    best = np.iinfo(np.int64).max
    best_xz: Optional[tuple[np.ndarray, np.ndarray]] = None
    early = False
    trials_run = 0

    for t in range(num_trials):
        trials_run = t + 1
        ens = ensembles[t % len(ensembles)]
        gx_t, gz_t, tags_t, mode = ens["GX"], ens["GZ"], ens["TAGS"], ens["mode"]
        if mode == "sym":
            sc = _Scramble(W, rng)
            Xs, Zs = sc.apply(gx_t, gz_t)
            cols = np.concatenate([rng.permutation(W), W + rng.permutation(W)])
            rng.shuffle(cols)
            cols = np.where(cols < W, cols, cols - W + wx * 64)
        else:
            sc = None
            Xs, Zs = gx_t, gz_t
            cols = rng.permutation(W)
            if mode == "z":
                cols = cols + wx * 64
        P0, _ = _pack_pair(Xs, Zs)
        P = np.hstack([P0, tags_t.copy()])
        P, _ = _packed_rref(P, cols, wx)

        real = P[:, : 2 * wx]
        tags = P[:, 2 * wx:]
        wts = np.bitwise_count(real[:, :wx] | real[:, wx:]).sum(axis=1).astype(np.int64)
        is_log = np.bitwise_count(tags).sum(axis=1) > 0
        cand = np.flatnonzero(is_log & (wts > 0) & (wts < best))
        if cand.size:
            for i in cand[np.argsort(wts[cand])][:4]:
                v = P[int(i)].copy()
                # greedy pair improvement (keep the tag nonzero)
                for _ in range(pair_passes):
                    xor = v[None, :] ^ P
                    trial_w = np.bitwise_count(xor[:, :wx] | xor[:, wx:2 * wx]).sum(axis=1)
                    tag_ok = np.bitwise_count(xor[:, 2 * wx:]).sum(axis=1) > 0
                    trial_w = np.where(tag_ok & (trial_w > 0), trial_w, 1 << 30)
                    j = int(trial_w.argmin())
                    if int(trial_w[j]) < int(np.bitwise_count(v[:wx] | v[wx:2 * wx]).sum()):
                        v ^= P[j]
                    else:
                        break
                wv = int(np.bitwise_count(v[:wx] | v[wx:2 * wx]).sum())
                if 0 < wv < best:
                    xv, zv = _unpack_pair(v[None, : 2 * wx], wx, W)
                    if sc is not None:
                        xv, zv = sc.undo(xv, zv)
                    best = wv
                    best_xz = (xv[0].copy(), zv[0].copy())
                    if verbose:
                        print(f"  trial {t} [{mode}]: new best d <= {best}")
                    if d_target is not None and best < d_target:
                        early = True
                        break
        if early:
            break
        if verbose and num_trials >= 10 and t % max(1, num_trials // 10) == 0:
            print(f"  trial {t}/{num_trials}, best={best if best_xz else None}")

    if best_xz is None:  # pragma: no cover - k>0: every trial has logical rows
        raise RuntimeError("no nontrivial logical sampled; increase num_trials.")
    return dict(
        d_est=int(best),
        witness_x=best_xz[0],
        witness_z=best_xz[1],
        trials_run=int(trials_run),
        num_trials=int(num_trials),
        seed=seed,
        early_stopped=bool(early),
        k=int(k),
    )


def estimate_coset_min_weight(
    GX: np.ndarray,
    GZ: np.ndarray,
    vx: np.ndarray,
    vz: np.ndarray,
    *,
    num_trials: int,
    d_target: Optional[int] = None,
    seed: Optional[int] = None,
    greedy_passes: int = 4,
) -> dict:
    """Minimum symplectic weight over the AFFINE coset ``v + rowspan(G)``.

    Used for the masked measurement-fault distances of the Y gadget:
    ``G = [S; Init; L_surviving]`` rows and ``v = X̄`` (or ``Z̄``) embedded —
    the result lower-bounds nothing by itself (it is an upper-bound estimate
    of the coset minimum, converging with trials); certification asserts the
    estimate stayed ``>= d`` after ``num_trials`` with no strict-``<`` hit.

    Args / returns / semantics mirror :func:`estimate_stab_distance` (dict
    with ``w_est``, ``witness_x/z``, ``trials_run``, ``num_trials``, ``seed``,
    ``early_stopped``); ``v`` itself must NOT be in ``rowspan(G)`` (raises —
    the coset minimum would be 0 and the masked operator trivial).
    """
    GX, GZ = _f2(GX), _f2(GZ)
    vx, vz = _f2(vx).ravel(), _f2(vz).ravel()
    W = GX.shape[1]
    rng = np.random.default_rng(seed)

    # v in rowspan(G) check (natural order, packed).
    GP, wx = _pack_pair(GX, GZ)
    GP0, gpiv0 = _packed_rref(GP.copy(), np.arange(2 * wx * 64), wx)
    vp0, _ = _pack_pair(vx[None, :], vz[None, :])
    if not _reduce_vec(vp0[0], GP0, gpiv0, wx).any():
        raise ValueError("v is in rowspan(G); coset minimum is 0 (trivial).")

    best = np.iinfo(np.int64).max
    best_xz: Optional[tuple[np.ndarray, np.ndarray]] = None
    early = False
    trials_run = 0

    for t in range(num_trials):
        trials_run = t + 1
        sc = _Scramble(W, rng)
        Xs, Zs = sc.apply(GX, GZ)
        xvs, zvs = sc.apply(vx[None, :], vz[None, :])
        P, _ = _pack_pair(Xs, Zs)
        vp, _ = _pack_pair(xvs, zvs)
        v = vp[0]
        cols = np.concatenate([rng.permutation(W), W + rng.permutation(W)])
        rng.shuffle(cols)
        cols = np.where(cols < W, cols, cols - W + wx * 64)
        P, piv = _packed_rref(P, cols, wx)
        v = _reduce_vec(v, P, piv, wx)
        # greedy improvement sweeps
        for _ in range(greedy_passes):
            trial_w = np.bitwise_count((v[None, :] ^ P)[:, :wx]
                                       | (v[None, :] ^ P)[:, wx:]).sum(axis=1)
            j = int(trial_w.argmin())
            if int(trial_w[j]) < int(np.bitwise_count(v[:wx] | v[wx:]).sum()):
                v ^= P[j]
            else:
                break
        wv = int(np.bitwise_count(v[:wx] | v[wx:]).sum())
        if wv < best:
            xv, zv = _unpack_pair(v[None, :], wx, W)
            xo, zo = sc.undo(xv, zv)
            best = wv
            best_xz = (xo[0].copy(), zo[0].copy())
            if d_target is not None and best < d_target:
                early = True
                break

    assert best_xz is not None
    return dict(
        w_est=int(best),
        witness_x=best_xz[0],
        witness_z=best_xz[1],
        trials_run=int(trials_run),
        num_trials=int(num_trials),
        seed=seed,
        early_stopped=bool(early),
    )
