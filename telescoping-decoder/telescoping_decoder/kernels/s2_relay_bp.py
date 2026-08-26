"""The S2 belief-propagation kernel: relay-BP with coset-quorum acceptance.

Implements relay-BP with per-iteration memory: an ordered leg followed by legs with
per-variable signed memory, each warm-starting from the previous leg's
marginals), plus a coset-quorum acceptance gate: a shot is accepted only when
at least ``quorum`` converged legs agree on the logical coset ``L @ e mod 2``,
computed on the GPU. Legs that satisfy every check but disagree on the coset
cause the shot to be deferred to S3/S4.

Operates on a plain (H, L, priors) system with full-row convergence — used by
the S2 stage for the original and init-detector systems. The GARI variant
lives in ``s2_relay_bp_gari.py``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import cupy as cp


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_DC_PAD = 160
_DV_PAD = 24
_BWARP = 32
_N_WARPS = 32
_REBATCH_BUCKETS = (32, 64, 96, 128, 256, 512)


# -----------------------------------------------------------------------------
# Kernel sources
# -----------------------------------------------------------------------------

# Non-templated kernels: residual, per-shot weight reduction, select kernels.
_NONTEMPLATED_KERNELS_SRC = r"""
/** @brief Count parity-check violations for every shot.
 *  @param[in] posterior,c_neighbors_real,c_valid_real,synd Beliefs and checks.
 *  @param[in] B,m,n,DC_REAL Batch and matrix dimensions.
 *  @param[in,out] residual Per-shot count; must start at zero.
 *  @param[out] unsat Per-check violation flags.
 */
extern "C" __global__ void residual_count(
    const float* __restrict__ posterior,         // (n, B)
    const int*   __restrict__ c_neighbors_real,  // (m, dc_max_real)
    const int*   __restrict__ c_valid_real,      // (m, dc_max_real)
    const unsigned char* __restrict__ synd,      // (m, B)
    const int B, const int m, const int n, const int DC_REAL,
    int* __restrict__ residual,                  // (B,)
    unsigned char* __restrict__ unsat            // (m, B)
) {
    int c = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= m) return;
    const int* nbr = c_neighbors_real + (long long)c * DC_REAL;
    const int* val = c_valid_real + (long long)c * DC_REAL;
    int s = 0;
    for (int k = 0; k < DC_REAL; k++) {
        int v = nbr[k];
        if (val[k] && v < n) {
            if (posterior[(long long)v * B + b] < 0.0f) s ^= 1;
        }
    }
    int s_target = synd[(long long)c * B + b];
    int u = (s != s_target) ? 1 : 0;
    unsat[(long long)c * B + b] = (unsigned char)u;
    if (u) atomicAdd(residual + b, 1);
}

// Per-shot correction weight: w[b] = sum_v (posterior[v,b] < 0) * chan_w[v].
// One block per shot; threads stride over variables; shared-mem reduction.
/** @brief Compute the weighted Hamming cost of each hard decision.
 *  @param[in] posterior,channel_llr Beliefs and variable costs.
 *  @param[in] n,B Variable count and batch size.
 *  @param[out] weight One reduced candidate cost per shot.
 */
extern "C" __global__ void compute_weight(
    const float* __restrict__ posterior,  // (n, B)
    const float* __restrict__ chan_w,      // (n,)  (channel_llr, all > 0)
    const int n, const int B,
    float* __restrict__ weight             // (B,)
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float local = 0.0f;
    for (int v = tid; v < n; v += nthreads) {
        if (posterior[(long long)v * B + b] < 0.0f) local += chan_w[v];
    }
    extern __shared__ float sdata[];
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) weight[b] = sdata[0];
}

// ---- selection: leg-0 init ----
//   best_res[b]    = residual[b]
//   n_conv[b]      = (residual==0) ? 1 : 0
//   best_weight[b] = (residual==0) ? weight[b] : +inf
//   found[b]       = (n_conv >= stop_nconv)
/** @brief Initialize per-shot selection state from relay leg zero.
 *  @param[in] residual,weight,B,stop_nconv Initial scores and freeze threshold.
 *  @param[out] best_res,best_weight,n_conv,found Selection state arrays.
 */
extern "C" __global__ void select_perb_init(
    const int* __restrict__ residual, const float* __restrict__ weight,
    const int B, const int stop_nconv,
    int* __restrict__ best_res, float* __restrict__ best_weight,
    int* __restrict__ n_conv, unsigned char* __restrict__ found
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int r = residual[b];
    bool conv = (r == 0);
    best_res[b] = r;
    int nc = conv ? 1 : 0;
    n_conv[b] = nc;
    best_weight[b] = conv ? weight[b] : 3.4e38f;
    found[b] = (nc >= stop_nconv) ? 1 : 0;
}

// ---- selection: leg-0 per-(v,b) init ----
//   best_post = current_post = posterior; corr = (posterior < 0)
/** @brief Store leg zero as the initial candidate and warm-start state.
 *  @param[in] posterior,n,B Initial posterior and dimensions.
 *  @param[out] best_post,current_post,corr Candidate beliefs and hard bits.
 */
extern "C" __global__ void select_var_init(
    const float* __restrict__ posterior, const int n, const int B,
    float* __restrict__ best_post, float* __restrict__ current_post,
    unsigned char* __restrict__ corr
) {
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    long long off = (long long)v * B + b;
    float p = posterior[off];
    best_post[off]    = p;
    current_post[off] = p;
    corr[off] = (p < 0.0f) ? 1 : 0;
}

// ---- selection: leg-N per-(v,b) step (reads PRE-step per-shot scalars) ----
// Must run BEFORE select_perb_step in the same stream (intra-stream ordering).
//   take_conv = (residual==0) && (weight < best_weight)
//   take_nc   = (residual!=0) && (n_conv==0) && (residual < best_res)
//   if take_conv||take_nc: best_post = posterior; corr = (posterior<0)
//   current_post = posterior   unless the shot is frozen AFTER this leg
/** @brief Copy a later leg when it improves the selected candidate.
 *  @param[in] posterior,residual,weight,best_res,best_weight,n_conv,found Scores.
 *  @param[in] stop_nconv,n,B Freeze threshold and dimensions.
 *  @param[in,out] best_post,current_post,corr Selected and warm-start state.
 */
extern "C" __global__ void select_var_step(
    const float* __restrict__ posterior,
    const int*   __restrict__ residual,
    const float* __restrict__ weight,
    const int*   __restrict__ best_res,
    const float* __restrict__ best_weight,
    const int*   __restrict__ n_conv,
    const unsigned char* __restrict__ found,
    const int stop_nconv, const int n, const int B,
    float* __restrict__ best_post,
    float* __restrict__ current_post,
    unsigned char* __restrict__ corr
) {
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    if (found[b]) return;            // frozen: no updates
    int r = residual[b];
    float w = weight[b];
    int nc = n_conv[b];
    bool conv = (r == 0);
    bool take_conv = conv && (w < best_weight[b]);
    bool take_nc   = (!conv) && (nc == 0) && (r < best_res[b]);
    long long off = (long long)v * B + b;
    float p = posterior[off];
    if (take_conv || take_nc) {
        best_post[off] = p;
        corr[off] = (p < 0.0f) ? 1 : 0;
    }
    int nc_after = nc + (conv ? 1 : 0);
    bool frozen_after = (nc_after >= stop_nconv);
    if (!frozen_after) current_post[off] = p;
}

// ---- selection: leg-N per-shot step (runs AFTER select_var_step) ----
/** @brief Commit per-shot scores after the variable candidate update.
 *  @param[in] residual,weight,B,stop_nconv Current scores and threshold.
 *  @param[in,out] best_res,best_weight,n_conv,found Selection state.
 */
extern "C" __global__ void select_perb_step(
    const int* __restrict__ residual, const float* __restrict__ weight,
    const int B, const int stop_nconv,
    int* __restrict__ best_res, float* __restrict__ best_weight,
    int* __restrict__ n_conv, unsigned char* __restrict__ found
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (found[b]) return;
    int r = residual[b];
    float w = weight[b];
    int nc = n_conv[b];
    bool conv = (r == 0);
    bool take_conv = conv && (w < best_weight[b]);
    bool take_nc   = (!conv) && (nc == 0) && (r < best_res[b]);
    if (take_conv) best_weight[b] = w;
    if (take_nc)   best_res[b] = r;
    if (conv) { nc += 1; n_conv[b] = nc; }
    found[b] = (nc >= stop_nconv) ? 1 : 0;
}
"""


# Templated kernels: check + resync + memory_iter_init.
_TEMPLATED_KERNELS_SRC = r"""
#define DC_PAD {dc_pad}
#define DV_PAD {dv_pad}
#define BW {bwarp}
#define N_WARPS {n_warps}
#define LB_MIN_BLOCKS {lb_min_blocks}

/** @brief Apply one layered min-sum check update.
 *  @param[in] c2e,c_neigh,check_order,c_degrees,synd Check graph and targets.
 *  @param[in] alpha,B,m,n,E,layer_start Update constants and dimensions.
 *  @param[in] posterior Current variable beliefs.
 *  @param[in,out] c2v Check-to-variable messages for this layer.
 */
extern "C" __global__ __launch_bounds__(BW * N_WARPS, LB_MIN_BLOCKS)
void layered_check_compute_inplace(
    const int*   __restrict__ c2e,
    const int*   __restrict__ c_neigh,
    const int*   __restrict__ check_order,
    const int*   __restrict__ c_degrees,
    const unsigned char* __restrict__ synd,
    const float alpha,
    const int B, const int m, const int n, const int E,
    const int layer_start,
    const float* __restrict__ posterior,
    float* __restrict__ c2v
) {{
    int c_pos = layer_start + blockIdx.x;
    if (c_pos >= m) return;
    int c = check_order[c_pos];
    int dc_c = c_degrees[c];

    int tx  = threadIdx.x;
    int wid = threadIdx.y;
    int b   = blockIdx.y * BW + tx;

    int k_start = (dc_c * wid) / N_WARPS;
    int k_end   = (dc_c * (wid + 1)) / N_WARPS;

    extern __shared__ unsigned char smem_raw[];
    int*   sh_c2e         = (int*)smem_raw;
    int*   sh_nbr         = sh_c2e + DC_PAD;
    float* part_min1      = (float*)(sh_nbr + DC_PAD);
    float* part_min2      = part_min1 + N_WARPS * BW;
    int*   part_argmin1   = (int*)(part_min2 + N_WARPS * BW);
    int*   part_count_neg = part_argmin1 + N_WARPS * BW;

    int total_threads = BW * N_WARPS;
    int linear_tid = wid * BW + tx;
    for (int i = linear_tid; i < dc_c; i += total_threads) {{
        sh_c2e[i] = c2e[c * DC_PAD + i];
        sh_nbr[i] = c_neigh[c * DC_PAD + i];
    }}
    __syncthreads();

    if (b >= B) return;
    int s_idx = synd[(long long)c * B + b];

    float min1 = 1e30f, min2 = 1e30f;
    int argmin1 = -1;
    unsigned long long my_signs = 0ULL;
    int slice_idx = 0;
    #pragma unroll 4
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        int v_idx = sh_nbr[k];
        float v = posterior[(long long)v_idx * B + b] - c2v[(long long)e * B + b];
        my_signs |= ((unsigned long long)(v < 0.0f) << slice_idx);
        float av = fabsf(v);
        if (av < min1) {{ min2 = min1; min1 = av; argmin1 = k; }}
        else if (av < min2) {{ min2 = av; }}
        slice_idx++;
    }}
    int count_neg = __popcll(my_signs);

    part_min1     [wid * BW + tx] = min1;
    part_min2     [wid * BW + tx] = min2;
    part_argmin1  [wid * BW + tx] = argmin1;
    part_count_neg[wid * BW + tx] = count_neg;
    __syncthreads();

    if (wid == 0) {{
        #pragma unroll
        for (int w = 1; w < N_WARPS; w++) {{
            float w_min1      = part_min1     [w * BW + tx];
            float w_min2      = part_min2     [w * BW + tx];
            int   w_argmin1   = part_argmin1  [w * BW + tx];
            int   w_count_neg = part_count_neg[w * BW + tx];
            count_neg += w_count_neg;
            if (w_min1 < min1) {{
                float new_min2 = (min1 < w_min2) ? min1 : w_min2;
                min1 = w_min1; argmin1 = w_argmin1; min2 = new_min2;
            }} else if (w_min1 == min1) {{
                int new_argmin1 = (w_argmin1 < argmin1) ? w_argmin1 : argmin1;
                float new_min2 = (min2 < w_min2) ? min2 : w_min2;
                if (w_min1 < new_min2) new_min2 = w_min1;
                argmin1 = new_argmin1; min2 = new_min2;
            }} else {{
                float new_min2 = (min2 < w_min1) ? min2 : w_min1;
                min2 = new_min2;
            }}
        }}
        part_min1     [tx] = min1;
        part_min2     [tx] = min2;
        part_argmin1  [tx] = argmin1;
        part_count_neg[tx] = count_neg;
    }}
    __syncthreads();

    min1      = part_min1     [tx];
    min2      = part_min2     [tx];
    argmin1   = part_argmin1  [tx];
    count_neg = part_count_neg[tx];

    int parity = (count_neg + s_idx) & 1;
    float pre = (parity ? -alpha : alpha);

    int slice_idx2 = 0;
    #pragma unroll 4
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        float min_excl = (k == argmin1) ? min2 : min1;
        float sign_k = ((my_signs >> slice_idx2) & 1ULL) ? -1.0f : 1.0f;
        float new_msg = pre * sign_k * min_excl;
        c2v[(long long)e * B + b] = new_msg;
        slice_idx2++;
    }}
}}

/** @brief Recompute posteriors for variables touched by a check layer.
 *  @param[in] c2v,v2e,prior,subset_idx Messages, graph, priors, and variables.
 *  @param[in] B,n_layer,E,DV_unused Batch and graph dimensions.
 *  @param[out] posterior Prior plus all incident check messages.
 */
extern "C" __global__ void layered_var_resync_subset(
    const float* __restrict__ c2v,
    const int*   __restrict__ v2e,
    const float* __restrict__ prior,
    const int*   __restrict__ subset_idx,
    const int B, const int n_layer, const int E, const int DV_unused,
    float* __restrict__ posterior
) {{
    int blk = blockIdx.x;
    if (blk >= n_layer) return;
    int v = subset_idx[blk];
    int tx = threadIdx.x;
    int b = blockIdx.y * BW + tx;

    extern __shared__ int sh_v2e[];
    #pragma unroll
    for (int i = 0; i < (DV_PAD + BW - 1) / BW; i++) {{
        int kk = tx + i * BW;
        if (kk < DV_PAD) {{
            sh_v2e[kk] = v2e[v * DV_PAD + kk];
        }}
    }}
    __syncthreads();

    if (b >= B) return;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < DV_PAD; k++) {{
        sum += c2v[(long long)sh_v2e[k] * B + b];
    }}
    posterior[(long long)v * B + b] = prior[(long long)v * B + b] + sum;
}}

// Relay-BP per-iteration memory term over all variables.
//   bias = Lambda(t) = (1 - gamma[v]) * prior_in[v,b] + gamma[v] * M_prev[v,b]
//   prior_out[v,b] = bias                 (per-iteration bias for resyncs)
//   posterior[v,b] = bias + sum_e(c2v)    (seeds the layered sweep this iter)
/** @brief Initialize one BP iteration from channel prior and relay memory.
 *  @param[in] prior_in,M_prev,gamma,c2v,v2e Priors, memory, weights, and graph.
 *  @param[in] B,n,E Batch and graph dimensions.
 *  @param[out] prior_out Mixed per-iteration bias.
 *  @param[out] posterior Bias plus current check messages.
 */
extern "C" __global__ void memory_iter_init(
    const float* __restrict__ prior_in,   // Lambda0 (n, B)
    const float* __restrict__ M_prev,      // (n, B)
    const float* __restrict__ gamma,       // (n,)
    const float* __restrict__ c2v,         // (E+1, B)
    const int*   __restrict__ v2e,         // (n, DV_PAD)
    const int B, const int n, const int E,
    float* __restrict__ prior_out,         // (n, B)
    float* __restrict__ posterior          // (n, B)
) {{
    int v = blockIdx.x;
    int b = blockIdx.y * BW + threadIdx.x;
    if (b >= B || v >= n) return;
    long long off = (long long)v * B + b;
    float g = gamma[v];
    float bias = (1.0f - g) * prior_in[off] + g * M_prev[off];
    float sum = 0.0f;
    const int* e = v2e + (long long)v * DV_PAD;
    #pragma unroll
    for (int k = 0; k < DV_PAD; k++) {{
        sum += c2v[(long long)e[k] * B + b];
    }}
    prior_out[off] = bias;
    posterior[off] = bias + sum;
}}
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_arrays(H, dc_pad: int, dv_pad: int):
    if sp.issparse(H):
        csr = H.tocsr().astype(np.uint8)
    else:
        csr = sp.csr_matrix(np.asarray(H, dtype=np.uint8))
    csr.sort_indices()
    csr.eliminate_zeros()
    m, n = csr.shape
    indptr = csr.indptr.astype(np.int32)
    indices = csr.indices.astype(np.int32)
    nnz = csr.nnz
    dc_real = int(np.diff(indptr).max())
    if dc_real > dc_pad:
        raise ValueError(f"dc_max={dc_real} > dc_pad={dc_pad}")
    csc = csr.tocsc()
    csc_indptr = csc.indptr.astype(np.int32)
    csc_indices = csc.indices.astype(np.int32)
    dv_real = int(np.diff(csc_indptr).max())
    if dv_real > dv_pad:
        raise ValueError(f"dv_max={dv_real} > dv_pad={dv_pad}")

    edge_v = indices
    c2e = np.full((m, dc_pad), nnz, dtype=np.int32)
    c_neigh_padded = np.full((m, dc_pad), n, dtype=np.int32)
    c_neighbors = np.full((m, dc_real), n, dtype=np.int32)
    c_valid = np.zeros((m, dc_real), dtype=np.int32)
    for c in range(m):
        s, e = indptr[c], indptr[c + 1]
        d = e - s
        c2e[c, :d] = np.arange(s, e)
        c_neigh_padded[c, :d] = indices[s:e]
        c_neighbors[c, :d] = indices[s:e]
        c_valid[c, :d] = 1

    v2e = np.full((n, dv_pad), nnz, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    for ei in range(nnz):
        v = edge_v[ei]
        v2e[v, counts[v]] = ei
        counts[v] += 1

    v_checks = np.full((n, dv_real), m, dtype=np.int32)
    v_valid = np.zeros((n, dv_real), dtype=np.int32)
    for v in range(n):
        s, e = csc_indptr[v], csc_indptr[v + 1]
        v_checks[v, :e - s] = csc_indices[s:e]
        v_valid[v, :e - s] = 1

    return dict(
        m=m, n=n, nnz=nnz, dc_pad=dc_pad, dv_pad=dv_pad, dc_real=dc_real,
        dv_real=dv_real,
        edge_v=edge_v, c2e=c2e, v2e=v2e,
        c_neigh_padded=c_neigh_padded,
        c_neighbors=c_neighbors, c_valid=c_valid,
        v_checks=v_checks, v_valid=v_valid,
    )


def _alpha_schedule(max_iter: int, alpha_max: float, tau: float,
                    const: bool) -> np.ndarray:
    """Constant alpha (IBM alpha_iteration_scaling_factor=1.0) or an exp ramp."""
    if const:
        return np.full(max_iter, np.float32(alpha_max), dtype=np.float32)
    return np.array(
        [alpha_max * (1.0 - np.exp(-(t + 1) / tau)) for t in range(max_iter)],
        dtype=np.float32,
    )


def _layer_ranges(m: int, K: int):
    if K < 1 or K > m:
        raise ValueError(f"K must satisfy 1<=K<=m; got K={K}, m={m}")
    base = m // K
    rem = m - base * K
    ranges = []
    s = 0
    for k in range(K):
        sz = base + (1 if k < rem else 0)
        ranges.append((s, s + sz))
        s += sz
    return ranges


def _build_layer_touched_vars(H: sp.spmatrix, K: int):
    H = sp.csr_matrix(H)
    m = H.shape[0]
    layer_ranges = _layer_ranges(m, K)
    pieces = []
    offsets = np.zeros(K + 1, dtype=np.int32)
    for k, (ls, le) in enumerate(layer_ranges):
        rows = H[ls:le]
        v_uniq = np.unique(rows.indices).astype(np.int32)
        pieces.append(v_uniq)
        offsets[k + 1] = offsets[k] + v_uniq.size
    flat = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32)
    return flat.astype(np.int32), offsets


def _bucket_size(target: int, buckets=_REBATCH_BUCKETS) -> int:
    for b in buckets:
        if b >= target:
            return b
    return buckets[-1]


# -----------------------------------------------------------------------------
# Phase config
# -----------------------------------------------------------------------------

@dataclass
class S2RelayPhase:
    """Relay-BP phase with one ordered leg followed by disordered legs."""
    name: str
    priors: list                  # [(prior_scale, noise_std)]
    num_sets: int = 40            # disordered relay legs (after leg 0)
    leg_max_iter: int = 60        # iterations per disordered leg (IBM Tᵣ)
    gamma0: float = 0.1           # ordered memory strength for leg 0
    gamma0_iters: int = 80        # iterations for leg 0 (IBM T₀)
    gamma_center: float = 0.21    # disordered γ ~ U[center−width/2, center+width/2]
    gamma_width: float = 0.90     # benchmark range [-0.24, 0.66]; tune per graph
    alpha: float = 1.0            # min-sum α base (alpha_iteration_scaling_factor)
    alpha_const: bool = True      # True = flat alpha (IBM); False = exp ramp
    tau: float = 5.0              # exp-ramp time constant (only if alpha_const=False)
    quorum: int = 2               # collect exactly this many converged legs;
                                  # this is also the freeze threshold. Accept if all agree
                                  # on the coset, else NC.

    @property
    def n_runs(self) -> int:
        return self.num_sets + 1  # leg 0 + disordered legs


DEFAULT_S2_PHASE = S2RelayPhase(
    "S2", [(1.0, 0.0)],
    num_sets=40, leg_max_iter=60,
    gamma0=0.1, gamma0_iters=80,
    gamma_center=0.21, gamma_width=0.90,
    alpha=1.0, alpha_const=True, quorum=2,
)


@dataclass
class _GraphBundle:
    """Per-(slot, B) bundle: persistent device buffers + 2 cuda graphs."""
    leg0_graph: object
    legN_graph: object
    # BP working buffers
    synd: cp.ndarray
    prior_in: cp.ndarray   # Λ₀ — immutable channel-scaled prior
    prior: cp.ndarray      # per-iteration bias Λ(t) (written by memory_iter_init)
    c2v: cp.ndarray
    posterior: cp.ndarray
    m_prev: cp.ndarray     # M(t-1) (warm-started across legs)
    # Residual / weight outputs
    residual: cp.ndarray
    unsat: cp.ndarray
    weight: cp.ndarray     # (B,) current-leg correction weight
    # Persistent selection accumulators
    best_post: cp.ndarray
    best_res: cp.ndarray
    best_weight: cp.ndarray  # (B,) min weight among converged legs
    n_conv: cp.ndarray       # (B,) number of converged legs so far
    corr: cp.ndarray
    found: cp.ndarray        # (B,) frozen flag = (n_conv >= quorum)
    current_post: cp.ndarray
    # Per-leg memory vector (host writes per leg)
    gamma: cp.ndarray
    B: int


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class S2RelayBP:
    """Block-layered relay-BP with per-iteration memory,
    i.i.d. signed γ, ordered gamma0 leg, marginal warm-start, min-weight
    selection). Standalone class."""

    BASE_SEED = 12345

    def __init__(
        self,
        H,
        channel_llr,
        *,
        obs_matrix,
        K: int = 64,
        phase: S2RelayPhase = DEFAULT_S2_PHASE,
        batch_size: int = 64,
        dc_pad: int = _DC_PAD,
        dv_pad: int = _DV_PAD,
        base_seed: int = 12345,
        rebatch: bool = True,
        rebatch_buckets: tuple = _REBATCH_BUCKETS,
        num_slots: int = 1,
        prebuild_bundles: bool = False,
        n_warps: int = _N_WARPS,
        lb_min_blocks: int | None = 1,
    ):
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1; got {num_slots}")
        if int(phase.quorum) < 1:
            raise ValueError(f"quorum must be >= 1; got {phase.quorum}")

        arr = _build_arrays(H, dc_pad, dv_pad)
        self.m = arr["m"]
        self.n = arr["n"]
        self.E = arr["nnz"]
        self.dc_pad = arr["dc_pad"]
        self.dv_pad = arr["dv_pad"]
        self.dc_real = arr["dc_real"]
        self.dv_real = arr["dv_real"]
        self.channel_llr = np.ascontiguousarray(channel_llr, dtype=np.float32)
        self.K = K
        self.phase = phase

        # Observable / logical matrix on the GPU (cuSPARSE), for per-leg coset
        # labels ob = L @ ê (mod 2). Mirrors the pipeline's LE-check pattern.
        import cupyx.scipy.sparse as cusp
        L_csr = sp.csr_matrix(obs_matrix)
        self.k = int(L_csr.shape[0])
        self._L_gpu = cusp.csr_matrix(L_csr.astype(np.float32))
        self.batch_size = batch_size
        self.BASE_SEED = base_seed
        self.rebatch = bool(rebatch)
        self.rebatch_buckets = tuple(rebatch_buckets)
        self.num_slots = int(num_slots)

        if int(n_warps) < 1 or (int(n_warps) & (int(n_warps) - 1)) != 0:
            raise ValueError(f"n_warps must be a positive power of 2; got {n_warps}")
        self.n_warps = int(n_warps)
        max_elems_per_warp = (int(self.dc_pad) + self.n_warps - 1) // self.n_warps
        if max_elems_per_warp > 64:
            raise ValueError(
                f"n_warps={self.n_warps} too small for dc_pad={self.dc_pad}: "
                f"max_elems_per_warp={max_elems_per_warp} exceeds 64-bit sign cache."
            )
        if lb_min_blocks is None:
            self.lb_min_blocks = max(1, 2048 // (self.n_warps * _BWARP))
        else:
            self.lb_min_blocks = int(lb_min_blocks)

        _layer_ranges(self.m, self.K)

        # Read-only device matrix arrays (shared across slots).
        self._c2e_d = cp.asarray(arr["c2e"])
        self._c_neigh_padded_d = cp.asarray(arr["c_neigh_padded"])
        self._v2e_d = cp.asarray(arr["v2e"])
        self._c_neighbors_d = cp.asarray(arr["c_neighbors"])
        self._c_valid_d = cp.asarray(arr["c_valid"])
        self._v_checks_d = cp.asarray(arr["v_checks"])
        self._v_valid_d = cp.asarray(arr["v_valid"])

        H_csr = sp.csr_matrix(H)
        c_degrees = np.diff(H_csr.indptr).astype(np.int32)
        self._c_degrees_d = cp.asarray(c_degrees)

        flat, offsets = _build_layer_touched_vars(H, self.K)
        self._touched_flat_d = cp.asarray(flat)
        self._touched_offsets = offsets

        self._check_order_d = cp.arange(self.m, dtype=cp.int32)

        # Modules.
        self._mod = cp.RawModule(code=_NONTEMPLATED_KERNELS_SRC)
        self._resid = self._mod.get_function("residual_count")
        self._weight = self._mod.get_function("compute_weight")
        self._sel_perb_init = self._mod.get_function("select_perb_init")
        self._sel_perb_step = self._mod.get_function("select_perb_step")
        self._sel_var_init = self._mod.get_function("select_var_init")
        self._sel_var_step = self._mod.get_function("select_var_step")

        src = _TEMPLATED_KERNELS_SRC.format(
            dc_pad=int(self.dc_pad), dv_pad=int(self.dv_pad), bwarp=int(_BWARP),
            n_warps=int(self.n_warps), lb_min_blocks=int(self.lb_min_blocks),
        )
        self._tmod = cp.RawModule(code=src)
        self._lcompute = self._tmod.get_function("layered_check_compute_inplace")
        self._lresync_sub = self._tmod.get_function("layered_var_resync_subset")
        self._mem_init = self._tmod.get_function("memory_iter_init")

        smem_check = self._smem_check()
        if smem_check > 48 * 1024:
            cap = int(cp.cuda.Device().attributes.get(
                "MaxSharedMemoryPerBlockOptin", 0))
            if cap and smem_check > cap:
                raise RuntimeError(
                    f"check kernel needs {smem_check} B of dynamic shmem but "
                    f"device opt-in cap is {cap} B.")
            self._lcompute.max_dynamic_shared_size_bytes = smem_check

        # Per-slot streams + graph caches.
        self._slot_streams = [
            cp.cuda.Stream(non_blocking=True) for _ in range(self.num_slots)
        ]
        self._slot_graph_caches: list[dict[tuple, _GraphBundle]] = [
            {} for _ in range(self.num_slots)
        ]
        self._channel_llr_d = cp.asarray(self.channel_llr)  # also the weight vector

        if c_degrees.size:
            print(
                f"[S2] dc: min={int(c_degrees.min())} avg={float(c_degrees.mean()):.1f} "
                f"max={int(c_degrees.max())} DC_PAD={int(self.dc_pad)} "
                f"N_WARPS={self.n_warps} lb={self.lb_min_blocks} "
                f"num_sets={phase.num_sets} Tr={phase.leg_max_iter} T0={phase.gamma0_iters} "
                f"gamma0={phase.gamma0} gctr={phase.gamma_center} gwid={phase.gamma_width} "
                f"alpha={phase.alpha} const={phase.alpha_const} "
                f"quorum={phase.quorum} num_slots={self.num_slots}"
            )

        # α schedules: leg 0 (T₀) and disordered legs (Tᵣ).
        self._alpha0 = _alpha_schedule(
            phase.gamma0_iters, phase.alpha, phase.tau, phase.alpha_const)
        self._alphaN = _alpha_schedule(
            phase.leg_max_iter, phase.alpha, phase.tau, phase.alpha_const)

        # Disordered gamma range. _run_chunk_legs samples one schedule per
        # batch, keyed by (prior_idx, chunk_start, leg). Runs are repeatable
        # with the same input order and batch boundaries; changing either can
        # change the schedule for a shot.
        self._gamma_lo = float(phase.gamma_center - phase.gamma_width / 2.0)
        self._gamma_hi = float(phase.gamma_center + phase.gamma_width / 2.0)

        if prebuild_bundles:
            import time as _time
            t_pb = _time.perf_counter()
            n_built = 0
            for slot in range(self.num_slots):
                with self._slot_streams[slot]:
                    for B in self.rebatch_buckets:
                        self._get_bundle(slot, int(B))
                        n_built += 1
            print(f"[S2] prebuilt {n_built} bundles in "
                  f"{_time.perf_counter() - t_pb:.1f}s")

    # ---- shared-memory sizes -------------------------------------------------

    def _smem_check(self) -> int:
        return (2 * self.dc_pad + 4 * self.n_warps * _BWARP) * 4

    def _smem_resync(self) -> int:
        return self.dv_pad * 4

    # ---- BP graph building ---------------------------------------------------

    def _build_bundle(self, B: int) -> _GraphBundle:
        """Allocate persistent buffers + capture leg-0 and leg-N graphs."""
        m, n, E = self.m, self.n, self.E

        synd_d = cp.zeros((m, B), dtype=cp.uint8)
        prior_in_d = cp.zeros((n, B), dtype=cp.float32)
        prior_d = cp.zeros((n, B), dtype=cp.float32)
        c2v_d = cp.zeros((E + 1, B), dtype=cp.float32)
        posterior_d = cp.zeros((n, B), dtype=cp.float32)
        m_prev_d = cp.zeros((n, B), dtype=cp.float32)
        residual_d = cp.zeros((B,), dtype=cp.int32)
        unsat_d = cp.zeros((m, B), dtype=cp.uint8)
        weight_d = cp.zeros((B,), dtype=cp.float32)
        best_post_d = cp.zeros((n, B), dtype=cp.float32)
        best_res_d = cp.zeros((B,), dtype=cp.int32)
        best_weight_d = cp.zeros((B,), dtype=cp.float32)
        n_conv_d = cp.zeros((B,), dtype=cp.int32)
        corr_d = cp.zeros((n, B), dtype=cp.uint8)
        found_d = cp.zeros((B,), dtype=cp.uint8)
        current_post_d = cp.zeros((n, B), dtype=cp.float32)
        gamma_d = cp.zeros((n,), dtype=cp.float32)

        gy = (B + _BWARP - 1) // _BWARP
        check_block = (_BWARP, self.n_warps)
        resync_block = (_BWARP,)
        smem_check = self._smem_check()
        smem_resync = self._smem_resync()
        layer_ranges = _layer_ranges(m, self.K)
        offsets = self._touched_offsets
        # The freeze threshold equals the quorum: a shot keeps relaying until
        # it has collected `quorum` converged legs (then n_conv >= quorum freezes
        # it). The kernels' parameter is still named stop_nconv internally.
        quorum_nconv = np.int32(self.phase.quorum)

        pb_block = (64,)
        pb_grid = ((B + 63) // 64,)
        w_block = (256,)
        w_smem = 256 * 4

        def _capture_bp_iters(alpha_np):
            """Per-iteration memory BP. M_prev must be seeded before this call;
            c2v is reset here (messages do not warm-start, only marginals)."""
            c2v_d.fill(0)
            n_iters = int(alpha_np.shape[0])
            for it in range(n_iters):
                a = float(alpha_np[it])
                # Λ(t) = (1-γ)Λ₀ + γ M(t-1); posterior = Λ(t) + Σc2v.
                self._mem_init(
                    (n, gy), (_BWARP,),
                    (
                        prior_in_d, m_prev_d, gamma_d, c2v_d, self._v2e_d,
                        np.int32(B), np.int32(n), np.int32(E),
                        prior_d, posterior_d,
                    ),
                )
                for k_layer, (ls, le) in enumerate(layer_ranges):
                    self._lcompute(
                        (le - ls, gy), check_block,
                        (
                            self._c2e_d, self._c_neigh_padded_d,
                            self._check_order_d, self._c_degrees_d,
                            synd_d, np.float32(a),
                            np.int32(B), np.int32(m), np.int32(n), np.int32(E),
                            np.int32(ls),
                            posterior_d, c2v_d,
                        ),
                        shared_mem=smem_check,
                    )
                    off0 = int(offsets[k_layer])
                    off1 = int(offsets[k_layer + 1])
                    n_layer = off1 - off0
                    if n_layer == 0:
                        continue
                    subset_view = self._touched_flat_d[off0:off1]
                    self._lresync_sub(
                        (n_layer, gy), resync_block,
                        (
                            c2v_d, self._v2e_d, prior_d, subset_view,
                            np.int32(B), np.int32(n_layer), np.int32(E),
                            np.int32(self.dv_pad),
                            posterior_d,
                        ),
                        shared_mem=smem_resync,
                    )
                # M(t) <- posterior, for next iteration's memory term.
                m_prev_d[...] = posterior_d

        def _capture_residual_weight():
            residual_d.fill(0)
            self._resid(
                (m, gy), (_BWARP,),
                (
                    posterior_d, self._c_neighbors_d, self._c_valid_d, synd_d,
                    np.int32(B), np.int32(m), np.int32(n), np.int32(self.dc_real),
                    residual_d, unsat_d,
                ),
            )
            weight_d.fill(0)
            self._weight(
                (B,), w_block,
                (
                    posterior_d, self._channel_llr_d,
                    np.int32(n), np.int32(B), weight_d,
                ),
                shared_mem=w_smem,
            )

        # ---- leg-0 graph: M_prev = Λ₀ (cold), ordered gamma0, init state ----
        cap0 = cp.cuda.Stream(non_blocking=True)
        with cap0:
            cap0.begin_capture()
            m_prev_d[...] = prior_in_d          # M(0) = channel prior
            _capture_bp_iters(self._alpha0)
            _capture_residual_weight()
            self._sel_var_init(
                (n, gy), (_BWARP,),
                (posterior_d, np.int32(n), np.int32(B),
                 best_post_d, current_post_d, corr_d),
            )
            self._sel_perb_init(
                pb_grid, pb_block,
                (residual_d, weight_d, np.int32(B), quorum_nconv,
                 best_res_d, best_weight_d, n_conv_d, found_d),
            )
            graph0 = cap0.end_capture()

        # ---- leg-N graph: M_prev = current_post (warm-start), select step ----
        capN = cp.cuda.Stream(non_blocking=True)
        with capN:
            capN.begin_capture()
            m_prev_d[...] = current_post_d       # warm-start marginals
            _capture_bp_iters(self._alphaN)
            _capture_residual_weight()
            # var_step reads PRE-step scalars; perb_step writes them after.
            self._sel_var_step(
                (n, gy), (_BWARP,),
                (
                    posterior_d, residual_d, weight_d,
                    best_res_d, best_weight_d, n_conv_d, found_d,
                    quorum_nconv, np.int32(n), np.int32(B),
                    best_post_d, current_post_d, corr_d,
                ),
            )
            self._sel_perb_step(
                pb_grid, pb_block,
                (residual_d, weight_d, np.int32(B), quorum_nconv,
                 best_res_d, best_weight_d, n_conv_d, found_d),
            )
            graphN = capN.end_capture()

        return _GraphBundle(
            leg0_graph=graph0, legN_graph=graphN,
            synd=synd_d, prior_in=prior_in_d, prior=prior_d,
            c2v=c2v_d, posterior=posterior_d, m_prev=m_prev_d,
            residual=residual_d, unsat=unsat_d, weight=weight_d,
            best_post=best_post_d, best_res=best_res_d, best_weight=best_weight_d,
            n_conv=n_conv_d, corr=corr_d, found=found_d,
            current_post=current_post_d, gamma=gamma_d, B=B,
        )

    def _get_bundle(self, slot: int, B: int) -> _GraphBundle:
        cache = self._slot_graph_caches[slot]
        bundle = cache.get(B)
        if bundle is None:
            bundle = self._build_bundle(B)
            cache[B] = bundle
        return bundle

    # ---- prior generation (CPU, thread-safe) ---------------------------------

    def _gen_prior_single(self, n_shots, scale, noise_std, seed_offset):
        n_var = self.n
        if noise_std == 0.0:
            return np.tile((self.channel_llr * scale)[:, None], (1, n_shots))
        priors = np.empty((n_var, n_shots), dtype=np.float32)
        for si in range(n_shots):
            seed = (self.BASE_SEED + seed_offset + si * 997) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, noise_std, size=n_var).astype(np.float32)
            priors[:, si] = self.channel_llr * scale * (1.0 + noise)
        return priors

    # ---- core: _run_phase with stream pipelining + rebatching ----------------

    def _run_phase(self, syndromes, phase, batch_size):
        n_shots = syndromes.shape[0]
        n_var = self.n

        out_conv = np.zeros(n_shots, dtype=bool)
        out_res = np.full(n_shots, 999999, dtype=np.int32)
        out_weight = np.full(n_shots, np.inf, dtype=np.float64)
        out_corr = np.zeros((n_shots, n_var), dtype=np.uint8)
        out_llr = np.zeros((n_shots, n_var), dtype=np.float32)
        out_variant = np.full(n_shots, -1, dtype=np.int32)
        out_nconv = np.zeros(n_shots, dtype=np.int32)
        out_obs = np.zeros((n_shots, self.k), dtype=np.uint8)

        remaining_idx = np.arange(n_shots)

        for prior_idx, (scale, noise_std) in enumerate(phase.priors):
            if len(remaining_idx) == 0:
                break
            R = len(remaining_idx)
            rem_syndromes = syndromes[remaining_idx]

            conv_R = np.zeros(R, dtype=bool)
            best_res_R = np.full(R, 999999, dtype=np.int32)
            best_w_R = np.full(R, np.inf, dtype=np.float64)
            best_corr_R = np.zeros((R, n_var), dtype=np.uint8)
            best_llr_R = np.zeros((R, n_var), dtype=np.float32)
            nconv_R = np.zeros(R, dtype=np.int32)
            obs_R = np.zeros((R, self.k), dtype=np.uint8)

            chunk_starts = list(range(0, R, batch_size))

            if self.num_slots == 1 or len(chunk_starts) == 1:
                with self._slot_streams[0]:
                    for chunk_start in chunk_starts:
                        self._run_chunk_legs(
                            0, rem_syndromes, chunk_start, batch_size,
                            prior_idx, scale, noise_std, phase,
                            conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                            nconv_R, obs_R,
                        )
            else:
                slot_chunk_starts = [
                    chunk_starts[s::self.num_slots] for s in range(self.num_slots)
                ]
                threads = []
                worker_excs: list = [None] * self.num_slots
                for slot in range(self.num_slots):
                    t = threading.Thread(
                        target=self._slot_worker,
                        args=(
                            slot, slot_chunk_starts[slot], rem_syndromes,
                            batch_size, prior_idx, scale, noise_std, phase,
                            conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                            nconv_R, obs_R, worker_excs,
                        ),
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                for exc in worker_excs:
                    if exc is not None:
                        raise exc

            out_nconv[remaining_idx] = nconv_R

            # Converged shots: keep the global min-weight correction.
            cg = remaining_idx[conv_R]
            if cg.size:
                loc = np.where(conv_R)[0]
                better = best_w_R[loc] < out_weight[cg]
                upd_loc = loc[better]
                upd_g = cg[better]
                out_conv[upd_g] = True
                out_res[upd_g] = 0
                out_weight[upd_g] = best_w_R[upd_loc]
                out_corr[upd_g] = best_corr_R[upd_loc]
                out_llr[upd_g] = best_llr_R[upd_loc]
                out_obs[upd_g] = obs_R[upd_loc]
                out_variant[upd_g] = prior_idx
                # already-converged shots stay converged
                out_conv[cg] = True

            # Never-converged shots: min-residual fallback; carry to next prior.
            nc = ~conv_R
            ncg = remaining_idx[nc]
            if ncg.size:
                loc = np.where(nc)[0]
                imp = (~out_conv[ncg]) & (best_res_R[loc] < out_res[ncg])
                upd_loc = loc[imp]
                upd_g = ncg[imp]
                out_res[upd_g] = best_res_R[upd_loc]
                out_corr[upd_g] = best_corr_R[upd_loc]
                out_llr[upd_g] = best_llr_R[upd_loc]
            remaining_idx = ncg

        return {
            "converged": out_conv, "best_llr": out_llr, "best_res": out_res,
            "best_weight": out_weight, "best_corr": out_corr,
            "variant_idx": out_variant, "n_conv": out_nconv,
            "obs_pred": out_obs,
        }

    def _slot_worker(self, slot, my_chunk_starts, rem_syndromes, batch_size,
                     prior_idx, scale, noise_std, phase,
                     conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                     nconv_R, obs_R, worker_excs):
        try:
            with self._slot_streams[slot]:
                for chunk_start in my_chunk_starts:
                    self._run_chunk_legs(
                        slot, rem_syndromes, chunk_start, batch_size,
                        prior_idx, scale, noise_std, phase,
                        conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                        nconv_R, obs_R,
                    )
        except BaseException as e:  # noqa: BLE001
            worker_excs[slot] = e

    def _sample_gamma_sched(self, prior_idx, n_runs, seed_extra):
        """Draw a disordered-γ schedule for one batch on device.

        Returns a (n_runs, n) array; row `leg` is the γ vector broadcast across
        all shots in disordered leg `leg` (row 0 unused — leg 0 uses gamma0).
        The seed mixes in `seed_extra` (the batch's chunk_start) so different
        batches get independent γ draws while remaining reproducible run-to-run.
        """
        legs = cp.empty((n_runs, self.n), dtype=cp.float32)
        for leg in range(1, n_runs):
            seed = (self.BASE_SEED + prior_idx * 100003 + leg * 7919
                    + int(seed_extra) * 2654435761) & 0xFFFFFFFF
            g = np.random.default_rng(seed).uniform(
                self._gamma_lo, self._gamma_hi, size=self.n).astype(np.float32)
            legs[leg] = cp.asarray(g)
        return legs

    def _run_chunk_legs(self, slot, rem_syndromes, chunk_start, batch_size,
                        prior_idx, scale, noise_std, phase,
                        conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                        nconv_R, obs_R):
        n_var = self.n
        R = rem_syndromes.shape[0]
        chunk_end = min(chunk_start + batch_size, R)
        cs = chunk_end - chunk_start
        synd_chunk = rem_syndromes[chunk_start:chunk_end]
        synd_mB = np.ascontiguousarray(synd_chunk.T)  # (m, cs)
        seed_off = prior_idx * 10000 + chunk_start

        B = cs
        bundle = self._get_bundle(slot, B)

        bundle.synd[...] = cp.asarray(synd_mB)
        if noise_std == 0.0:
            bundle.prior_in[...] = cp.broadcast_to(
                self._channel_llr_d[:, None] * np.float32(scale), (n_var, B))
        else:
            priors_nB = self._gen_prior_single(cs, scale, noise_std, seed_off)
            bundle.prior_in[...] = cp.asarray(priors_nB)

        # ---- coset-quorum state (chunk-local, indexed by prior - chunk_start) ----
        quorum = int(phase.quorum)
        # Per shot: cosets recorded for its converged legs (capped at `quorum`).
        rec_cosets: list[list[bytes]] = [[] for _ in range(cs)]
        n_rec = np.zeros(cs, dtype=np.int32)

        def _capture_cosets(bnd, l2p):
            """Record this leg's coset for each newly converged live shot that
            has not yet reached `quorum`. Computes ob = L @ ê (mod 2) on the GPU
            and copies only the (k,) labels to the host."""
            res = cp.asnumpy(bnd.residual)                     # (Bcur,)
            conv_cols = np.nonzero(res == 0)[0]
            if conv_cols.size == 0:
                return
            gl = l2p[conv_cols] - chunk_start                  # chunk-local idx
            keep = n_rec[gl] < quorum
            conv_cols = conv_cols[keep]
            gl = gl[keep]
            if conv_cols.size == 0:
                return
            cols_d = cp.asarray(conv_cols)
            corr_gpu = (bnd.posterior[:, cols_d] < 0.0).astype(cp.float32)  # (n, nc)
            coset_gpu = (self._L_gpu @ corr_gpu).astype(cp.uint8) & 1       # (k, nc)
            coset_h = cp.asnumpy(coset_gpu).T                  # (nc, k)
            for j in range(conv_cols.size):
                gi = int(gl[j])
                rec_cosets[gi].append(coset_h[j].tobytes())
                n_rec[gi] += 1

        # Leg 0: ordered gamma0.
        bundle.gamma[...] = cp.full(n_var, np.float32(phase.gamma0))
        bundle.leg0_graph.launch(stream=self._slot_streams[slot])

        live_to_prior = np.arange(chunk_start, chunk_start + cs, dtype=np.int32)
        _capture_cosets(bundle, live_to_prior)
        # Sample gamma for this batch, keyed by chunk_start.
        gamma_sched = self._sample_gamma_sched(prior_idx, phase.n_runs, chunk_start)

        for leg in range(1, phase.n_runs):
            live_count = int((bundle.found == 0).sum().get())
            if live_count == 0:
                break

            if self.rebatch and live_count < B:
                target_B = _bucket_size(live_count, self.rebatch_buckets)
                if live_count <= target_B < B:
                    bundle, B, live_to_prior = self._rebatch(
                        slot, bundle, target_B, live_to_prior,
                        conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R,
                        nconv_R)

            # Disordered i.i.d. γ for this leg (may be negative). Drawn for this
            # batch above, keyed by (prior_idx, chunk_start, leg).
            bundle.gamma[...] = gamma_sched[leg]

            bundle.legN_graph.launch(stream=self._slot_streams[slot])
            _capture_cosets(bundle, live_to_prior)

        # Scatter remaining columns to per-prior result arrays.
        best_post_h = cp.asnumpy(bundle.best_post)
        best_res_h = cp.asnumpy(bundle.best_res)
        best_w_h = cp.asnumpy(bundle.best_weight).astype(np.float64)
        corr_h = cp.asnumpy(bundle.corr)
        nconv_h = cp.asnumpy(bundle.n_conv)
        conv_h = nconv_h >= 1

        best_llr_R[live_to_prior] = best_post_h.T
        best_res_R[live_to_prior] = best_res_h
        best_w_R[live_to_prior] = best_w_h
        best_corr_R[live_to_prior] = corr_h.T
        conv_R[live_to_prior] = conv_h
        nconv_R[live_to_prior] = nconv_h

        # ---- coset-quorum gate (authoritative; overrides conv above and the
        # `conv_R[dp]` writes made inside _rebatch). A shot is accepted iff it
        # collected `quorum` converged legs AND they all agree on the coset. The
        # returned best_corr (min-weight) already lies in that agreed coset.
        conv_quorum = np.zeros(cs, dtype=bool)
        obs_quorum = np.zeros((cs, self.k), dtype=np.uint8)
        for i in range(cs):
            cs_list = rec_cosets[i]
            if len(cs_list) >= quorum and len(set(cs_list)) == 1:
                conv_quorum[i] = True
                obs_quorum[i] = np.frombuffer(cs_list[0], dtype=np.uint8)
        conv_R[chunk_start:chunk_start + cs] = conv_quorum
        obs_R[chunk_start:chunk_start + cs] = obs_quorum

    def _rebatch(self, slot, old_bundle, target_B, live_to_prior,
                 conv_R, best_res_R, best_w_R, best_corr_R, best_llr_R, nconv_R):
        """Slice live + padded frozen columns into a smaller-B bundle."""
        B = old_bundle.B
        live_idx_d = cp.where(old_bundle.found == 0)[0]
        frozen_idx_d = cp.where(old_bundle.found != 0)[0]
        live_count = int(live_idx_d.size)
        n_pad = target_B - live_count
        pad_idx_d = frozen_idx_d[:n_pad]
        new_order_d = cp.concatenate([live_idx_d, pad_idx_d])

        dropped_idx_d = frozen_idx_d[n_pad:]
        if dropped_idx_d.size > 0:
            dropped_idx_h = cp.asnumpy(dropped_idx_d)
            dp = live_to_prior[dropped_idx_h]
            best_llr_R[dp] = cp.asnumpy(old_bundle.best_post[:, dropped_idx_d]).T
            best_res_R[dp] = cp.asnumpy(old_bundle.best_res[dropped_idx_d])
            best_w_R[dp] = cp.asnumpy(
                old_bundle.best_weight[dropped_idx_d]).astype(np.float64)
            best_corr_R[dp] = cp.asnumpy(old_bundle.corr[:, dropped_idx_d]).T
            nconv_dropped = cp.asnumpy(old_bundle.n_conv[dropped_idx_d])
            nconv_R[dp] = nconv_dropped
            # Frozen => n_conv >= quorum => provisional accept; the coset-
            # agreement gate at the end of _run_chunk_legs overwrites conv_R
            # for the full chunk range, so this write is not authoritative.
            conv_R[dp] = nconv_dropped >= 1

        new_bundle = self._get_bundle(slot, target_B)
        new_bundle.synd[...] = old_bundle.synd[:, new_order_d]
        new_bundle.prior_in[...] = old_bundle.prior_in[:, new_order_d]
        new_bundle.best_post[...] = old_bundle.best_post[:, new_order_d]
        new_bundle.best_res[...] = old_bundle.best_res[new_order_d]
        new_bundle.best_weight[...] = old_bundle.best_weight[new_order_d]
        new_bundle.n_conv[...] = old_bundle.n_conv[new_order_d]
        new_bundle.corr[...] = old_bundle.corr[:, new_order_d]
        new_bundle.current_post[...] = old_bundle.current_post[:, new_order_d]
        new_bundle.found[...] = cp.concatenate([
            cp.zeros(live_count, dtype=cp.uint8),
            cp.ones(n_pad, dtype=cp.uint8),
        ])

        new_order_h = cp.asnumpy(new_order_d)
        return new_bundle, target_B, live_to_prior[new_order_h]

    # ---- top-level entry point -----------------------------------------------

    def run(self, syndromes):
        """Decode a batch of syndromes (n_shots, m) through one relay phase.

        Returns a dict: ``converged`` (n_shots,) bool, ``correction``
        (n_shots, n) uint8, ``obs_pred`` (n_shots, k) uint8 (zero for
        non-converged shots), ``config`` (n_shots,) object labels, plus the
        per-shot relay diagnostics."""
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        n_shots = syndromes.shape[0]

        p = self._run_phase(syndromes, self.phase, self.batch_size)

        conv = p["converged"]
        config = np.array(["NC"] * n_shots, dtype=object)
        for i in range(n_shots):
            vi = int(p["variant_idx"][i])
            if conv[i] and 0 <= vi < len(self.phase.priors):
                ps = self.phase.priors[vi][0]
                config[i] = f"{self.phase.name}_p{ps}"

        return {
            "converged": conv,
            "correction": p["best_corr"],
            "posterior_llr": p["best_llr"],
            "best_residual": p["best_res"],
            "best_weight": p["best_weight"],
            "n_conv": p["n_conv"],
            "obs_pred": p["obs_pred"],
            "config": config,
        }


__all__ = ["S2RelayBP", "S2RelayPhase", "DEFAULT_S2_PHASE"]
