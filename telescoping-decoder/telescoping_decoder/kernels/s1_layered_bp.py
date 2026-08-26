"""Fixed-iteration GPU belief propagation for S1.

The decoder supports layered and flooding check schedules; the S1 wrapper
uses layered updates. A solve has no early exit or restart. Its check updates,
variable resynchronization, residual calculation, and hard decision are
captured in a CUDA graph keyed by batch size.

``layered_var_resync_subset_mw`` assigns several warps to one variable across
different batch lanes. The single-warp implementation remains available for
comparison. ``S1_PRESETS`` in :mod:`telescoping_decoder.config` contains the
shipped iteration counts and layer settings.

``converged=True`` means that ``residual_count`` verified
``H @ correction == syndrome``. This verifies the syndrome, not the logical
class of the correction.

In flooding mode, each block owns one check and each check owns a disjoint set
of message edges. The kernel can therefore overwrite its own ``c2v`` entries
without racing another check. The posterior is read-only during the check
update.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cusp


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_DC_PAD = 160
_DV_PAD = 24
_BWARP = 32
_N_WARPS = 32


# -----------------------------------------------------------------------------
# Kernel sources
# -----------------------------------------------------------------------------

_NONTEMPLATED_KERNELS_SRC = r"""
/** @brief Count parity-check violations for every shot.
 *  @param[in] posterior Hard decisions are taken from these LLR signs.
 *  @param[in] c_neighbors_real,c_valid_real,synd Check graph and targets.
 *  @param[in] B,m,n,DC_REAL Batch and matrix dimensions.
 *  @param[in,out] residual Per-shot violation count; must start at zero.
 *  @param[out] unsat Per-check, per-shot violation flags.
 */
extern "C" __global__ void residual_count(
    const float* __restrict__ posterior,
    const int*   __restrict__ c_neighbors_real,
    const int*   __restrict__ c_valid_real,
    const unsigned char* __restrict__ synd,
    const int B, const int m, const int n, const int DC_REAL,
    int* __restrict__ residual,
    unsigned char* __restrict__ unsat
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

/** @brief Mark variables adjacent to a violated parity check.
 *  @param[in] unsat,v_checks,v_valid Current violations and variable graph.
 *  @param[in] B,n,m,DV_REAL Batch and graph dimensions.
 *  @param[out] near_mask One flag for each variable and shot.
 */
extern "C" __global__ void var_near_unsat(
    const unsigned char* __restrict__ unsat,
    const int* __restrict__ v_checks,
    const int* __restrict__ v_valid,
    const int B, const int n, const int m, const int DV_REAL,
    unsigned char* __restrict__ near_mask
) {
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    const int* nbr = v_checks + (long long)v * DV_REAL;
    const int* val = v_valid + (long long)v * DV_REAL;
    int any = 0;
    for (int k = 0; k < DV_REAL; k++) {
        int c = nbr[k];
        if (val[k] && c < m && unsat[(long long)c * B + b]) { any = 1; break; }
    }
    near_mask[(long long)v * B + b] = (unsigned char)any;
}

/** @brief Form relay priors by blending channel and current beliefs.
 *  @param[in] scaled_prior,current,mask,gw,gn Beliefs and relay weights.
 *  @param[in] B,n Batch size and variable count.
 *  @param[out] new_prior Blended prior for each variable and shot.
 */
extern "C" __global__ void apply_relay_shared(
    const float* __restrict__ scaled_prior,
    const float* __restrict__ current,
    const unsigned char* __restrict__ mask,
    const float* __restrict__ gw,
    const float* __restrict__ gn,
    const int B, const int n,
    float* __restrict__ new_prior
) {
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    long long off = (long long)v * B + b;
    float gamma = mask[off] ? gw[v] : gn[v];
    float p = scaled_prior[off];
    new_prior[off] = p + (current[off] - p) * gamma;
}

/** @brief Scale channel LLRs and broadcast them across the batch.
 *  @param[in] channel_llr,scale Base LLR vector and scale factor.
 *  @param[in] B,n Batch size and variable count.
 *  @param[out] scaled_prior Variable-major prior matrix.
 */
extern "C" __global__ void scale_prior(
    const float* __restrict__ channel_llr,
    const int B, const int n, const float scale,
    float* __restrict__ scaled_prior
) {
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    scaled_prior[(long long)v * B + b] = channel_llr[v] * scale;
}

/** @brief Retain a candidate when it has a smaller residual.
 *  @param[in] posterior_new,residual_new Candidate state and score.
 *  @param[in] B,n Batch size and variable count.
 *  @param[in,out] posterior_best,residual_best Best state seen per shot.
 */
extern "C" __global__ void copy_better(
    const float* __restrict__ posterior_new,
    const int* __restrict__ residual_new,
    float* __restrict__ posterior_best,
    int* __restrict__ residual_best,
    const int B, const int n
) {
    // For each shot, if residual_new[b] < residual_best[b], copy posterior
    // and residual. Block layout: (n_blocks, gy) where each thread handles
    // one (v, b) pair. Decision is per-shot (per b), shared across v.
    int v = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    if (residual_new[b] < residual_best[b]) {
        long long off = (long long)v * B + b;
        posterior_best[off] = posterior_new[off];
        if (v == 0) residual_best[b] = residual_new[b];
    }
}

/** @brief Set every element of a float buffer to zero.
 *  @param[in] n_elems Number of elements to clear.
 *  @param[out] buf Destination buffer.
 */
extern "C" __global__ void zero_buffer_f32(
    float* __restrict__ buf,
    const long long n_elems
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) buf[i] = 0.0f;
}

/** @brief Set every element of an integer buffer to zero.
 *  @param[in] n_elems Number of elements to clear.
 *  @param[out] buf Destination buffer.
 */
extern "C" __global__ void zero_buffer_i32(
    int* __restrict__ buf,
    const long long n_elems
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) buf[i] = 0;
}

/** @brief Convert posterior LLR signs into binary correction bits.
 *  @param[in] posterior Variable-major posterior matrix.
 *  @param[in] B,n Batch size and variable count.
 *  @param[out] correction One hard-decision bit per variable and shot.
 */
extern "C" __global__ void hard_decision(
    const float* __restrict__ posterior,  // (n, B)
    const int B, const int n,
    unsigned char* __restrict__ correction  // (n, B) uint8
) {
    // correction[v, b] = (posterior[v, b] < 0) ? 1 : 0
    long long v = blockIdx.x;
    long long b = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || v >= n) return;
    long long off = v * B + b;
    correction[off] = (posterior[off] < 0.0f) ? 1 : 0;
}
"""


_TEMPLATED_KERNELS_SRC = r"""
#define DC_PAD {dc_pad}
#define DV_PAD {dv_pad}
#define BW {bwarp}
#define N_WARPS {n_warps}

/** @brief Apply one layered offset-min-sum check update.
 *  @param[in] c2e,c_neigh,check_order,c_degrees,synd Check graph and targets.
 *  @param[in] alpha,offset,B,m,n,E,layer_start Update constants and dimensions.
 *  @param[in] posterior Current variable beliefs.
 *  @param[in,out] c2v Check-to-variable messages for this layer.
 *  @note One block owns one check; x lanes cover shots and y warps split edges.
 */
extern "C" __global__ __launch_bounds__(BW * N_WARPS, 2)
void layered_check_compute_inplace(
    const int*   __restrict__ c2e,
    const int*   __restrict__ c_neigh,
    const int*   __restrict__ check_order,
    const int*   __restrict__ c_degrees,
    const unsigned char* __restrict__ synd,
    const float alpha,
    const float offset,
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
    int s_idx = synd[c * B + b];

    float min1 = 1e30f, min2 = 1e30f;
    int argmin1 = -1;
    unsigned my_signs = 0u;
    int slice_idx = 0;
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        int v_idx = sh_nbr[k];
        float v = posterior[v_idx * B + b] - c2v[e * B + b];
        my_signs |= ((unsigned)(v < 0.0f) << slice_idx);
        float av = fabsf(v);
        if (av < min1) {{ min2 = min1; min1 = av; argmin1 = k; }}
        else if (av < min2) {{ min2 = av; }}
        slice_idx++;
    }}
    int count_neg = __popc(my_signs);

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
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        float min_excl = (k == argmin1) ? min2 : min1;
        // Offset min-sum: subtract a positive offset to approximate
        // sum-product. With offset=0 this is pure min-sum. Typical
        // offset values are 0.5 - 1.5 depending on check degree.
        min_excl = fmaxf(min_excl - offset, 0.0f);
        float sign_k = ((my_signs >> slice_idx2) & 1u) ? -1.0f : 1.0f;
        float new_msg = pre * sign_k * min_excl;
        c2v[e * B + b] = new_msg;
        slice_idx2++;
    }}
}}

/** @brief Apply one exact sum-product check update in log-tanh form.
 *  @param[in] c2e,c_neigh,check_order,c_degrees,synd Check graph and targets.
 *  @param[in] alpha,B,m,n,E,layer_start Update constants and dimensions.
 *  @param[in] posterior Current variable beliefs.
 *  @param[in,out] c2v Check-to-variable messages for this layer.
 *  @note The offset argument is accepted for a shared launch signature but unused.
 */
extern "C" __global__ __launch_bounds__(BW * N_WARPS, 2)
void sum_product_check_compute_inplace(
    const int*   __restrict__ c2e,
    const int*   __restrict__ c_neigh,
    const int*   __restrict__ check_order,
    const int*   __restrict__ c_degrees,
    const unsigned char* __restrict__ synd,
    const float alpha,
    const float offset_unused,
    const int B, const int m, const int n, const int E,
    const int layer_start,
    const float* __restrict__ posterior,
    float* __restrict__ c2v
) {{
    // Sum-product BP check update in log-tanh domain.
    //
    //   For each check c with neighbors V and syndrome s_c, the c->v_i
    //   message is:
    //       L_{{c->v_i}} = (-1)^s_c * prod_{{j!=i}} sign(L_{{v_j->c}}) *
    //                      g( sum_{{j!=i}} g(|L_{{v_j->c}}|) )
    //   where g(x) = -log(tanh(x/2)) is involutive on x > 0. Equivalent
    //   to L = 2 * artanh( prod tanh(|L|/2) ) but numerically safer.
    //
    //   Two-pass implementation: pass 1 accumulates total_sum_g and
    //   total_count_neg across edges; pass 2 recomputes per-edge g and
    //   uses the exclusion formula sum_excl = total_sum_g - g_e.
    //   Recomputing g (1 tanhf + 1 logf per edge per pass) is cheaper
    //   on this workload than caching to register array (which forces
    //   launch_bounds(*,1) and halves SM occupancy).
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
    float* part_sum_g     = (float*)(sh_nbr + DC_PAD);
    int*   part_count_neg = (int*)(part_sum_g + N_WARPS * BW);

    int total_threads = BW * N_WARPS;
    int linear_tid = wid * BW + tx;
    for (int i = linear_tid; i < dc_c; i += total_threads) {{
        sh_c2e[i] = c2e[c * DC_PAD + i];
        sh_nbr[i] = c_neigh[c * DC_PAD + i];
    }}
    __syncthreads();

    if (b >= B) return;
    int s_idx = synd[c * B + b];

    // Pass 1: accumulate sum_g and sign bits over our slice.
    float my_sum_g = 0.0f;
    unsigned my_signs = 0u;
    int slice_idx = 0;
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        int v_idx = sh_nbr[k];
        float v = posterior[v_idx * B + b] - c2v[e * B + b];
        my_signs |= ((unsigned)(v < 0.0f) << slice_idx);
        float av = fabsf(v);
        if (av < 1e-12f) av = 1e-12f;
        float t = tanhf(av * 0.5f);
        if (t > 0.99999988f) t = 0.99999988f;
        float g = -__logf(t);
        my_sum_g += g;
        slice_idx++;
    }}
    int my_count_neg = __popc(my_signs);

    part_sum_g    [wid * BW + tx] = my_sum_g;
    part_count_neg[wid * BW + tx] = my_count_neg;
    __syncthreads();

    // Cross-warp reduction (warp 0 reduces over all warps).
    if (wid == 0) {{
        #pragma unroll
        for (int w = 1; w < N_WARPS; w++) {{
            my_sum_g     += part_sum_g    [w * BW + tx];
            my_count_neg += part_count_neg[w * BW + tx];
        }}
        part_sum_g    [tx] = my_sum_g;
        part_count_neg[tx] = my_count_neg;
    }}
    __syncthreads();

    float total_sum_g = part_sum_g[tx];
    int   total_count_neg = part_count_neg[tx];

    int parity = (total_count_neg + s_idx) & 1;
    float pre = (parity ? -alpha : alpha);

    // Pass 2: write outputs using exclusion (recompute g).
    int slice_idx2 = 0;
    for (int k = k_start; k < k_end; k++) {{
        int e = sh_c2e[k];
        int v_idx = sh_nbr[k];
        float v = posterior[v_idx * B + b] - c2v[e * B + b];
        float av = fabsf(v);
        if (av < 1e-12f) av = 1e-12f;
        float t = tanhf(av * 0.5f);
        if (t > 0.99999988f) t = 0.99999988f;
        float g = -__logf(t);
        float s_excl = total_sum_g - g;
        if (s_excl < 1e-12f) s_excl = 1e-12f;
        float t2 = tanhf(s_excl * 0.5f);
        if (t2 > 0.99999988f) t2 = 0.99999988f;
        float mag_out = -__logf(t2);
        float sign_k = ((my_signs >> slice_idx2) & 1u) ? -1.0f : 1.0f;
        float new_msg = pre * sign_k * mag_out;
        c2v[e * B + b] = new_msg;
        slice_idx2++;
    }}
}}

/** @brief Recompute touched-variable posteriors after a layer update.
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
        sum += c2v[sh_v2e[k] * B + b];
    }}
    posterior[v * B + b] = prior[v * B + b] + sum;
}}

// Multi-warp variable resynchronization.
//
// Same math as the baseline kernel, but block shape is (BW, var_nwy):
// (32, 4) by default. Grid.y = ceil(B / (BW * var_nwy)).
//
// Each block processes one variable across BW*NWY batch slots. Multiple
// warps provide independent c2v reads for the scheduler to interleave.
//
// `sh_v2e` is shared across warps in the same block (single shared
// allocation), loaded once per block.
//
// Coalescing: each warp reads c2v[e * B + b_warp..b_warp+31] which
// is 32 consecutive fp32 values and is therefore coalesced.
/** @brief Recompute touched-variable posteriors with multiple batch warps.
 *  @param[in] c2v,v2e,prior,subset_idx Messages, graph, priors, and variables.
 *  @param[in] B,n_layer,E,DV_unused Batch and graph dimensions.
 *  @param[out] posterior Prior plus all incident check messages.
 *  @note Each y-warp handles a different contiguous group of batch lanes.
 */
extern "C" __global__ void layered_var_resync_subset_mw(
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
    int wy = threadIdx.y;
    int nwy = blockDim.y;
    int b = blockIdx.y * (BW * nwy) + wy * BW + tx;

    extern __shared__ int sh_v2e[];
    int linear_tid = wy * BW + tx;
    int total_threads = BW * nwy;
    #pragma unroll
    for (int i = linear_tid; i < DV_PAD; i += total_threads) {{
        sh_v2e[i] = v2e[v * DV_PAD + i];
    }}
    __syncthreads();

    if (b >= B) return;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < DV_PAD; k++) {{
        sum += c2v[sh_v2e[k] * B + b];
    }}
    posterior[v * B + b] = prior[v * B + b] + sum;
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


def _build_layer_touched_vars(H, K: int):
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


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

@dataclass
class _S1GraphBundle:
    graph: object
    synd: cp.ndarray
    channel_llr: cp.ndarray  # (n,) — pristine, used as base for prior scaling
    prior: cp.ndarray         # (n, B) — current prior; mutated between legs
    c2v: cp.ndarray
    posterior: cp.ndarray
    residual: cp.ndarray
    unsat: cp.ndarray
    near_mask: cp.ndarray
    # Per-leg buffers (only used if n_legs > 1)
    gw_per_leg: list  # list of (n,) device arrays, one per leg-1 transitions
    gn_per_leg: list
    # Best-so-far tracking across legs
    posterior_best: cp.ndarray
    residual_best: cp.ndarray
    correction: cp.ndarray   # (n, B) uint8 — computed in-graph from posterior
    B: int


class S1LayeredBP:
    """The S1 stage's BP decoder.

    Fixed n_iters BP iterations, no early exit, one CUDA graph per batch
    size B. Supports layered schedules.

    converged[i] is True iff ``(H @ correction[i]) % 2 == syndrome[i]``.
    When ``obs_matrix`` is supplied, :meth:`run` also returns its product
    with the device correction. Logical-error counting remains the caller's
    responsibility because the true observable is evaluation data.
    """

    BASE_SEED = 12345

    def __init__(
        self,
        H,
        channel_llr,
        *,
        schedule: str = "flooding",
        n_iters: int = 8,
        alpha: float | np.ndarray = 1.0,
        alpha_schedule: str | None = None,
        alpha_max: float = 0.9,
        alpha_tau: float = 5.0,
        K: int = 64,
        n_legs: int = 1,
        prior_scales: tuple = (1.0, 0.8, 1.2),
        gw_max: float = 6.0,
        gn_lo: float = 0.5,
        gn_hi: float = 1.5,
        algorithm: str = "min_sum",
        hybrid_sp_iters: int = 6,
        offset: float = 0.0,
        fast_var_resync: bool = True,
        var_nwy: int = 4,
        dc_pad: int = _DC_PAD,
        dv_pad: int = _DV_PAD,
        obs_matrix=None,
        acceptance_matrix=None,
        acceptance_syndrome_cols=None,
        return_posterior: bool = False,
        base_seed: int = 12345,
        verbose: bool = True,
    ):
        """Build the decoder.

        ``alpha`` may be a scalar (constant per-iter damping) or a 1D
        array of length ``n_iters`` (per-iter schedule). Alternatively,
        pass ``alpha_schedule="ramp"`` to use the ramped schedule
        ``alpha_max * (1 - exp(-(t+1)/tau))`` with the given
        ``alpha_max`` and ``alpha_tau``.

        ``n_legs > 1`` enables a relay-BP cascade. Between legs, the
        prior is updated via the ``apply_relay_shared`` kernel using
        per-call random gw/gn vectors. Each leg uses ``prior_scales``
        cyclically to bias the prior. The best (lowest-residual)
        posterior across all legs is reported.

        ``obs_matrix`` optionally enables device-side observable extraction
        from the final correction. ``acceptance_matrix`` and
        ``acceptance_syndrome_cols`` optionally define a second parity check
        used by wrappers such as GARI S1, whose routing rule covers only a
        subset of the decoded rows.
        """
        if schedule not in ("flooding", "layered"):
            raise ValueError(
                f"schedule must be 'flooding' or 'layered'; got {schedule!r}"
            )
        if algorithm not in ("min_sum", "sum_product", "hybrid"):
            raise ValueError(
                f"algorithm must be 'min_sum', 'sum_product', or "
                f"'hybrid'; got {algorithm!r}"
            )
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1; got {n_iters}")

        self.schedule = schedule
        self.n_iters = int(n_iters)

        self._obs_gpu = None
        self.n_obs = 0
        if obs_matrix is not None:
            obs_csr = sp.csr_matrix(obs_matrix)
            if obs_csr.shape[1] != H.shape[1]:
                raise ValueError(
                    "obs_matrix must have the same number of columns as H")
            self._obs_gpu = cusp.csr_matrix(obs_csr.astype(np.float32))
            self.n_obs = int(obs_csr.shape[0])

        self._acceptance_gpu = None
        self._acceptance_syndrome_cols_gpu = None
        if acceptance_matrix is not None:
            acc_csr = sp.csr_matrix(acceptance_matrix)
            if acc_csr.shape[1] != H.shape[1]:
                raise ValueError(
                    "acceptance_matrix must have the same number of columns as H")
            cols = np.asarray(acceptance_syndrome_cols, dtype=np.int64)
            if cols.shape != (acc_csr.shape[0],):
                raise ValueError(
                    "acceptance_syndrome_cols must provide one decoded "
                    "syndrome row per acceptance-matrix row")
            if cols.size and (int(cols.min()) < 0 or int(cols.max()) >= H.shape[0]):
                raise ValueError("acceptance_syndrome_cols contains an invalid row")
            self._acceptance_gpu = cusp.csr_matrix(
                acc_csr.astype(np.float32))
            self._acceptance_syndrome_cols_gpu = cp.asarray(cols)
        elif acceptance_syndrome_cols is not None:
            raise ValueError(
                "acceptance_syndrome_cols requires acceptance_matrix")
        if alpha_schedule == "ramp":
            self.alpha = np.array(
                [alpha_max * (1.0 - np.exp(-(t + 1) / alpha_tau))
                 for t in range(self.n_iters)],
                dtype=np.float32,
            )
        elif alpha_schedule is None:
            arr = np.asarray(alpha, dtype=np.float32)
            if arr.ndim == 0:
                self.alpha = np.full(self.n_iters, float(arr), dtype=np.float32)
            elif arr.shape == (self.n_iters,):
                self.alpha = arr.astype(np.float32, copy=True)
            else:
                raise ValueError(
                    f"alpha must be scalar or shape ({self.n_iters},); "
                    f"got shape {arr.shape}"
                )
        else:
            raise ValueError(
                f"alpha_schedule must be None or 'ramp'; got {alpha_schedule!r}"
            )
        self.K = int(K)
        self.n_legs = max(1, int(n_legs))
        self.prior_scales = tuple(float(s) for s in prior_scales)
        if not self.prior_scales:
            self.prior_scales = (1.0,)
        self.gw_max = float(gw_max)
        self.gn_lo = float(gn_lo)
        self.gn_hi = float(gn_hi)
        self.algorithm = algorithm
        self.hybrid_sp_iters = int(hybrid_sp_iters)
        self.offset = float(offset)
        self.fast_var_resync = bool(fast_var_resync)
        self.var_nwy = int(var_nwy)
        self.return_posterior = bool(return_posterior)

        arr = _build_arrays(H, dc_pad, dv_pad)
        self.m = arr["m"]
        self.n = arr["n"]
        self.E = arr["nnz"]
        self.dc_pad = arr["dc_pad"]
        self.dv_pad = arr["dv_pad"]
        self.dc_real = arr["dc_real"]
        self.dv_real = arr["dv_real"]
        self.channel_llr = np.ascontiguousarray(channel_llr, dtype=np.float32)
        self.BASE_SEED = base_seed

        # Read-only device matrix arrays.
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

        if schedule == "layered":
            flat, offsets = _build_layer_touched_vars(H, self.K)
            self._touched_flat_d = cp.asarray(flat)
            self._touched_offsets = offsets
            self._all_vars_d = None
        else:
            self._touched_flat_d = None
            self._touched_offsets = None
            self._all_vars_d = cp.arange(self.n, dtype=cp.int32)

        self._check_order_d = cp.arange(self.m, dtype=cp.int32)

        # Modules.
        self._mod = cp.RawModule(code=_NONTEMPLATED_KERNELS_SRC)
        self._resid = self._mod.get_function("residual_count")
        self._near = self._mod.get_function("var_near_unsat")
        self._relay = self._mod.get_function("apply_relay_shared")
        self._scale_prior = self._mod.get_function("scale_prior")
        self._copy_better = self._mod.get_function("copy_better")
        self._zero_f32 = self._mod.get_function("zero_buffer_f32")
        self._zero_i32 = self._mod.get_function("zero_buffer_i32")
        self._hard_decision = self._mod.get_function("hard_decision")

        src = _TEMPLATED_KERNELS_SRC.format(
            dc_pad=int(self.dc_pad),
            dv_pad=int(self.dv_pad),
            bwarp=int(_BWARP),
            n_warps=int(_N_WARPS),
        )
        self._tmod = cp.RawModule(code=src)
        self._lcompute_ms = self._tmod.get_function(
            "layered_check_compute_inplace"
        )
        self._lcompute_sp = self._tmod.get_function(
            "sum_product_check_compute_inplace"
        )
        # For homogeneous algorithm runs, _lcompute is the active kernel.
        # For algorithm="hybrid", _emit_one_bp_solve picks per-iter.
        if algorithm == "sum_product":
            self._lcompute = self._lcompute_sp
        elif algorithm == "min_sum":
            self._lcompute = self._lcompute_ms
        else:  # hybrid
            self._lcompute = self._lcompute_sp  # fallback only
        self._lresync_sub_baseline = self._tmod.get_function(
            "layered_var_resync_subset"
        )
        self._lresync_sub_mw = self._tmod.get_function(
            "layered_var_resync_subset_mw"
        )
        # Default to the multi-warp version unless explicitly disabled.
        self._lresync_sub = (
            self._lresync_sub_mw if self.fast_var_resync
            else self._lresync_sub_baseline
        )

        # Compute shmem opt-in for both kernels (hybrid uses both).
        smem_ms = self._smem_check_for("min_sum")
        smem_sp = self._smem_check_for("sum_product")
        smem_max = max(smem_ms, smem_sp)
        if smem_max > 48 * 1024:
            cap = int(cp.cuda.Device().attributes.get(
                "MaxSharedMemoryPerBlockOptin", 0))
            if cap and smem_max > cap:
                raise RuntimeError(
                    f"check kernel needs {smem_max} B of dynamic shmem but "
                    f"device opt-in cap is {cap} B "
                    f"(dc_pad={self.dc_pad}, BWARP={_BWARP}, "
                    f"N_WARPS={_N_WARPS})."
                )
            if smem_ms > 48 * 1024:
                self._lcompute_ms.max_dynamic_shared_size_bytes = smem_ms
            if smem_sp > 48 * 1024:
                self._lcompute_sp.max_dynamic_shared_size_bytes = smem_sp

        self._graph_cache: dict[int, _S1GraphBundle] = {}

        if verbose:
            if self.alpha.std() < 1e-6:
                alpha_str = f"alpha={float(self.alpha[0]):.3f}"
            else:
                alpha_str = (
                    f"alpha=[{self.alpha[0]:.3f}..{self.alpha[-1]:.3f}]"
                )
            print(
                f"[S1] algorithm={self.algorithm} schedule={schedule} "
                f"n_iters={self.n_iters} {alpha_str} K={self.K} "
                f"n_legs={self.n_legs} prior_scales={self.prior_scales}"
            )
            print(
                f"[S1] H={self.m}x{self.n} nnz={self.E} "
                f"dc_real={self.dc_real} dv_real={self.dv_real} "
                f"dc_pad={self.dc_pad} dv_pad={self.dv_pad}"
            )

    def _smem_check(self) -> int:
        # Conservative max across both kernels (hybrid uses both).
        return max(
            self._smem_check_for("min_sum"),
            self._smem_check_for("sum_product"),
        )

    def _smem_check_for(self, algo: str) -> int:
        # Min-sum needs 4 N_WARPS*BW partials: min1, min2, argmin1, count.
        # Sum-product needs only 2: sum_g, count_neg.
        n_partials = 2 if algo == "sum_product" else 4
        return (2 * self.dc_pad + n_partials * _N_WARPS * _BWARP) * 4

    def _smem_resync(self) -> int:
        return self.dv_pad * 4

    # ---- graph capture --------------------------------------------------

    def _emit_one_bp_solve(self, bundle, B: int) -> None:
        """Capture one BP solve (n_iters of check+var). Reads from
        bundle.prior + bundle.posterior + bundle.c2v, mutates them.

        Var-resync dispatch differs by `fast_var_resync`:
        * baseline: block=(BW,), grid=(n_layer, ceil(B/BW)).
        * multi-warp: block=(BW, var_nwy), grid=(n_layer, ceil(B/(BW*var_nwy))).
          `var_nwy` is the number of y-dimension warps; its default is 4.
        """
        m, n, E = self.m, self.n, self.E
        gy = (B + _BWARP - 1) // _BWARP
        check_block = (_BWARP, _N_WARPS)
        if self.fast_var_resync:
            nwy = self.var_nwy
            resync_block = (_BWARP, nwy)
            resync_gy = (B + _BWARP * nwy - 1) // (_BWARP * nwy)
        else:
            resync_block = (_BWARP,)
            resync_gy = gy
        smem_resync = self._smem_resync()
        for it in range(self.n_iters):
            a = float(self.alpha[it])
            # Per-iter kernel pick (for hybrid; constant for others).
            if self.algorithm == "hybrid":
                if it < self.hybrid_sp_iters:
                    cur_kernel = self._lcompute_sp
                    cur_smem = self._smem_check_for("sum_product")
                else:
                    cur_kernel = self._lcompute_ms
                    cur_smem = self._smem_check_for("min_sum")
            else:
                cur_kernel = self._lcompute
                cur_smem = self._smem_check_for(self.algorithm)
            smem_check = cur_smem
            if self.schedule == "flooding":
                cur_kernel(
                    (m, gy), check_block,
                    (
                        self._c2e_d, self._c_neigh_padded_d,
                        self._check_order_d, self._c_degrees_d,
                        bundle.synd, np.float32(a),
                        np.float32(self.offset),
                        np.int32(B), np.int32(m), np.int32(n),
                        np.int32(E), np.int32(0),
                        bundle.posterior, bundle.c2v,
                    ),
                    shared_mem=smem_check,
                )
                self._lresync_sub(
                    (n, resync_gy), resync_block,
                    (
                        bundle.c2v, self._v2e_d, bundle.prior,
                        self._all_vars_d,
                        np.int32(B), np.int32(n), np.int32(E),
                        np.int32(self.dv_pad),
                        bundle.posterior,
                    ),
                    shared_mem=smem_resync,
                )
            else:  # layered
                layer_ranges = _layer_ranges(m, self.K)
                offsets = self._touched_offsets
                for k_layer, (ls, le) in enumerate(layer_ranges):
                    cur_kernel(
                        (le - ls, gy), check_block,
                        (
                            self._c2e_d, self._c_neigh_padded_d,
                            self._check_order_d, self._c_degrees_d,
                            bundle.synd, np.float32(a),
                            np.float32(self.offset),
                            np.int32(B), np.int32(m), np.int32(n),
                            np.int32(E), np.int32(ls),
                            bundle.posterior, bundle.c2v,
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
                        (n_layer, resync_gy), resync_block,
                        (
                            bundle.c2v, self._v2e_d, bundle.prior,
                            subset_view,
                            np.int32(B), np.int32(n_layer),
                            np.int32(E), np.int32(self.dv_pad),
                            bundle.posterior,
                        ),
                        shared_mem=smem_resync,
                    )

    def _build_graph(self, B: int) -> _S1GraphBundle:
        m = self.m
        n = self.n
        E = self.E
        synd_d = cp.zeros((m, B), dtype=cp.uint8)
        prior_d = cp.zeros((n, B), dtype=cp.float32)
        channel_d = cp.asarray(self.channel_llr)
        c2v_d = cp.zeros((E + 1, B), dtype=cp.float32)
        posterior_d = cp.zeros((n, B), dtype=cp.float32)
        residual_d = cp.zeros((B,), dtype=cp.int32)
        unsat_d = cp.zeros((m, B), dtype=cp.uint8)
        near_d = cp.zeros((n, B), dtype=cp.uint8)
        posterior_best_d = cp.zeros((n, B), dtype=cp.float32)
        residual_best_d = cp.zeros((B,), dtype=cp.int32)
        correction_d = cp.zeros((n, B), dtype=cp.uint8)

        # Per-leg gw/gn buffers (one per leg, even though leg 0 doesn't use
        # them — keeps indexing simple).
        gw_per_leg = [cp.zeros((n,), dtype=cp.float32)
                      for _ in range(self.n_legs)]
        gn_per_leg = [cp.zeros((n,), dtype=cp.float32)
                      for _ in range(self.n_legs)]

        bundle = _S1GraphBundle(
            graph=None, synd=synd_d, channel_llr=channel_d, prior=prior_d,
            c2v=c2v_d, posterior=posterior_d, residual=residual_d,
            unsat=unsat_d, near_mask=near_d,
            gw_per_leg=gw_per_leg, gn_per_leg=gn_per_leg,
            posterior_best=posterior_best_d,
            residual_best=residual_best_d,
            correction=correction_d,
            B=B,
        )

        gy = (B + _BWARP - 1) // _BWARP
        threads_per_block = 256
        c2v_blocks = (E + 1) * B
        c2v_grid = (c2v_blocks + threads_per_block - 1) // threads_per_block
        capture_stream = cp.cuda.Stream(non_blocking=True)
        with capture_stream:
            capture_stream.begin_capture()
            # Initialize prior = channel_llr * scale[0]
            self._scale_prior(
                (n, gy), (_BWARP,),
                (channel_d, np.int32(B), np.int32(n),
                 np.float32(self.prior_scales[0]), prior_d),
            )
            # posterior = prior (initial state for BP)
            posterior_d[...] = prior_d
            # c2v = 0 (zero kernel within graph)
            self._zero_f32(
                (c2v_grid,), (threads_per_block,),
                (c2v_d, np.int64(c2v_blocks)),
            )
            # Clear the best-residual buffer. Leg 0 overwrites it
            # unconditionally; later legs replace it only when they improve
            # the residual.
            self._zero_i32(
                ((B + 255) // 256,), (256,),
                (residual_best_d, np.int64(B)),
            )

            for leg in range(self.n_legs):
                # Run one BP solve on the current prior.
                self._emit_one_bp_solve(bundle, B)
                # Zero residual_d before each residual_count call —
                # the kernel atomicAdds into it, so it must start at 0.
                self._zero_i32(
                    ((B + 255) // 256,), (256,),
                    (residual_d, np.int64(B)),
                )
                # Compute residual + unsat from current posterior.
                self._resid(
                    (m, gy), (_BWARP,),
                    (
                        posterior_d, self._c_neighbors_d,
                        self._c_valid_d, synd_d,
                        np.int32(B), np.int32(m), np.int32(n),
                        np.int32(self.dc_real),
                        residual_d, unsat_d,
                    ),
                )
                if leg == 0:
                    # First leg: copy posterior+residual to best
                    # unconditionally.
                    posterior_best_d[...] = posterior_d
                    residual_best_d[...] = residual_d
                else:
                    # Track best across legs: copy posterior_new and
                    # residual_new where residual_new < residual_best.
                    self._copy_better(
                        (n, gy), (_BWARP,),
                        (posterior_d, residual_d,
                         posterior_best_d, residual_best_d,
                         np.int32(B), np.int32(n)),
                    )

                if leg < self.n_legs - 1:
                    # Compute near_mask from unsat.
                    self._near(
                        (n, gy), (_BWARP,),
                        (unsat_d, self._v_checks_d, self._v_valid_d,
                         np.int32(B), np.int32(n), np.int32(m),
                         np.int32(self.dv_real),
                         near_d),
                    )
                    # Compute scaled_prior = channel_llr * scale[leg+1]
                    # into a TEMPORARY: we reuse the prior buffer because
                    # apply_relay_shared writes new_prior. We need the
                    # old scaled_prior as input. Write
                    # scaled_prior to c2v's first n*B floats (which we
                    # immediately zero after, since BP starts fresh).
                    # Scale into prior_d in place,
                    # then apply relay using prior_d as scaled_prior AND
                    # the *output* (read-then-write per element is OK
                    # because each thread reads/writes the same offset
                    # and the kernel reads scaled_prior[off] before
                    # writing new_prior[off] within the same thread).
                    scale_next = float(
                        self.prior_scales[(leg + 1) % len(self.prior_scales)]
                    )
                    self._scale_prior(
                        (n, gy), (_BWARP,),
                        (channel_d, np.int32(B), np.int32(n),
                         np.float32(scale_next), prior_d),
                    )
                    # Apply relay: new_prior = prior + (post - prior) * gamma
                    # The kernel reads scaled_prior[off] and current[off]
                    # then writes new_prior[off]. Reading scaled_prior
                    # (= prior_d) and writing new_prior (= prior_d) at
                    # the same offset within one thread is safe (same
                    # address, sequenced read-then-write).
                    self._relay(
                        (n, gy), (_BWARP,),
                        (
                            prior_d, posterior_d, near_d,
                            gw_per_leg[leg], gn_per_leg[leg],
                            np.int32(B), np.int32(n),
                            prior_d,  # in-place: new_prior = prior_d
                        ),
                    )
                    # Reset BP state for next leg.
                    posterior_d[...] = prior_d
                    self._zero_f32(
                        (c2v_grid,), (threads_per_block,),
                        (c2v_d, np.int64(c2v_blocks)),
                    )

            # Final: write best residual back into `residual_d` and best
            # posterior into `posterior_d` so the run() D2H finds them.
            posterior_d[...] = posterior_best_d
            residual_d[...] = residual_best_d
            # Compute correction = (posterior < 0) on GPU. Avoids a 257
            # MB D2H of fp32 posterior + 64M-element CPU compare.
            self._hard_decision(
                (n, gy), (_BWARP,),
                (posterior_d, np.int32(B), np.int32(n), correction_d),
            )
            graph = capture_stream.end_capture()

        bundle.graph = graph
        return bundle

    def _get_or_build_graph(self, B: int) -> _S1GraphBundle:
        bundle = self._graph_cache.get(B)
        if bundle is None:
            bundle = self._build_graph(B)
            self._graph_cache[B] = bundle
        return bundle

    # ---- public API -----------------------------------------------------

    def run(self, syndromes, *, seed=None, n_valid=None):
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : (B, m) uint8
            Per-shot syndrome bits.
        seed : optional int — overrides ``base_seed`` for this call's
            gw/gn random draws.
        n_valid : optional int
            Number of leading, non-padding shots for device-side observable
            and alternate-acceptance extraction. The BP graph still runs at
            the full input batch size.

        Returns
        -------
        dict with keys: ``converged`` (B,) bool, ``correction`` (B, n)
        uint8, ``best_residual`` (B,) int32, ``config`` (B,) object,
        and ``n_iters_used`` int. When configured, ``obs_pred`` contains
        ``obs_matrix @ correction`` and ``relevant_converged`` contains the
        acceptance-matrix parity result.
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        B = syndromes.shape[0]
        if syndromes.shape[1] != self.m:
            raise ValueError(
                f"syndrome shape {syndromes.shape} != (B={B}, m={self.m})"
            )
        if n_valid is None:
            n_valid = B
        n_valid = int(n_valid)
        if not 0 <= n_valid <= B:
            raise ValueError(f"n_valid must be in [0, {B}]; got {n_valid}")

        bundle = self._get_or_build_graph(B)

        # H2D syndrome.
        synd_mB = np.ascontiguousarray(syndromes.T)
        bundle.synd[...] = cp.asarray(synd_mB)

        # Per-call random gw/gn for each leg transition. Leg 0's buffers
        # are unused (no relay before leg 0) but populated for safety.
        if self.n_legs > 1:
            seed_use = self.BASE_SEED if seed is None else int(seed)
            rng = np.random.default_rng(seed_use)
            for leg in range(self.n_legs):
                gw_h = rng.uniform(
                    0.0, self.gw_max, size=self.n
                ).astype(np.float32)
                gn_h = rng.uniform(
                    self.gn_lo, self.gn_hi, size=self.n
                ).astype(np.float32)
                bundle.gw_per_leg[leg][...] = cp.asarray(gw_h)
                bundle.gn_per_leg[leg][...] = cp.asarray(gn_h)
            self.BASE_SEED = (self.BASE_SEED * 1103515245 + 12345) & 0xFFFFFFFF

        # Graph initializes prior, posterior, c2v internally and computes
        # correction on the GPU. Copy the residual and compact uint8 correction
        # (B*n = 64 MB at B=256), avoiding a 257 MB D2H
        # of fp32 posterior + a 64M-elt CPU compare.
        bundle.graph.launch()
        cp.cuda.Stream.null.synchronize()

        residual_h = cp.asnumpy(bundle.residual)        # (B,) int32

        # The hard decision already lives on the GPU. Extract observables and
        # any alternate acceptance result here, before copying the correction
        # to the CPU, so callers never have to upload that correction again.
        obs_pred = None
        relevant_converged = None
        if self._obs_gpu is not None or self._acceptance_gpu is not None:
            correction_f32 = bundle.correction[:, :n_valid].astype(cp.float32)
            if self._obs_gpu is not None:
                obs_gpu = (self._obs_gpu @ correction_f32).astype(cp.uint8) & 1
                obs_pred = cp.asnumpy(obs_gpu).T
            if self._acceptance_gpu is not None:
                pred_gpu = (
                    self._acceptance_gpu @ correction_f32
                ).astype(cp.uint8) & 1
                target_gpu = bundle.synd[
                    self._acceptance_syndrome_cols_gpu, :n_valid]
                relevant_converged = cp.asnumpy(
                    ~cp.any(pred_gpu != target_gpu, axis=0))

        correction = cp.asnumpy(bundle.correction).T    # (n, B) → (B, n)
        converged = (residual_h == 0)

        config = np.array(
            ["S1" if c else "NC" for c in converged], dtype=object,
        )
        out = {
            "converged": converged,
            "correction": correction,
            "best_residual": residual_h,
            "config": config,
            "n_iters_used": self.n_iters,
        }
        if obs_pred is not None:
            out["obs_pred"] = obs_pred
        if relevant_converged is not None:
            out["relevant_converged"] = relevant_converged
        if self.return_posterior:
            out["posterior_llr"] = cp.asnumpy(bundle.posterior).T
        else:
            out["posterior_llr"] = np.empty((B, 0), dtype=np.float32)
        return out


__all__ = ["S1LayeredBP"]
