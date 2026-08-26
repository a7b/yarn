/*
 * Check-serial (layered) min-sum BP with relay memory for S3-B.
 *
 * Standalone CPU implementation of the relay dynamics of arXiv:2506.01779.
 * The check-serial sweep, RNG, shuffle helpers, convergence check and residual
 * count are duplicated from checkserial_bp.c (the S3-A/S3-C kernel). The
 * additional operation is the per-iteration memory term:
 *
 *     Lambda(t) = (1 - gamma) * Lambda0 + gamma * M(t-1)      [Eq. 4, arXiv:2506.01779]
 *
 * applied every iteration (gamma fixed for the leg), where Lambda0 = channel_llr
 * (immutable) and M(t-1) is the previous iteration's marginal. Marginals are
 * warm-started across legs via `m_init`; messages (c2v) are NOT warm-started
 * (reset each call), matching the GPU relay kernel in kernels/s2_relay_bp.py.
 *
 * Because the check-serial sweep maintains the invariant
 *     llr[j] == Lambda(t)[j] + sum_over_incident_edges(c2v[edge])
 * at every iteration boundary, the bias swap can be done incrementally as
 *     llr[j] += Lambda(t)[j] - Lambda(t-1)[j]
 * which needs no variable-to-edge map and is
 * equivalent to a full `posterior = bias + sum(c2v)` resync.
 *
 * Maintenance: changes to the duplicated check sweep in checkserial_bp.c may
 * need to be mirrored here.
 *
 * Compile:
 *   gcc -O3 -ffast-math -shared -fPIC -o relay_mem_bp.so relay_mem_bp.c -lm
 *
 * Exports a single symbol: relay_mem_bp_decode (gamma=NULL => plain BP).
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_DEG 2048
/* Clamp the per-iteration bias under -ffast-math. The bound is far above the
 * expected channel LLR and protects extrapolating gamma values. */
#define LAMBDA_CLIP 1e4f

/* ── xorshift32 RNG ── */
static unsigned int xor32(unsigned int *state)
{
    unsigned int s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return s;
}

/* ── Fisher-Yates shuffle ── */
static void shuffle(int *arr, int n, unsigned int *rng_state)
{
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(xor32(rng_state) % (unsigned int)(i + 1));
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/*
 * Block-shuffle: divide [0..n) into blocks of size block_sz.
 * Shuffle the block order (Fisher-Yates on block indices),
 * but within each block the elements stay sequential.
 */
static void block_shuffle(int *check_order, int n, int block_sz, unsigned int *rng_state)
{
    int n_blocks = (n + block_sz - 1) / block_sz;
    int *block_idx;
    int on_heap = (n_blocks > 8192);
    if (on_heap)
        block_idx = (int *)malloc((size_t)n_blocks * sizeof(int));
    else
        block_idx = (int *)__builtin_alloca((size_t)n_blocks * sizeof(int));

    for (int b = 0; b < n_blocks; b++) block_idx[b] = b;
    shuffle(block_idx, n_blocks, rng_state);

    int pos = 0;
    for (int bi = 0; bi < n_blocks; bi++) {
        int b = block_idx[bi];
        int start = b * block_sz;
        int end = start + block_sz;
        if (end > n) end = n;
        for (int i = start; i < end; i++) {
            check_order[pos++] = i;
        }
    }

    if (on_heap) free(block_idx);
}

void relay_mem_bp_decode(
    const int *indptr,
    const int *indices,
    int n_checks,
    int n_vars,
    int nnz,
    const uint8_t *syndrome,
    const float *channel_llr,      /* Lambda0 — immutable bias (required) */
    int max_iter,
    const float *alpha_schedule,
    float beta,
    int random_order,
    unsigned int rng_seed,
    /* caller-provided buffers */
    float *__restrict__ llr,
    float *__restrict__ c2v,
    int *check_order,
    /* relay-memory inputs + scratch (gamma==NULL => plain BP, no memory) */
    const float *gamma,            /* per-variable memory strength, n_vars */
    const float *m_init,           /* warm-start marginal, n_vars, or NULL (cold) */
    float *m_prev,                 /* caller scratch, n_vars (used iff gamma!=NULL) */
    float *lambda_prev,            /* caller scratch, n_vars (used iff gamma!=NULL) */
    /* outputs */
    uint8_t *decoding,
    int *out_iters,
    int *out_converged,
    int *out_residual
)
{
    /* Safety: stack array v2c_val below is sized MAX_DEG. A negative
     * out_iters is an explicit unsupported-degree signal to the caller. */
    int max_row_deg = 0;
    for (int i = 0; i < n_checks; i++) {
        int d = indptr[i + 1] - indptr[i];
        if (d > max_row_deg) max_row_deg = d;
    }
    if (max_row_deg > MAX_DEG) {
        *out_iters = -1;
        *out_converged = 0;
        *out_residual = n_checks;
        for (int j = 0; j < n_vars; j++) decoding[j] = 0;
        return;
    }

    if (channel_llr != NULL) {
        memcpy(llr, channel_llr, (size_t)n_vars * sizeof(float));
        memset(c2v, 0, (size_t)nnz * sizeof(float));
    }

    /* ── relay-memory init: seed M(0), set llr = Lambda(1) ──
     * Lambda(1) = (1-gamma)*Lambda0 + gamma*M(0), with M(0) = warm-start
     * marginal (or channel prior if cold). c2v is 0 here so llr == Lambda(1). */
    if (gamma != NULL) {
        for (int j = 0; j < n_vars; j++) {
            float Mp = (m_init != NULL) ? m_init[j] : channel_llr[j];
            float lam = (1.0f - gamma[j]) * channel_llr[j] + gamma[j] * Mp;
            if (lam >  LAMBDA_CLIP) lam =  LAMBDA_CLIP;
            if (lam < -LAMBDA_CLIP) lam = -LAMBDA_CLIP;
            m_prev[j]      = Mp;
            lambda_prev[j] = lam;
            llr[j]         = lam;
        }
    }

    for (int i = 0; i < n_checks; i++) check_order[i] = i;
    unsigned int rng = (rng_seed != 0) ? rng_seed : 42u;

    /* Per-edge v2c value buffer (stack). */
    float v2c_val[MAX_DEG];

    const uint32_t SIGN_MASK = 0x80000000u;

    int last_residual = n_checks;
    int last_unsat_check = -1;
    int it;
    for (it = 1; it <= max_iter; it++) {

        if (random_order == 1) {
            shuffle(check_order, n_checks, &rng);
        } else if (random_order >= 2) {
            block_shuffle(check_order, n_checks, random_order, &rng);
        }

        float alpha = alpha_schedule[it - 1];

        /* ── relay-memory bias swap (it > 1; it==1 handled at init) ──
         * llr currently == Lambda(t-1) + sum(c2v). Recompute the bias from
         * the previous iteration's marginal M(t-1) (= m_prev) and add the
         * delta, preserving the message sum. */
        if (gamma != NULL && it > 1) {
            for (int j = 0; j < n_vars; j++) {
                float lam = (1.0f - gamma[j]) * channel_llr[j] + gamma[j] * m_prev[j];
                if (lam >  LAMBDA_CLIP) lam =  LAMBDA_CLIP;
                if (lam < -LAMBDA_CLIP) lam = -LAMBDA_CLIP;
                llr[j]        += lam - lambda_prev[j];
                lambda_prev[j] = lam;
            }
        }

        /* ── Check-serial sweep using min1/min2 single-pass ── */
        for (int ci = 0; ci < n_checks; ci++) {
            int i = check_order[ci];
            int start = indptr[i];
            int end   = indptr[i + 1];
            int deg   = end - start;
            if (deg == 0) continue;

            float min1a = 1e30f, min2a = 1e30f;
            float min1b = 1e30f, min2b = 1e30f;
            int   argmina = 0, argminb = 1;
            uint32_t sign_acc = ((uint32_t)syndrome[i]) << 31; /* 0 or 0x80000000 */

            int k = 0;
            int deg2 = deg & ~1;
            for (; k < deg2; k += 2) {
                if (k + 16 < deg) {
                    __builtin_prefetch(&llr[indices[start + k + 16]], 0, 1);
                    __builtin_prefetch(&llr[indices[start + k + 17]], 0, 1);
                }
                float vv0 = llr[indices[start + k    ]] - c2v[start + k    ];
                float vv1 = llr[indices[start + k + 1]] - c2v[start + k + 1];
                v2c_val[k    ] = vv0;
                v2c_val[k + 1] = vv1;

                uint32_t vvb0, vvb1;
                memcpy(&vvb0, &vv0, sizeof(vvb0));
                memcpy(&vvb1, &vv1, sizeof(vvb1));
                sign_acc ^= (vvb0 & SIGN_MASK) ^ (vvb1 & SIGN_MASK);

                uint32_t avb0 = vvb0 & 0x7FFFFFFFu;
                uint32_t avb1 = vvb1 & 0x7FFFFFFFu;
                float av0, av1;
                memcpy(&av0, &avb0, sizeof(av0));
                memcpy(&av1, &avb1, sizeof(av1));

                if (av0 < min1a) {
                    min2a = min1a;
                    min1a = av0;
                    argmina = k;
                } else if (av0 < min2a) {
                    min2a = av0;
                }
                if (av1 < min1b) {
                    min2b = min1b;
                    min1b = av1;
                    argminb = k + 1;
                } else if (av1 < min2b) {
                    min2b = av1;
                }
            }
            /* Tail */
            for (; k < deg; k++) {
                float vv = llr[indices[start + k]] - c2v[start + k];
                v2c_val[k] = vv;
                uint32_t vvb;
                memcpy(&vvb, &vv, sizeof(vvb));
                sign_acc ^= (vvb & SIGN_MASK);
                uint32_t avb = vvb & 0x7FFFFFFFu;
                float av;
                memcpy(&av, &avb, sizeof(av));
                if (av < min1a) {
                    min2a = min1a;
                    min1a = av;
                    argmina = k;
                } else if (av < min2a) {
                    min2a = av;
                }
            }

            /* Merge the two min-streams. */
            float min1, min2;
            int   argmin;
            if (min1a <= min1b) {
                min1 = min1a;
                argmin = argmina;
                min2 = (min2a < min1b) ? min2a : min1b;
            } else {
                min1 = min1b;
                argmin = argminb;
                min2 = (min2b < min1a) ? min2b : min1a;
            }

            float m1, m2;
            if (beta > 0.0f) {
                m1 = min1 - beta; if (m1 < 0.0f) m1 = 0.0f;
                m2 = min2 - beta; if (m2 < 0.0f) m2 = 0.0f;
                m1 *= alpha;
                m2 *= alpha;
            } else {
                m1 = min1 * alpha;
                m2 = min2 * alpha;
            }

            uint32_t m1b, m2b;
            memcpy(&m1b, &m1, sizeof(m1b));
            memcpy(&m2b, &m2, sizeof(m2b));

            /* Main pass: every edge gets new_c2v = sign(sign_acc ^ sign(vv)) * m1 */
            for (int kk = 0; kk < deg; kk++) {
                float vv = v2c_val[kk];
                uint32_t vvb;
                memcpy(&vvb, &vv, sizeof(vvb));
                uint32_t outb = m1b ^ sign_acc ^ (vvb & SIGN_MASK);
                float new_c2v;
                memcpy(&new_c2v, &outb, sizeof(new_c2v));
                int j = indices[start + kk];
                llr[j] = vv + new_c2v;
                c2v[start + kk] = new_c2v;
            }

            /* Patch argmin: replace its m1 contribution with m2. */
            {
                int kk = argmin;
                float vv = v2c_val[kk];
                uint32_t vvb;
                memcpy(&vvb, &vv, sizeof(vvb));
                uint32_t outb = m2b ^ sign_acc ^ (vvb & SIGN_MASK);
                float new_c2v;
                memcpy(&new_c2v, &outb, sizeof(new_c2v));
                int j = indices[start + kk];
                float old_c2v = c2v[start + kk];
                llr[j] = llr[j] - old_c2v + new_c2v;
                c2v[start + kk] = new_c2v;
            }
        }

        /* ── relay-memory snapshot: M(it) = current marginal (for next iter) ──
         * Placed before the convergence check; if we converge we exit and the
         * snapshot is unused (harmless). */
        if (gamma != NULL) {
            memcpy(m_prev, llr, (size_t)n_vars * sizeof(float));
        }

        /* ── Convergence check ── */
        {
            int residual = 0;
            if (random_order < 0) {
                for (int i = 0; i < n_checks; i++) {
                    uint8_t s = 0;
                    for (int k = indptr[i]; k < indptr[i + 1]; k++) {
                        if (llr[indices[k]] < 0.0f) s ^= 1;
                    }
                    if (s != syndrome[i]) {
                        residual++;
                        check_order[i] = -1;
                    } else {
                        check_order[i] = 0;
                    }
                }
                int *buf = (int *)malloc((size_t)n_checks * sizeof(int));
                if (buf == NULL) {
                    *out_iters = it;
                    *out_converged = 0;
                    *out_residual = residual;
                    for (int j = 0; j < n_vars; j++)
                        decoding[j] = (llr[j] < 0.0f) ? 1 : 0;
                    return;
                }
                int n_unsat = 0, n_sat = 0;
                for (int i = 0; i < n_checks; i++) {
                    if (check_order[i] == -1)
                        buf[n_unsat++] = i;
                }
                int offset = n_unsat;
                for (int i = 0; i < n_checks; i++) {
                    if (check_order[i] == 0)
                        buf[offset + n_sat++] = i;
                }
                shuffle(buf, n_unsat, &rng);
                shuffle(buf + n_unsat, n_sat, &rng);
                memcpy(check_order, buf, (size_t)n_checks * sizeof(int));
                free(buf);
                last_residual = residual;
                if (residual == 0) {
                    *out_iters = it;
                    *out_converged = 1;
                    *out_residual = 0;
                    goto done;
                }
            } else {
                int any_unsat = 0;

                if (last_unsat_check >= 0) {
                    int i = last_unsat_check;
                    uint8_t s = 0;
                    int kend = indptr[i + 1];
                    for (int k = indptr[i]; k < kend; k++) {
                        uint32_t bits;
                        memcpy(&bits, &llr[indices[k]], sizeof(bits));
                        s ^= (uint8_t)(bits >> 31);
                    }
                    if (s != syndrome[i]) {
                        any_unsat = 1;
                    }
                }

                if (!any_unsat) {
                    for (int i = 0; i < n_checks; i++) {
                        uint8_t s = 0;
                        int kend = indptr[i + 1];
                        for (int k = indptr[i]; k < kend; k++) {
                            uint32_t bits;
                            memcpy(&bits, &llr[indices[k]], sizeof(bits));
                            s ^= (uint8_t)(bits >> 31);
                        }
                        if (s != syndrome[i]) {
                            any_unsat = 1;
                            last_unsat_check = i;
                            break;
                        }
                    }
                }

                if (!any_unsat) {
                    *out_iters = it;
                    *out_converged = 1;
                    *out_residual = 0;
                    goto done;
                }
                last_residual = 1;
            }
        }
    }

    /* Reached max_iter without convergence: compute exact residual. */
    {
        int residual = 0;
        for (int i = 0; i < n_checks; i++) {
            uint8_t s = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; k++) {
                if (llr[indices[k]] < 0.0f) s ^= 1;
            }
            if (s != syndrome[i]) residual++;
        }
        last_residual = residual;
    }

    *out_iters = max_iter;
    *out_converged = 0;
    *out_residual = last_residual;

done:
    for (int j = 0; j < n_vars; j++) {
        decoding[j] = (llr[j] < 0.0f) ? 1 : 0;
    }
}
