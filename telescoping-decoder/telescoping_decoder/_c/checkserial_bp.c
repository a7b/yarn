/*
 * Check-serial (layered) float32 min-sum BP for S3-A and S3-C.
 *
 * Built on first use by _c/build.py::ensure_lib into
 * ~/.cache/telescoping_decoder/. -march=native is disabled by default to
 * reduce machine-dependent floating-point variation:
 *   gcc -O3 -ffast-math -shared -fPIC -o checkserial_bp.so checkserial_bp.c -lm
 *
 * The exported function is checkserial_bp_decode_fast. It uses a min1/min2
 * check update, caller-owned buffers, optional offset min-sum, configurable
 * check ordering, and residual-based scheduling.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_DEG 2048

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

void checkserial_bp_decode_fast(
    const int *indptr,
    const int *indices,
    int n_checks,
    int n_vars,
    int nnz,
    const uint8_t *syndrome,
    const float *channel_llr,
    int max_iter,
    const float *alpha_schedule,
    float beta,
    int random_order,
    unsigned int rng_seed,
    /* caller-provided buffers */
    float *__restrict__ llr,
    float *__restrict__ c2v,
    int *check_order,
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

    for (int i = 0; i < n_checks; i++) check_order[i] = i;
    unsigned int rng = (rng_seed != 0) ? rng_seed : 42u;

    /* Per-edge v2c value buffer (stack). */
    float v2c_val[MAX_DEG];

    const uint32_t SIGN_MASK = 0x80000000u;

    int last_residual = n_checks;
    /* Cache of last-seen unsatisfied check index; probed first on each
     * subsequent convergence scan so we can early-exit in O(deg) rather
     * than scanning ~n_checks/2 checks on average. */
    int last_unsat_check = -1;
    int it;
    for (it = 1; it <= max_iter; it++) {

        if (random_order == 1) {
            shuffle(check_order, n_checks, &rng);
        } else if (random_order >= 2) {
            block_shuffle(check_order, n_checks, random_order, &rng);
        }

        float alpha = alpha_schedule[it - 1];

        /* ── Check-serial sweep using min1/min2 single-pass ── */
        for (int ci = 0; ci < n_checks; ci++) {
            int i = check_order[ci];
            int start = indptr[i];
            int end   = indptr[i + 1];
            int deg   = end - start;
            if (deg == 0) continue;

            /* Compute v2c[k], min1, min2, argmin, and a 32-bit sign
             * accumulator containing the XOR of the sign bits
             * of all v2c values, seeded with the syndrome bit shifted to
             * the sign position.
             *
             * Unrolled by 2 with two independent min-streams to break the
             * dependency chain on min1 updates. */
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
                /* min2 is min of min2a and min1b */
                min2 = (min2a < min1b) ? min2a : min1b;
            } else {
                min1 = min1b;
                argmin = argminb;
                min2 = (min2b < min1a) ? min2b : min1a;
            }

            /* Apply update: branchless default pass with m1 for every edge,
             * then patch argmin with m2. */
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

            /* Bit-patterns of m1, m2 (both non-negative, so sign bit = 0). */
            uint32_t m1b, m2b;
            memcpy(&m1b, &m1, sizeof(m1b));
            memcpy(&m2b, &m2, sizeof(m2b));

            /* Main pass: every edge gets new_c2v = sign(sign_acc ^ sign(vv)) * m1 */
            for (int kk = 0; kk < deg; kk++) {
                float vv = v2c_val[kk];
                uint32_t vvb;
                memcpy(&vvb, &vv, sizeof(vvb));
                /* sign of new c2v: total_sign XOR sign(vv) (excluding edge kk) */
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
                /* Undo the m1-based update and apply m2 instead. */
                float old_c2v = c2v[start + kk];
                llr[j] = llr[j] - old_c2v + new_c2v;
                c2v[start + kk] = new_c2v;
            }
        }

        /* ── Convergence check ── */
        {
            int residual = 0;
            if (random_order < 0) {
                /* Residual-based scheduling needs per-check sat flags; do
                 * the full scan. */
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
                /* Fast "any unsatisfied?" early-exit scan. Most iterations
                 * fail to converge; we only need the exact residual count
                 * on the final iteration.
                 *
                 * Probe the previously unsatisfied check first. Until
                 * convergence, it often remains unsatisfied and permits an
                 * early exit without scanning every check. */
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
                        /* keep last_unsat_check as-is */
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
                /* Not converged: don't bother computing the exact count
                 * during intermediate iters. Track loosely. */
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
