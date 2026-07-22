/*
 * sqetch -- GPU random information-set decoder for CSS quantum codes.
 *
 * Two kernels share a random column permutation, a k_sub sketch of
 * ker(H_check), and a collaborative RREF; they differ only in the final
 * scan:
 *   sqetch_ksub_kernel           -- minimum logical-coset weight.
 *   sqetch_ksub_recover_kernel   -- also writes back the winning row.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128

__device__ __forceinline__ uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

__device__ __forceinline__ int row_weight(const uint64_t* row, int nw) {
    int w = 0;
    for (int i = 0; i < nw; i++) w += __popcll(row[i]);
    return w;
}

__device__ __forceinline__ int gf2_dot(const uint64_t* a, const uint64_t* b, int nw) {
    uint64_t acc = 0;
    for (int i = 0; i < nw; i++) acc ^= (a[i] & b[i]);
    acc ^= acc >> 32; acc ^= acc >> 16; acc ^= acc >> 8;
    acc ^= acc >> 4;  acc ^= acc >> 2;  acc ^= acc >> 1;
    return (int)(acc & 1);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
sqetch_ksub_kernel(
    const uint64_t* __restrict__ W_null,
    const uint64_t* __restrict__ W_logical,
    int k,
    int kx,
    int nw,
    int n,
    int k_sub,
    int* global_best,
    unsigned long long base_seed,
    int d_target,
    int* found_flag
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char raw_shmem[];

    uint16_t* perm = (uint16_t*)raw_shmem;
    int perm_bytes = (n * 2 + 7) & ~7;

    uint64_t* W_sub = (uint64_t*)(raw_shmem + perm_bytes);
    int wsub_bytes = k_sub * nw * 8;

    uint64_t* pivot_row_shmem = (uint64_t*)(raw_shmem + perm_bytes + wsub_bytes);
    int pivot_bytes = nw * 8;

    int* control = (int*)(raw_shmem + perm_bytes + wsub_bytes + pivot_bytes);
    int* thread_best = control + 2;

    uint64_t block_seed = base_seed ^ ((uint64_t)bid * 6364136223846793005ULL + 1442695040888963407ULL);

    if (tid == 0) {
        uint64_t rng = block_seed;
        xorshift64(&rng);
        for (int i = 0; i < n; i++) perm[i] = (uint16_t)i;
        for (int i = n - 1; i > 0; i--) {
            int j = (int)(xorshift64(&rng) % (uint64_t)(i + 1));
            uint16_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
        control[1] = 0;
    }

    uint64_t thread_rng = block_seed ^ ((uint64_t)tid * 2685821657736338717ULL + 1);
    xorshift64(&thread_rng);

    for (int s = tid; s < k_sub; s += BLOCK_SIZE) {
        int src_row = (int)(xorshift64(&thread_rng) % (uint64_t)k);
        const uint64_t* src = W_null + (size_t)src_row * nw;
        uint64_t* dst = W_sub + (size_t)s * nw;
        for (int w = 0; w < nw; w++) dst[w] = __ldg(src + w);
    }
    __syncthreads();

    for (int c_virt = 0; c_virt < n; c_virt++) {
        int pr = control[1];
        if (pr >= k_sub) { __syncthreads(); __syncthreads(); continue; }

        int c_phys = (int)perm[c_virt];
        int word = c_phys >> 6;
        int bit_pos = c_phys & 63;
        uint64_t mask = (uint64_t)1 << bit_pos;

        if (tid == 0) {
            int found = -1;
            for (int r = pr; r < k_sub; r++) {
                if (W_sub[(size_t)r * nw + word] & mask) { found = r; break; }
            }
            control[0] = found;
            if (found != -1) {
                if (found != pr) {
                    for (int w = 0; w < nw; w++) {
                        uint64_t tmp = W_sub[(size_t)pr * nw + w];
                        W_sub[(size_t)pr * nw + w] = W_sub[(size_t)found * nw + w];
                        W_sub[(size_t)found * nw + w] = tmp;
                    }
                }
                for (int w = 0; w < nw; w++)
                    pivot_row_shmem[w] = W_sub[(size_t)pr * nw + w];
                control[1] = pr + 1;
            }
        }
        __syncthreads();

        if (control[0] == -1) { __syncthreads(); continue; }

        int pr2 = pr;
        for (int r = tid; r < k_sub; r += BLOCK_SIZE) {
            if (r != pr2 && (W_sub[(size_t)r * nw + word] & mask)) {
                for (int w = 0; w < nw; w++)
                    W_sub[(size_t)r * nw + w] ^= pivot_row_shmem[w];
            }
        }
        __syncthreads();
    }

    thread_best[tid] = n + 1;

    for (int r = tid; r < k_sub; r += BLOCK_SIZE) {
        uint64_t* row = W_sub + (size_t)r * nw;
        int all_zero = 1;
        for (int w = 0; w < nw; w++) if (row[w]) { all_zero = 0; break; }
        if (all_zero) continue;

        int wt = row_weight(row, nw);
        if (wt >= thread_best[tid]) continue;

        int is_logical = 0;
        for (int rx = 0; rx < kx && !is_logical; rx++)
            if (gf2_dot(W_logical + (size_t)rx * nw, row, nw)) is_logical = 1;

        if (is_logical && wt < thread_best[tid])
            thread_best[tid] = wt;
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            if (thread_best[tid + stride] < thread_best[tid])
                thread_best[tid] = thread_best[tid + stride];
        __syncthreads();
    }

    if (tid == 0 && thread_best[0] <= n) {
        atomicMin(global_best, thread_best[0]);
        if (d_target >= 0 && thread_best[0] < d_target && found_flag)
            atomicExch(found_flag, 1);
    }
}


torch::Tensor run_sqetch_ksub(
    torch::Tensor W_null,
    torch::Tensor W_logical,
    int n,
    int k_sub,
    int num_trials,
    uint64_t seed,
    int d_target
) {
    int k  = (int)W_null.size(0);
    int kx = (int)W_logical.size(0);
    int nw = (int)W_null.size(1);

    auto dev = W_null.device();
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    torch::Tensor best = torch::full({1}, n + 1, opt_i32);
    torch::Tensor flag = torch::zeros({1}, opt_i32);

    int perm_bytes = (n * 2 + 7) & ~7;
    int wsub_bytes = k_sub * nw * 8;
    int pivot_bytes = nw * 8;
    int ctrl_bytes = 8;
    int best_bytes = BLOCK_SIZE * 4;
    int shmem_bytes = perm_bytes + wsub_bytes + pivot_bytes + ctrl_bytes + best_bytes;

    int shmem_request = (shmem_bytes + 4095) & ~4095;
    cudaFuncSetAttribute(
        sqetch_ksub_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_request
    );
    cudaFuncSetCacheConfig(sqetch_ksub_kernel, cudaFuncCachePreferShared);

    sqetch_ksub_kernel<<<num_trials, BLOCK_SIZE, shmem_bytes>>>(
        (const uint64_t*)W_null.data_ptr<int64_t>(),
        (const uint64_t*)W_logical.data_ptr<int64_t>(),
        k, kx, nw, n, k_sub,
        best.data_ptr<int32_t>(),
        (unsigned long long)seed,
        d_target,
        flag.data_ptr<int32_t>()
    );

    return torch::cat({best, flag});
}


__global__ void __launch_bounds__(BLOCK_SIZE)
sqetch_ksub_recover_kernel(
    const uint64_t* __restrict__ W_null,
    const uint64_t* __restrict__ W_logical,
    int k, int kx, int nw, int n, int k_sub,
    int* global_best,
    unsigned long long base_seed,
    int d_target,
    int* found_flag,
    int64_t* out_vec,
    int32_t* out_perm,
    int* done_flag
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char raw_shmem[];

    uint16_t* perm = (uint16_t*)raw_shmem;
    int perm_bytes = (n * 2 + 7) & ~7;

    uint64_t* W_sub = (uint64_t*)(raw_shmem + perm_bytes);
    int wsub_bytes = k_sub * nw * 8;

    uint64_t* pivot_row_shmem = (uint64_t*)(raw_shmem + perm_bytes + wsub_bytes);
    int pivot_bytes = nw * 8;

    int* control = (int*)(raw_shmem + perm_bytes + wsub_bytes + pivot_bytes);
    int* thread_best = control + 2;
    int* thread_best_row = thread_best + BLOCK_SIZE;

    uint64_t block_seed = base_seed ^ ((uint64_t)bid * 6364136223846793005ULL + 1442695040888963407ULL);

    if (tid == 0) {
        uint64_t rng = block_seed;
        xorshift64(&rng);
        for (int i = 0; i < n; i++) perm[i] = (uint16_t)i;
        for (int i = n - 1; i > 0; i--) {
            int j = (int)(xorshift64(&rng) % (uint64_t)(i + 1));
            uint16_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
        control[1] = 0;
    }

    uint64_t thread_rng = block_seed ^ ((uint64_t)tid * 2685821657736338717ULL + 1);
    xorshift64(&thread_rng);

    for (int s = tid; s < k_sub; s += BLOCK_SIZE) {
        int src_row = (int)(xorshift64(&thread_rng) % (uint64_t)k);
        const uint64_t* src = W_null + (size_t)src_row * nw;
        uint64_t* dst = W_sub + (size_t)s * nw;
        for (int w = 0; w < nw; w++) dst[w] = __ldg(src + w);
    }
    __syncthreads();

    for (int c_virt = 0; c_virt < n; c_virt++) {
        int pr = control[1];
        if (pr >= k_sub) { __syncthreads(); __syncthreads(); continue; }

        int c_phys = (int)perm[c_virt];
        int word = c_phys >> 6;
        int bit_pos = c_phys & 63;
        uint64_t mask = (uint64_t)1 << bit_pos;

        if (tid == 0) {
            int found = -1;
            for (int r = pr; r < k_sub; r++) {
                if (W_sub[(size_t)r * nw + word] & mask) { found = r; break; }
            }
            control[0] = found;
            if (found != -1) {
                if (found != pr) {
                    for (int w = 0; w < nw; w++) {
                        uint64_t tmp = W_sub[(size_t)pr * nw + w];
                        W_sub[(size_t)pr * nw + w] = W_sub[(size_t)found * nw + w];
                        W_sub[(size_t)found * nw + w] = tmp;
                    }
                }
                for (int w = 0; w < nw; w++)
                    pivot_row_shmem[w] = W_sub[(size_t)pr * nw + w];
                control[1] = pr + 1;
            }
        }
        __syncthreads();

        if (control[0] == -1) { __syncthreads(); continue; }

        int pr2 = pr;
        for (int r = tid; r < k_sub; r += BLOCK_SIZE) {
            if (r != pr2 && (W_sub[(size_t)r * nw + word] & mask)) {
                for (int w = 0; w < nw; w++)
                    W_sub[(size_t)r * nw + w] ^= pivot_row_shmem[w];
            }
        }
        __syncthreads();
    }

    thread_best[tid] = n + 1;
    thread_best_row[tid] = -1;
    for (int r = tid; r < k_sub; r += BLOCK_SIZE) {
        uint64_t* row = W_sub + (size_t)r * nw;
        int all_zero = 1;
        for (int w = 0; w < nw; w++) if (row[w]) { all_zero = 0; break; }
        if (all_zero) continue;

        int wt = row_weight(row, nw);
        if (wt >= thread_best[tid]) continue;

        int is_logical = 0;
        for (int rx = 0; rx < kx && !is_logical; rx++)
            if (gf2_dot(W_logical + (size_t)rx * nw, row, nw)) is_logical = 1;

        if (is_logical && wt < thread_best[tid]) {
            thread_best[tid] = wt;
            thread_best_row[tid] = r;
        }
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && thread_best[tid + stride] < thread_best[tid]) {
            thread_best[tid] = thread_best[tid + stride];
            thread_best_row[tid] = thread_best_row[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && thread_best[0] <= n) {
        int wt = thread_best[0];
        atomicMin(global_best, wt);
        if (d_target >= 0 && wt < d_target) {
            if (found_flag) atomicExch(found_flag, 1);
            if (out_vec && out_perm && done_flag &&
                atomicExch(done_flag, 1) == 0) {
                control[0] = thread_best_row[0];
            } else {
                control[0] = -1;
            }
        } else {
            control[0] = -1;
        }
    }
    __syncthreads();

    int winner_row = control[0];
    if (winner_row >= 0) {
        const uint64_t* wrow = W_sub + (size_t)winner_row * nw;
        for (int w = tid; w < nw; w += BLOCK_SIZE)
            out_vec[w] = (int64_t)wrow[w];
        for (int i = tid; i < n; i += BLOCK_SIZE)
            out_perm[i] = (int32_t)perm[i];
    }
}


std::vector<torch::Tensor> run_sqetch_ksub_recover(
    torch::Tensor W_null,
    torch::Tensor W_logical,
    int n,
    int k_sub,
    int num_trials,
    uint64_t seed,
    int d_target
) {
    int k  = (int)W_null.size(0);
    int kx = (int)W_logical.size(0);
    int nw = (int)W_null.size(1);

    auto dev = W_null.device();
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    torch::Tensor best     = torch::full({1}, n + 1, opt_i32);
    torch::Tensor flag     = torch::zeros({1}, opt_i32);
    torch::Tensor done     = torch::zeros({1}, opt_i32);
    torch::Tensor out_vec  = torch::zeros({nw}, opt_i64);
    torch::Tensor out_perm = torch::zeros({n},  opt_i32);

    int perm_bytes = (n * 2 + 7) & ~7;
    int wsub_bytes = k_sub * nw * 8;
    int pivot_bytes = nw * 8;
    int ctrl_bytes = 8;
    int best_bytes = BLOCK_SIZE * 4;
    int brow_bytes = BLOCK_SIZE * 4;
    int shmem_bytes = perm_bytes + wsub_bytes + pivot_bytes + ctrl_bytes + best_bytes + brow_bytes;

    int shmem_request = (shmem_bytes + 4095) & ~4095;
    cudaFuncSetAttribute(
        sqetch_ksub_recover_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_request
    );
    cudaFuncSetCacheConfig(sqetch_ksub_recover_kernel, cudaFuncCachePreferShared);

    sqetch_ksub_recover_kernel<<<num_trials, BLOCK_SIZE, shmem_bytes>>>(
        (const uint64_t*)W_null.data_ptr<int64_t>(),
        (const uint64_t*)W_logical.data_ptr<int64_t>(),
        k, kx, nw, n, k_sub,
        best.data_ptr<int32_t>(),
        (unsigned long long)seed,
        d_target,
        flag.data_ptr<int32_t>(),
        out_vec.data_ptr<int64_t>(),
        out_perm.data_ptr<int32_t>(),
        done.data_ptr<int32_t>()
    );

    return {torch::cat({best, flag, done}), out_vec, out_perm};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_sqetch_ksub", &run_sqetch_ksub,
          "sqetch k_sub-subsampled random ISD kernel",
          py::arg("W_null"), py::arg("W_logical"),
          py::arg("n"), py::arg("k_sub"),
          py::arg("num_trials"), py::arg("seed"), py::arg("d_target"));
    m.def("run_sqetch_ksub_recover", &run_sqetch_ksub_recover,
          "sqetch recovery kernel -- also returns winning codeword + permutation",
          py::arg("W_null"), py::arg("W_logical"),
          py::arg("n"), py::arg("k_sub"),
          py::arg("num_trials"), py::arg("seed"), py::arg("d_target"));
}
