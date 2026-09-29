#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------
// Tuning Parameters (Injected by Python via nvcc -D)
// Fallback defaults are provided for standalone compilation
// ---------------------------------------------------------
#define PRAGMA(x) _Pragma(#x)
#define UNROLL_HELPER(x) PRAGMA(unroll x)
#define UNROLL(x) UNROLL_HELPER(x)

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

#ifndef THREAD_X
#define THREAD_X 16
#endif
#ifndef THREAD_Y
#define THREAD_Y 16
#endif
#ifndef THREAD_Z
#define THREAD_Z 1
#endif

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

#ifndef MAX_STATIC_SHARED_MEMORY_PER_BLOCK
#define MAX_STATIC_SHARED_MEMORY_PER_BLOCK 49152
#endif

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 4194304 // in bytes
#endif

// Relative variance threshold: (Variance / Mean^2)
// Default threshold 0.0025 corresponds to a ~5% relative standard deviation.
#ifndef VARIANCE_THRESHOLD
#define VARIANCE_THRESHOLD 0.0025
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

// ---------------------------------------------------------
// CUTLASS-Inspired Block-Tiled GEMM Kernel
// C = A * B
// ---------------------------------------------------------
__global__ void gemm_kernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int WORK_M = (TILE_M + THREAD_Y - 1) / THREAD_Y;
    const int WORK_N = (TILE_N + THREAD_X - 1) / THREAD_X;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float accum[WORK_M][WORK_N] = {0.0f};

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    for (int k_step = 0; k_step < K; k_step += TILE_K)
    {
        for (int i = tid; i < TILE_M * TILE_K; i += num_threads)
        {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = by * TILE_M + row;
            int global_col = k_step + col;
            As[row][col] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
        }

        for (int i = tid; i < TILE_K * TILE_N; i += num_threads)
        {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int global_row = k_step + row;
            int global_col = bx * TILE_N + col;
            Bs[row][col] = (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
        }
        __syncthreads();

        UNROLL(UNROLL_FACTOR)
        for (int k = 0; k < TILE_K; ++k)
        {
            for (int wm = 0; wm < WORK_M; ++wm)
            {
                for (int wn = 0; wn < WORK_N; ++wn)
                {
                    int tile_row = ty * WORK_M + wm;
                    int tile_col = tx * WORK_N + wn;

                    if (tile_row < TILE_M && tile_col < TILE_N)
                    {
                        accum[wm][wn] += As[tile_row][k] * Bs[k][tile_col];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int wm = 0; wm < WORK_M; ++wm)
    {
        for (int wn = 0; wn < WORK_N; ++wn)
        {
            int tile_row = ty * WORK_M + wm;
            int tile_col = tx * WORK_N + wn;

            // Ensure we don't attempt to map an out-of-bounds tile index
            // back to a global memory coordinate.
            if (tile_row < TILE_M && tile_col < TILE_N)
            {
                int global_row = by * TILE_M + tile_row;
                int global_col = bx * TILE_N + tile_col;
                if (global_row < M && global_col < N)
                {
                    C[global_row * N + global_col] = accum[wm][wn];
                }
            }
        }
    }
}

void init_random_matrix(float *d_ptr, size_t num_elements, float min_val = -1.0f, float max_val = 1.0f)
{
    std::vector<float> h_data(num_elements);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < num_elements; ++i)
    {
        h_data[i] = dist(gen);
    }

    cudaMemcpy(d_ptr, h_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
}

// Compute variance and mean of measured runtimes
double compute_variance(const std::vector<float> &data, double &mean_out)
{
    if (data.empty())
        return 0.0;

    double sum = 0.0;
    for (float val : data)
        sum += val;
    mean_out = sum / data.size();

    double sq_diff_sum = 0.0;
    for (float val : data)
    {
        double diff = val - mean_out;
        sq_diff_sum += diff * diff;
    }
    return sq_diff_sum / data.size();
}

// Compute median time from vector
float compute_median(std::vector<float> data)
{
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0)
    {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0f;
    }
    else
    {
        return data[n / 2];
    }
}

int main(int argc, char **argv)
{
    int threads_per_block = THREAD_X * THREAD_Y * THREAD_Z;
    if (threads_per_block > MAX_THREADS_PER_BLOCK || threads_per_block <= 0)
    {
        std::cout << "{\"status\": \"invalid_config\", \"error_message\": \"Thread count per block exceeds max limit\"}" << std::endl;
        return 0;
    }

    size_t shared_mem_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    if (shared_mem_bytes > MAX_STATIC_SHARED_MEMORY_PER_BLOCK)
    {
        std::cout << "{\"status\": \"invalid_config\", \"error_message\": \"Static shared memory request exceeds max limit\"}" << std::endl;
        return 0;
    }

    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int K = (argc > 3) ? std::atoi(argv[3]) : 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaError_t err_a = cudaMalloc(&d_A, bytes_A);
    cudaError_t err_b = cudaMalloc(&d_B, bytes_B);
    cudaError_t err_c = cudaMalloc(&d_C, bytes_C);

    if (err_a != cudaSuccess || err_b != cudaSuccess || err_c != cudaSuccess)
    {
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);
        std::cout << "{\"status\": \"cuda_oom\", \"error_message\": \"Memory allocation failed for matrices\"}" << std::endl;
        return 0;
    }

    float *d_flush = nullptr;
    if (L2_CACHE_SIZE > 0)
    {
        cudaError_t err_flush = cudaMalloc(&d_flush, L2_CACHE_SIZE);
        if (err_flush != cudaSuccess)
        {

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            std::cout << "{\"status\": \"cuda_oom\", \"error_message\": \"Failed to allocate L2 cache flush buffer\"}" << std::endl;
            return 0;
        }
    }

    init_random_matrix(d_A, M * K, -0.5f, 0.5f);
    init_random_matrix(d_B, K * N, -0.5f, 0.5f);
    cudaMemset(d_C, 0, bytes_C);

    dim3 threads(THREAD_X, THREAD_Y, THREAD_Z);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Warmup (10 times)
    for (int i = 0; i < 10; ++i)
    {
        gemm_kernel<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
    }

    cudaError_t launch_err = cudaGetLastError();

    if (launch_err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        if (d_flush)
            cudaFree(d_flush);
        std::cout << "{\"status\": \"launch_failure\", \"error_message\": \"Kernel launch failed\"}" << std::endl;
        return 0;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        if (d_flush)
            cudaFree(d_flush);
        std::cout << "{\"status\": \"execution_failure\", \"error_message\": \"Kernel execution failed\"}" << std::endl;
        return 0;
    }

    // 2. Adaptive Benchmarking Logic (up to 3 batches of 20 runs = max 60)
    std::vector<float> run_times;
    run_times.reserve(60);

    const int BATCH_SIZE = 20;
    const int MAX_BATCHES = 3;
    double final_mean = 0.0;
    double final_variance = 0.0;
    double final_rel_variance = 0.0;

    for (int batch = 0; batch < MAX_BATCHES; ++batch)
    {
        for (int i = 0; i < BATCH_SIZE; ++i)
        {
            if (L2_CACHE_SIZE > 0)
            {
                cudaMemset(d_flush, 0, L2_CACHE_SIZE);
            }

            cudaEventRecord(start);
            gemm_kernel<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            run_times.push_back(ms);
        }

        final_variance = compute_variance(run_times, final_mean);
        final_rel_variance = (final_mean > 0.0) ? (final_variance / (final_mean * final_mean)) : 0.0;

        if (final_rel_variance < VARIANCE_THRESHOLD)
        {
            break;
        }
    }

    float median_time = compute_median(run_times);

    // Output JSON string
    std::cout << "{"
              << "\"status\": \"success\", "
              << "\"median_ms\": " << median_time << ", "
              << "\"mean_ms\": " << final_mean << ", "
              << "\"variance\": " << final_variance << ", "
              << "\"rel_variance\": " << final_rel_variance << ", "
              << "\"iterations\": " << run_times.size()
              << "}" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (d_flush)
        cudaFree(d_flush);
    return 0;
}