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

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 256
#endif

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 4
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

#ifndef MAX_STATIC_SHARED_MEMORY_PER_BLOCK
#define MAX_STATIC_SHARED_MEMORY_PER_BLOCK 49152
#endif

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 1048576 // ? For nvidia gtx 1650 ti
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
// Block-Tiled GEMV Kernel
// y = A * x
// A is (M x K), x is (K x 1), y is (M x 1)
//
// Each block computes ROWS_PER_BLOCK output elements.
// BLOCK_DIM_X threads cooperatively reduce across K.
// x is cached in shared memory (TILE_K at a time) for
// reuse across the multiple rows within a block.
// ---------------------------------------------------------
__global__ void gemv_kernel(int M, int K, const float *A, const float *x, float *y)
{
    __shared__ float x_shared[TILE_K];
    __shared__ float partial_sums[ROWS_PER_BLOCK][BLOCK_DIM_X];

    int tx = threadIdx.x;
    int row_base = blockIdx.x * ROWS_PER_BLOCK;

    // Initialize partial sums to zero
    for (int r = 0; r < ROWS_PER_BLOCK; r++)
    {
        partial_sums[r][tx] = 0.0f;
    }

    // Outer loop: tile over K dimension
    for (int k_start = 0; k_start < K; k_start += TILE_K)
    {
        // Cooperatively load a tile of x into shared memory
        for (int i = tx; i < TILE_K; i += BLOCK_DIM_X)
        {
            int gk = k_start + i;
            x_shared[i] = (gk < K) ? x[gk] : 0.0f;
        }
        __syncthreads();

        // Each thread strides within the tile, accumulating for each row
        for (int r = 0; r < ROWS_PER_BLOCK; r++)
        {
            int global_row = row_base + r;
            if (global_row < M)
            {
                float local_sum = 0.0f;
                UNROLL(UNROLL_FACTOR)
                for (int k = tx; k < TILE_K; k += BLOCK_DIM_X)
                {
                    int gk = k_start + k;
                    if (gk < K)
                    {
                        local_sum += A[global_row * K + gk] * x_shared[k];
                    }
                }
                partial_sums[r][tx] += local_sum;
            }
        }
        __syncthreads();
    }

    // Tree reduction in shared memory for each row
    for (int r = 0; r < ROWS_PER_BLOCK; r++)
    {
        for (int stride = BLOCK_DIM_X / 2; stride > 0; stride >>= 1)
        {
            if (tx < stride)
            {
                partial_sums[r][tx] += partial_sums[r][tx + stride];
            }
            __syncthreads();
        }

        int global_row = row_base + r;
        if (tx == 0 && global_row < M)
        {
            y[global_row] = partial_sums[r][0];
        }
        __syncthreads();
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
    int threads_per_block = BLOCK_DIM_X;

    if (threads_per_block > MAX_THREADS_PER_BLOCK || threads_per_block <= 0)
    {
        std::cout << "{\"status\": \"invalid_config\", \"error_message\": \"Thread count per block exceeds max limit\"}" << std::endl;
        return 0;
    }

    if ((BLOCK_DIM_X & (BLOCK_DIM_X - 1)) != 0)
    {
        std::cout << "{\"status\": \"invalid_config\", \"error_message\": \"BLOCK_DIM_X must be a power of two\"}" << std::endl;
        return 0;
    }

    // Shared memory: x_shared[TILE_K] + partial_sums[ROWS_PER_BLOCK][BLOCK_DIM_X]
    size_t shared_mem_bytes = (TILE_K + ROWS_PER_BLOCK * BLOCK_DIM_X) * sizeof(float);
    if (shared_mem_bytes > MAX_STATIC_SHARED_MEMORY_PER_BLOCK)
    {
        std::cout << "{\"status\": \"invalid_config\", \"error_message\": \"Static shared memory request exceeds max limit\"}" << std::endl;
        return 0;
    }

    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int K = (argc > 2) ? std::atoi(argv[2]) : 1024;

    size_t bytes_A = (size_t)M * (size_t)K * sizeof(float);
    size_t bytes_x = (size_t)K * sizeof(float);
    size_t bytes_y = (size_t)M * sizeof(float);

    float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;

    cudaError_t err_a = cudaMalloc(&d_A, bytes_A);
    cudaError_t err_x = cudaMalloc(&d_x, bytes_x);
    cudaError_t err_y = cudaMalloc(&d_y, bytes_y);

    if (err_a != cudaSuccess || err_x != cudaSuccess || err_y != cudaSuccess)
    {
        if (d_A)
            cudaFree(d_A);
        if (d_x)
            cudaFree(d_x);
        if (d_y)
            cudaFree(d_y);
        std::cout << "{\"status\": \"cuda_oom\", \"error_message\": \"Memory allocation failed for matrix/vectors\"}" << std::endl;
        return 0;
    }

    float *d_flush = nullptr;
    if (L2_CACHE_SIZE > 0)
    {
        cudaError_t err_flush = cudaMalloc(&d_flush, L2_CACHE_SIZE);
        if (err_flush != cudaSuccess)
        {

            cudaFree(d_A);
            cudaFree(d_x);
            cudaFree(d_y);
            std::cout << "{\"status\": \"cuda_oom\", \"error_message\": \"Failed to allocate L2 cache flush buffer\"}" << std::endl;
            return 0;
        }
    }

    init_random_matrix(d_A, M * K, -0.5f, 0.5f);
    init_random_matrix(d_x, K, -0.5f, 0.5f);
    cudaMemset(d_y, 0, bytes_y);

    dim3 threads(BLOCK_DIM_X);
    dim3 blocks((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Warmup (10 times)
    for (int i = 0; i < 10; ++i)
    {
        gemv_kernel<<<blocks, threads>>>(M, K, d_A, d_x, d_y);
    }

    cudaError_t launch_err = cudaGetLastError();

    if (launch_err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        if (d_flush)
            cudaFree(d_flush);
        std::cout << "{\"status\": \"launch_failure\", \"error_message\": \"Kernel launch failed\"}" << std::endl;
        return 0;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
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

            cudaDeviceSynchronize();
            cudaEventRecord(start);
            gemv_kernel<<<blocks, threads>>>(M, K, d_A, d_x, d_y);
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
    cudaFree(d_x);
    cudaFree(d_y);

    if (d_flush)
        cudaFree(d_flush);
    return 0;
}