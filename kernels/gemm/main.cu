#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// Tuning Parameters (Injected by Python via nvcc -D)
// Fallback defaults are provided for standalone compilation
// ---------------------------------------------------------
// Helper macros to force expansion inside pragmas
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

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

// ---------------------------------------------------------
// CUTLASS-Inspired Block-Tiled GEMM Kernel
// C = A * B
// ---------------------------------------------------------
__global__ void gemm_autotune_kernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Determine the work footprint for each individual thread
    const int WORK_M = TILE_M / THREAD_Y;
    const int WORK_N = TILE_N / THREAD_X;

    // Allocate shared memory for the thread block tile
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // Thread-local accumulation registers
    float accum[WORK_M][WORK_N] = {0.0f};

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    // Loop over the K-dimension in steps of TILE_K
    for (int k_step = 0; k_step < K; k_step += TILE_K)
    {

        // 1. Collaborative loading of A and B tiles into shared memory
        // A 1D flattened loop ensures efficient loading regardless of thread shape
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

        // 2. Thread-level matrix multiplication on the tile
        UNROLL(UNROLL_FACTOR)
        for (int k = 0; k < TILE_K; ++k)
        {
            for (int wm = 0; wm < WORK_M; ++wm)
            {
                for (int wn = 0; wn < WORK_N; ++wn)
                {
                    accum[wm][wn] += As[ty * WORK_M + wm][k] * Bs[k][tx * WORK_N + wn];
                }
            }
        }
        __syncthreads();
    }

    // 3. Write thread-local accumulated results back to global memory
    for (int wm = 0; wm < WORK_M; ++wm)
    {
        for (int wn = 0; wn < WORK_N; ++wn)
        {
            int global_row = by * TILE_M + ty * WORK_M + wm;
            int global_col = bx * TILE_N + tx * WORK_N + wn;
            if (global_row < M && global_col < N)
            {
                C[global_row * N + global_col] = accum[wm][wn];
            }
        }
    }
}

// ---------------------------------------------------------
// Host Code: Setup, Launch, and Timing
// ---------------------------------------------------------
int main(int argc, char **argv)
{
    // Accept Matrix Shape (M, N, K) dynamically via command line arguments
    // so we don't have to recompile just to test different matrix sizes.
    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int K = (argc > 3) ? std::atoi(argv[3]) : 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // Initialize with dummy data (omitted for brevity)
    cudaMemset(d_A, 1, bytes_A);
    cudaMemset(d_B, 1, bytes_B);
    cudaMemset(d_C, 0, bytes_C);

    // Setup Grid and Block dimensions based on injected macros
    dim3 threads(THREAD_X, THREAD_Y, THREAD_Z);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    gemm_autotune_kernel<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Profile
    cudaEventRecord(start);
    int iterations = 10;
    for (int i = 0; i < iterations; ++i)
    {
        gemm_autotune_kernel<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / iterations;

    // Output solely the time so Python can parse it easily
    std::cout << avg_ms << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}