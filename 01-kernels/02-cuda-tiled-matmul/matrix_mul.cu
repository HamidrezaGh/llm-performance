/*
 * CUDA Matrix Multiplication - Naive vs Tiled (Shared Memory)
 * ============================================================
 * This program multiplies two square matrices:
 *   C = A x B
 *
 * It implements two GPU kernels:
 * 1) naiveMatMul: every thread reads directly from global memory
 * 2) tiledMatMul: threads cooperate by caching matrix tiles in shared memory
 *
 * Why this matters:
 * - Matrix multiply is memory-intensive.
 * - Shared memory tiling reduces repeated global-memory reads.
 * - This is a core optimization pattern used in high-performance GPU kernels.
 */

// --- HEADER FILES ---
// cuda_runtime.h: CUDA API (malloc/copy/kernel launch/timing events)
// stdio.h:        printf
// stdlib.h:       malloc, free, rand
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Problem size: matrices are N x N
#define N 2048
// Tile size: each thread block is TILE_SIZE x TILE_SIZE threads
#define TILE_SIZE 32

/*
 * KERNEL 1: Naive matrix multiplication
 * -------------------------------------
 * Mapping:
 * - One thread computes exactly one output element C[row, col].
 *
 * Formula:
 * - C[row, col] = sum_k A[row, k] * B[k, col]
 *
 * Characteristics:
 * - Simple and correct baseline
 * - Slow for large N because each thread repeatedly reads from global memory
 */
__global__ void naiveMatMul(float *A, float *B, float *C, int n) {
    // Convert 2D block/thread coordinates into global row/col in C.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard out-of-range threads (important when grid is rounded up).
    if (row < n && col < n) {
        float sum = 0.0f;

        // Dot product of row from A and column from B.
        for (int k = 0; k < n; k++) sum += A[row * n + k] * B[k * n + col];

        // Store one final output element.
        C[row * n + col] = sum;
    }
}

/*
 * KERNEL 2: Tiled matrix multiplication (shared memory)
 * -----------------------------------------------------
 * Idea:
 * - Split the output matrix into TILE_SIZE x TILE_SIZE blocks.
 * - Each thread block computes one output tile.
 * - For each phase t, threads load one tile of A and one tile of B into
 *   fast on-chip shared memory, then reuse those values for TILE_SIZE MACs.
 *
 * Benefits:
 * - Much lower global-memory traffic
 * - Better data reuse
 * - Usually much faster than naive kernel for large matrices
 */
__global__ void tiledMatMul(float *A, float *B, float *C, int n) {
    // Shared-memory tiles visible to all threads in the same block.
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Thread coordinates inside the tile (0..TILE_SIZE-1).
    int tx = threadIdx.x, ty = threadIdx.y;
    // Top-left global coordinate of this block's output tile.
    int bx = blockIdx.x * TILE_SIZE, by = blockIdx.y * TILE_SIZE;
    float sum = 0.0f;

    // Number of tile phases needed to complete one dot product.
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load: each thread loads one A element and one B element
        // from global memory into shared memory.
        sA[ty][tx] = (bx + tx < n && by + ty < n) ? A[(by + ty) * n + (t * TILE_SIZE + tx)] : 0.0f;
        sB[ty][tx] = (bx + tx < n && by + ty < n) ? B[(t * TILE_SIZE + ty) * n + (bx + tx)] : 0.0f;

        // Ensure the entire tile is loaded before any thread starts computing.
        __syncthreads();

        // Multiply-accumulate using fast shared-memory values.
        for (int k = 0; k < TILE_SIZE; k++) sum += sA[ty][k] * sB[k][tx];

        // Ensure all threads are done before shared memory is overwritten
        // by the next tile phase.
        __syncthreads();
    }

    // Write final output element for this thread.
    if (bx + tx < n && by + ty < n) C[(by + ty) * n + (bx + tx)] = sum;
}

/*
 * MAIN FUNCTION
 * -------------
 * Workflow:
 * 1) Allocate and initialize host data
 * 2) Allocate device memory and copy inputs
 * 3) Launch naive kernel and time it
 * 4) Launch tiled kernel and time it
 * 5) Print speedup
 * 6) Free all resources
 */
int main() {
    // Host pointers (CPU memory).
    float *h_A, *h_B, *h_C;
    // Device pointers (GPU memory).
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // --- 1) Allocate host memory ---
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize A and B with random values in [0, 1].
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // --- 2) Allocate device memory + copy inputs to GPU ---
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch configuration:
    // - block: 32x32 threads
    // - grid: enough blocks to cover all N x N output elements
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA events provide GPU-side timing in milliseconds.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 3) Time naive kernel ---
    cudaEventRecord(start);
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naiveTime;
    cudaEventElapsedTime(&naiveTime, start, stop);
    printf("Naive CUDA time: %.2f ms\n", naiveTime);

    // --- 4) Time tiled kernel ---
    cudaEventRecord(start);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiledTime;
    cudaEventElapsedTime(&tiledTime, start, stop);
    printf("Tiled CUDA time: %.2f ms\n", tiledTime);

    // --- 5) Compare runtime ---
    printf("Tiled speedup vs Naive: %.1fx\n", naiveTime / tiledTime);

    // --- 6) Cleanup ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
