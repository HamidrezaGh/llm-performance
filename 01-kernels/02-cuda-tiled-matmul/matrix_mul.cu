#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

__global__ void tiledMatMul(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (bx + tx < N && by + ty < N)
            sA[ty][tx] = A[(by + ty) * N + (t * TILE_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;

        if (bx + tx < N && by + ty < N)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * N + (bx + tx)];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (bx + tx < N && by + ty < N)
        C[(by + ty) * N + (bx + tx)] = sum;
}

int main() {
    // Basic test setup (we'll expand in benchmark.py)
    printf("Tiled Matrix Multiplication kernel compiled successfully.\n");
    return 0;
}