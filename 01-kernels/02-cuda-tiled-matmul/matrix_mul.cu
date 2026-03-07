#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048          // Matrix size (feel free to change)
#define TILE_SIZE 32

// Naive kernel (no shared memory)
__global__ void naiveMatMul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled kernel with shared memory
__global__ void tiledMatMul(float* A, float* B, float* C, int n) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (bx + tx < n && by + ty < n)
            sA[ty][tx] = A[(by + ty) * n + (t * TILE_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;

        if (bx + tx < n && by + ty < n)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * n + (bx + tx)];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (bx + tx < n && by + ty < n)
        C[(by + ty) * n + (bx + tx)] = sum;
}

int main() {
    // This main is just for compilation test
    printf("Tiled Matrix Multiplication kernel ready.\n");
    return 0;
}