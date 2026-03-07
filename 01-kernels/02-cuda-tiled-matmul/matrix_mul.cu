#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2048
#define TILE_SIZE 32

// Naive kernel
__global__ void naiveMatMul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[row*n + k] * B[k*n + col];
        C[row*n + col] = sum;
    }
}

// Tiled kernel (shared memory)
__global__ void tiledMatMul(float* A, float* B, float* C, int n) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE, by = blockIdx.y * TILE_SIZE;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        sA[ty][tx] = (bx + tx < n && by + ty < n) ? A[(by + ty)*n + (t*TILE_SIZE + tx)] : 0.0f;
        sB[ty][tx] = (bx + tx < n && by + ty < n) ? B[(t*TILE_SIZE + ty)*n + (bx + tx)] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (bx + tx < n && by + ty < n) C[(by + ty)*n + (bx + tx)] = sum;
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    // === TIMING: Naive ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float naiveTime; cudaEventElapsedTime(&naiveTime, start, stop);
    printf("Naive CUDA time: %.2f ms\n", naiveTime);

    // === TIMING: Tiled ===
    cudaEventRecord(start);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float tiledTime; cudaEventElapsedTime(&tiledTime, start, stop);
    printf("Tiled CUDA time: %.2f ms\n", tiledTime);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    printf("Tiled speedup vs Naive: %.1fx\n", naiveTime / tiledTime);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}