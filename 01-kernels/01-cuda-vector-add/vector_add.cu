/*
 * CUDA Vector Addition - A Beginner's Guide
 * ==========================================
 * This program adds two large vectors (arrays of numbers) together.
 * It demonstrates GPU parallel computing: instead of the CPU doing one addition
 * at a time, thousands of GPU "threads" work simultaneously, each adding one
 * pair of numbers. We'll compare GPU speed vs CPU speed at the end.
 */

// --- HEADER FILES (like importing tools) ---
// These give us access to pre-written code we need:
#include <cuda_runtime.h>      // CUDA: talk to the GPU (allocate memory, copy data, run kernels)
#include <device_launch_parameters.h>  // CUDA: built-in variables (blockIdx, threadIdx, blockDim)
#include <stdio.h>             // C: print output (printf)
#include <stdlib.h>            // C: allocate/free memory (malloc, free), random numbers (rand)
#include <time.h>              // C: measure elapsed time (clock)

/*
 * GPU KERNEL: vectorAdd
 * ---------------------
 * A "kernel" is a function that runs on the GPU. The __global__ keyword tells
 * the compiler: "this runs on the GPU, not the CPU."
 *
 * Key idea: This ONE function is executed by MANY threads in parallel.
 * Each thread gets its own copy of the variables and computes ONE element.
 *
 * Parameters:
 *   A, B  - Input vectors (read-only, hence "const")
 *   C     - Output vector (we write the sums here)
 *   N     - Number of elements in each vector
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    /*
     * Each thread needs to know: "Which element am I responsible for?"
     * CUDA gives us 3 built-in variables:
     *   blockIdx.x  - Which block am I in? (0, 1, 2, ...)
     *   blockDim.x  - How many threads per block? (e.g., 256)
     *   threadIdx.x - Which thread am I within my block? (0 to 255)
     *
     * Formula: global index = (which block) * (threads per block) + (my position in block)
     * Example: Thread 5 in block 2 with 256 threads/block -> i = 256*2 + 5 = 517
     */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /*
     * Bounds check: Our grid might have MORE threads than elements (e.g., N=100,
     * but we launch 256 threads per block). Threads with i >= N have nothing to do.
     */
    if (i < N) C[i] = A[i] + B[i];
}

/*
 * MAIN FUNCTION - Program entry point
 * -----------------------------------
 * The CPU runs this. We: (1) prepare data, (2) copy to GPU, (3) run kernel,
 * (4) copy result back, (5) compare with CPU timing, (6) free memory.
 */
int main() {
    // --- STEP 1: Allocate memory on the CPU (the "Host") ---
    int N = 100000000;  // 100 million elements - a lot for the CPU, easy for GPU
    size_t size = N * sizeof(float);  // Total bytes: N elements × 4 bytes per float

    // malloc = "memory allocate" - reserves RAM. We need 3 arrays: A, B, and result C.
    // The "h_" prefix is a convention meaning "host" (CPU) memory.
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Fill A and B with random numbers between 0 and 1 (for testing)
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;  // rand() gives 0..RAND_MAX, divide to get 0..1
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // --- STEP 2: Allocate memory on the GPU (the "Device") ---
    // The "d_" prefix means "device" (GPU) memory.
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);  // & means "address of" - cudaMalloc needs where to store the pointer
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy our input data FROM CPU TO GPU (Host -> Device)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- STEP 3: Launch the GPU kernel ---
    // We organize threads into "blocks" of 256. Each block runs on part of the GPU.
    int threadsPerBlock = 256;
    // How many blocks? Enough so (blocks × 256) >= N. The formula rounds up.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Time the GPU execution
    clock_t start = clock();
    // <<<blocksPerGrid, threadsPerBlock>>> is CUDA's special "launch configuration" syntax
    // It means: run this many blocks, each with this many threads (all in parallel!)
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Wait for GPU to finish before we continue (like "join" in threading)
    clock_t end = clock();
    printf("CUDA time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // --- STEP 4: Copy the result back from GPU to CPU ---
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // --- STEP 5: Compare with CPU (sequential) execution ---
    // Same computation, but one element at a time on the CPU. Usually much slower!
    start = clock();
    for (int i = 0; i < N; i++) h_C[i] = h_A[i] + h_B[i];
    end = clock();
    printf("CPU time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // --- STEP 6: Free all allocated memory ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);  // GPU memory
    free(h_A); free(h_B); free(h_C);              // CPU memory
    return 0;
}
