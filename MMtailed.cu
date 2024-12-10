#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

#define TILE_SIZE 16  // Tile size for shared memory optimization

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;  
    int ty = threadIdx.y;  

    int row = blockIdx.y * TILE_SIZE + ty;  
    int col = blockIdx.x * TILE_SIZE + tx;  

    float value = 0.0f;


    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < N && tile * TILE_SIZE + tx < N)
            Asub[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            Asub[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < N)
            Bsub[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            value += Asub[ty][i] * Bsub[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

void matrixMultiplyHost(float* A, float* B, float* C, int N) {
    int size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);


    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken " << elapsed.count() << " seconds" << std::endl;


    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 500;  
    int size = N * N * sizeof(float);

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    matrixMultiplyHost(A, B, C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
