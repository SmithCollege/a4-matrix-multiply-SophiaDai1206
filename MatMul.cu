#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

// CUDA kernel for matrix multiplication (no tiling)
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrixMultiplyHost(float* A, float* B, float* C, int N) {
    int size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16); 
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

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
