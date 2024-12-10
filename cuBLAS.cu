#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <stdio.h>


void matrixMultiplyCuBLAS(const float* A, const float* B, float* C, int N) {
    float alpha = 1.0f;  // Scalar multiplier for AB
    float beta = 0.0f;   // Scalar multiplier for C

    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  
        N, N, N,                  
        &alpha,                   
        d_A, N,                   
        d_B, N,                   
        &beta,                    
        d_C, N                   
    );

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 100;  

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;  
        B[i] = 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCuBLAS(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken " << elapsed.count() << " seconds" << std::endl;


    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
