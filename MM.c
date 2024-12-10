#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)(rand() % 100) / 10.0; 
        }
    }
}

void multiply_matrices(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

int main() {
    srand(time(NULL));

    int sizes[] = {100, 200, 300, 400, 500}; // Array of sizes
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int m = sizes[s];
        int n = sizes[s];
        int p = sizes[s];

        float *A = (float *)malloc(m * n * sizeof(float));
        float *B = (float *)malloc(n * p * sizeof(float));
        float *C = (float *)malloc(m * p * sizeof(float));

        // Initialize matrices A and B
        initialize_matrix(A, m, n);
        initialize_matrix(B, n, p);

        clock_t start = clock();

        multiply_matrices(A, B, C, m, n, p);

        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Matrix size: %dx%d, Time taken: %.6f seconds\n", m, n, time_taken);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
