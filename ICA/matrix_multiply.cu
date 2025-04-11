#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <cmath>

#define N 4096  // Matrix size (N x N)
#define TILE_SIZE 32  // Tile size for shared memory optimization
#define EPSILON 1e-4  // Error tolerance for comparison

// matrix multiplication without tiling
__global__ void matMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// matrix multiplication with tiling
__global__ void matMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    for (int i = 0; i < n / TILE_SIZE; i++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (i * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

void initializeMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; i++)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

bool compareMatrices(float *A, float *B, int n, float epsilon) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: A = %f, B = %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

float timeKernel(void (*kernel)(float *, float *, float *, int), float *A, float *B, float *C, int n, dim3 gridDim, dim3 blockDim) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *A, *B, *C, *C_tiled;
    float *d_A, *d_B, *d_C, *d_C_tiled;

    A = (float *)malloc(bytes);
    B = (float *)malloc(bytes);
    C = (float *)malloc(bytes);
    C_tiled = (float *)malloc(bytes);

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_tiled, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);

    float tiledTime = timeKernel(matMulTiled, d_A, d_B, d_C, N, gridDim, blockDim);
    float Time = timeKernel(matMul, d_A, d_B, d_C_tiled, N, gridDim, blockDim);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix Multiplication Time: %.2f ms\n", Time);
    printf("Tiled Matrix Multiplication Time: %.2f ms\n", tiledTime);
    printf("Speedup: %.2fx\n", Time / tiledTime);

    // Compare the results
    if (compareMatrices(C, C_tiled, N, EPSILON)) {
        printf("SUCCESS: The matrices match!\n");
    } else {
        printf("ERROR: The matrices do not match!\n");
    }
    printf("%f\n", C[1]);
    printf("%f\n", C_tiled[1]);
    free(A); free(B); free(C); free(C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_tiled);

    return 0;
}
