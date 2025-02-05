#include <iostream>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
int M = 10;
int N = 10;

dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
dim3 blockDim(32, 32, 1); // 32 * 32 * 1

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B, float beta, float *C)
{
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    
    if (x < M && y<N){
        float temp = 0.0;
        for(int i = 0;i<K;++i){
            temp += A[x*K+i] * B[i*N+y];
        }

        C[x*N+y] = alpha*temp + beta * C[x*N+y];
    }
}