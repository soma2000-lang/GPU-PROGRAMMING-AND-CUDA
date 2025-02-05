#include <iostream>
#include <cuda_runtime.h>

__global__ void selfAttentionKernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int N, // seq len
    int d  // embedding dim
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // (rows) sequence position
    const int k = blockIdx.y * blockDim.y + threadIdx.y; // (cols) embedding dimension
    // we have a 2D Matrix:
    // we have [seq_len x embedding]
    //      Output Matrix O (N=3, d=4)
    //       k=0   k=1   k=2   k=3
    // i=0 [                    ]  → Token 0's full embedding
    // i=1 [                    ]  → Token 1's full embedding
    // i=2 [                    ]  → Token 2's full embedding
    // so this is our O matrix final
    // now lets dive deeper

    if (i >= N || k >= d)
        return;
    // check so that we dont go out of bounds

    float max_val = -INFINITY; // prepare for softmax online
    float sum_exp = 0.0f;
    float acc = 0.0f;

    for (int j = 0; j < N; ++j)
    {
        // here we iterate on all sequence positions
        float score = 0.0f;

        for (int l = 0; l < d; ++l)
        {
            // dot product calculated here
            // so we do Q[i*d + l] we iterate on the row i
            // K -> iterated ons the
            score += Q[i * d + l] * K[j * d + l];
        }
        // score
        score /= sqrtf(d);

        float old_max = max_val;
        max_val = fmaxf(max_val, score);
        float exp_term = expf(score - max_val);

        sum_exp = sum_exp * expf(old_max - max_val) + exp_term;
        acc = acc * expf(old_max - max_val) + exp_term * V[j * d + k];
    }

    O[i * d + k] = acc / sum_exp;
}

void selfAttention(const float *Q,
                   const float *K,
                   const float *V,
                   float *O,
                   const int N,
                   const int d)
{
    float *dQ, *dK, *dV, *dO;
    size_t sizeMatrix = N * d * sizeof(float);
    dim3 blockDim(16, 16);
    dim3 gridDim(ceilf((float)N / blockDim.x), ceilf((float)d / blockDim.y));

    cudaMalloc((void **)&dQ, sizeMatrix);
    cudaMalloc((void **)&dK, sizeMatrix);
    cudaMalloc((void **)&dV, sizeMatrix);
    cudaMalloc((void **)&dO, sizeMatrix);
    cudaMemcpy(dQ, Q, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeMatrix, cudaMemcpyHostToDevice);

    selfAttentionKernel<<<gridDim, blockDim>>>(dQ, dK, dV, dO, N, d);

    cudaMemcpy(O, dO, sizeMatrix, cudaMemcpyDeviceToHost);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}

void randominit(float *A, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        A[i] = sin(i);
    }
}

int main()
{
    int N = 10;
    int d = 20;

    float *Q = new float[N * d];
    float *K = new float[N * d];
    float *V = new float[N * d];
    float *O = new float[N * d];

    randominit(Q, N * d);
    randominit(K, N * d);
    randominit(V, N * d);

    selfAttention(Q, K, V, O, N, d);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            std::cout << O[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    free(Q);
    free(K);
    free(V);
    free(O);

    return 0;
}