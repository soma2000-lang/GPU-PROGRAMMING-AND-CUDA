#include <iostream>
#include <cuda_runtime.h>

// TODO : create template

struct Tensor {
    float *data;
    int rows;
    int cols;

    __device__ int size() const {
        return rows * cols;
    }

    __device__ float get(const int row, const int col = 0) {
        return data[row * cols + col];
    }

    __device__ void set(const int row, const int col = 0, const float value = 0) {
        data[row * cols + col] = value;
    }
};



__global__ void layerNorm(Tensor* A) {
    extern __shared__ float sharedMemory[];  

    float* mean = sharedMemory;
    float* variance = sharedMemory + A->rows;
    float* invstdev = sharedMemory + 2 * A->rows;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < A->rows) {
        mean[tid] = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            mean[tid] += A->get(tid, j);
        }
        mean[tid] /= A->cols;
    }
    __syncthreads();

    if (tid < A->rows) {
        variance[tid] = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            float diff = A->get(tid, j) - mean[tid];
            variance[tid] += diff * diff;
        }
        variance[tid] /= A->cols;
        invstdev[tid] = rsqrtf(variance[tid] + 1e-5f);
    }
    __syncthreads();

    if (tid < A->rows) {
        for (int j = 0; j < A->cols; j++) {
            float normalized = (A->get(tid, j) - mean[tid]) * invstdev[tid];
            A->set(tid, j, normalized);
        }
    }
}

int main() {
    const int rows = 1;
    const int cols = 3;
    const int BLOCKSIZE = 16;
    const int tensorSize = rows * cols;
    const int size = rows * cols * sizeof(float);
    dim3 blockDim(16*16);
    dim3 gridDim((rows + BLOCKSIZE - 1) / BLOCKSIZE);
    size_t sharedMemorySize = 3 * rows * sizeof(float);
    // Shared memory for mean, variance, invstdev
    // we need 3 different arrays of rows * size(float)

    // host data 
    float h_data[tensorSize] = {
        5.0f, 1.5f, 2.0f,
    };


    // now lets allocate the tensor on the device
    // we create our tensor but we use the device data as the data
    // and we only copy the block
    
    float* d_data; // pointer to device data
    cudaMalloc(&d_data, size);   // alocate the space
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice); // copy the h_data to the d_data on the gpu
     
    Tensor h_tensor; // now we create the host tenosr on gpu
    h_tensor.data = d_data; // the pointer to the GPU !
    h_tensor.rows = rows; // copy rows
    h_tensor.cols = cols; // copy cols


    Tensor* d_tensor; // now create a device tensor
    cudaMalloc(&d_tensor, sizeof(Tensor)); // we allocate the space
    cudaMemcpy(d_tensor, &h_tensor, sizeof(Tensor), cudaMemcpyHostToDevice);
    // now we copy on the d_tensor from the h_tenosr


    layerNorm<<<gridDim ,blockDim, sharedMemorySize>>>(d_tensor);
    // we launch our kernel now 

    cudaDeviceSynchronize(); // syncing
    cudaMemcpy(h_data, d_data,size, cudaMemcpyDeviceToHost);
    // now copy back the struct
    
    std::cout << "Normalized Matrix:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_tensor);

    return 0;
}