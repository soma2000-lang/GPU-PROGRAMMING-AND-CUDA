#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
__global__ void sumKernel(
    const float *vector_pointer,
    float *output,
    const int size)
{
    extern __shared__ float sharedData[];          // shared memory for partial sums
    int ti = threadIdx.x;                          // thread index in the block
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global index in the input array

    // Load data into shared memory
    // So that each thread will have acces to the Data
    sharedData[ti] = (i < size) ? vector_pointer[i] : 0.0f;
    __syncthreads(); // Synchronize all threads in the block

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (ti < stride)
        {
            sharedData[ti] += sharedData[ti + stride];
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Write the partial sum for the block to the output
    if (ti == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

void chunkedPartialSums(const float *array, const int size)
{
    const int ThreadsPerBlock = 1024;                                  // 1024 threads per block . Can be any value whatever :D
    const int Chunks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock; // Calculate number of chunks

    dim3 BlockDim(ThreadsPerBlock); // Set the block size to 256
    dim3 GridDim(Chunks);           // One block for each chunk

    size_t SharedMemory = ThreadsPerBlock * sizeof(float); // Size of shared memory
    size_t SizeBytes = size * sizeof(float);               // Total size of the input array
    size_t SizePartialSums = Chunks * sizeof(float);       // Total size for the partial sums output

    float *DeviceInput, *DevicePartialSums;                            // Create Device Arrays
    cudaMalloc((void **)&DeviceInput, SizeBytes);                      // Allocate the Device array
    cudaMalloc((void **)&DevicePartialSums, SizePartialSums);          // Allocate the PartialSums Array
    cudaMemcpy(DeviceInput, array, SizeBytes, cudaMemcpyHostToDevice); // Copy the data

    sumKernel<<<GridDim, BlockDim, SharedMemory>>>(DeviceInput, DevicePartialSums, size); // Launch the kernel

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Cuda ERROR: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    float *PartialSums = new float[Chunks];
    cudaMemcpy(PartialSums, DevicePartialSums, SizePartialSums, cudaMemcpyDeviceToHost); // Copy the partial sums

    float TotalSum = 0.0f;
    for (int i = 0; i < Chunks; i++)
    {
        TotalSum += PartialSums[i];
    }
    std::cout << "\nTotal sum is: " << TotalSum << std::endl;
    std::cout << "Threads used: " << ThreadsPerBlock << std::endl;
    std::cout << "Chunkes used: " << Chunks << std::endl;
    // Delete Memory
    delete[] PartialSums;
    cudaFree(DeviceInput);
    cudaFree(DevicePartialSums);
}

int main()
{
    const int n = 1 << 28; // 2^28
    float *input = new float[n];

    for (int i = 0; i < n; i++)
    {
        input[i] = 1.0f;
    }

    float cpuResult = 0;
    auto startCPU = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        cpuResult += input[i];
    auto stopCPU = std::chrono::high_resolution_clock::now();


    auto startCUDA = std::chrono::high_resolution_clock::now();
    chunkedPartialSums(input, n);
    cudaDeviceSynchronize();
    auto stopCUDA = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> cpuDuration = stopCPU - startCPU;
    std::chrono::duration<double> cudaDuration = stopCUDA - startCUDA;
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " seconds\n";
    std::cout << "CUDA Execution Time: " << cudaDuration.count() << " seconds\n";


    delete[] input;
}
