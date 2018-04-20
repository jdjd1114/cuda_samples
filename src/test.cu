// CUDA kernel -- Generate dot product

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "error_util.h"

using namespace std;
const int THREADS_NUM = 1024;

__global__ static void dot_product(float * a, float * b, float * c, int array_size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
     
    if (tid < array_size)
    {
        extern __shared__ float c_tmp[];

        c_tmp[tid] = a[tid] * b[tid];
        __syncthreads();

        size_t length = array_size;
        size_t offset = (length - 1)/2 + 1;

        while (length >= 2)
        {
            if (tid + offset < length)
            {
                c_tmp[tid] = c_tmp[tid] + c_tmp[tid + offset];
                //__syncthreads();
            }

            length = (length - 1)/2 + 1;
            offset = (offset - 1)/2 + 1;
            __syncthreads(); 
        }

        c[0] = c_tmp[0];

//        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    int length = THREADS_NUM;
    float * a = new float [length];
    float * b = new float [length];
    float c = 0;

    for (size_t i = 0; i < length; i++)
    {
        a[i] = 1.5;
        b[i] = 2.5;
    }

    cudaSetDevice(1);

    // Create CUDA event
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
    // prepare data on GPU
    float * gpu_a;
    float * gpu_b;
    float * gpu_c;
    checkCudaErrors(cudaMalloc((void **)&gpu_a, sizeof(float) * length));
    checkCudaErrors(cudaMalloc((void **)&gpu_b, sizeof(float) * length));
    checkCudaErrors(cudaMalloc((void **)&gpu_c, sizeof(float) * length));
    checkCudaErrors(cudaMemcpy(gpu_a, a, sizeof(float) * length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_b, b, sizeof(float) * length, cudaMemcpyHostToDevice));

    // kernel function
    size_t blockNum = 1;
    size_t threadsNum = THREADS_NUM;
    dot_product<<<blockNum, threadsNum, sizeof(float) * length>>>(gpu_a, gpu_b, gpu_c, length);
    
    // Copy data back
    cudaMemcpy(&c, gpu_c, sizeof(float), cudaMemcpyDeviceToHost);

    // output
    cout<<"Sum = "<<c<<endl;

    checkCudaErrors(cudaFree(gpu_a));
    checkCudaErrors(cudaFree(gpu_b));
    checkCudaErrors(cudaFree(gpu_c));
    delete [] a;
    delete [] b;

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize( stop ));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf( "Time to generate dot product: %.3f ms\n", elapsedTime );

    return 0;
}
