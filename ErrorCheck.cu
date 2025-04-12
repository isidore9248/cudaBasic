#include <stdio.h>
#include "./module/GPU_Module.cuh"

__global__ void hello()
{

    const int OnlyId = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Hello World from thread %d\n", OnlyId);
}

int main()
{
    // GPU_Init();
    float *fpHost_A = (float *)malloc(sizeof(float) * 512);
    if (fpHost_A != NULL)
    {
        memset(fpHost_A, 0, 0);
    }

    float *fpDevice_A = NULL;
    cudaMalloc((float **)&fpDevice_A, sizeof(float) * 512);
    if (fpDevice_A != NULL)
    {
        ErrorCheck(cudaMemcpy((float **)&fpHost_A, (float **)&fpDevice_A, sizeof(float) * 512, cudaMemcpyDeviceToHost),
                   __FILE__, __LINE__);
    }

    hello<<<1, 3>>>();
    cudaDeviceSynchronize();
    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
}