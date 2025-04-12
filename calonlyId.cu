#include <stdio.h>

__global__ void hello()
{

    //* 一维网格唯一线程ID 
    const int OnlyId = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Hello World from thread %d\n", OnlyId);
}

__global__ void hello_2Dim()
{

    //* 二维网格唯一线程ID 
    const int OnlyblockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int OnlythreadId = threadIdx.x + threadIdx.y * blockDim.x;
    const int OnlyId = OnlyblockId * blockDim.x * blockDim.y + OnlythreadId;

    printf("Hello World from thread %d\n", OnlyId);
}

__global__ void hello_3Dim()
{

    //* 三维网格唯一线程ID 
    const int OnlyblockId = blockIdx.x +
                            gridDim.x * blockIdx.y +
                            gridDim.x * gridDim.y * blockIdx.z;
    const int OnlythreadId = threadIdx.z * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x;
    const int OnlyId = OnlyblockId * blockDim.x * blockDim.y * blockDim.z +
                       OnlythreadId;

    printf("Hello World from thread %d\n", OnlyId);
}

int main()
{
    dim3 grid(2, 3, 2);
    dim3 block(2, 2, 2);
    // block  thread
    // hello<<<4, 4>>>();
    // hello_2Dim<<<grid, block>>>();
    hello_3Dim<<<grid, block>>>();

    cudaDeviceSynchronize(); // Wait for GPU to finish

    return 0;
}