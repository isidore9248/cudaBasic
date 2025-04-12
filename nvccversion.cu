#include <stdio.h>

__global__ void hello()
{

    const int OnlyId = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Hello World from thread %d\n", OnlyId);
}

int main()
{
    printf("Hello World from CPU\n");
    dim3 grid(2, 3, 2);
    dim3 block(2, 2, 2);
    hello<<<grid, block>>>();
    cudaDeviceSynchronize(); // Wait for GPU to finish

    return 0;
}