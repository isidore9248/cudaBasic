#include <stdio.h>
#include "./module/GPU_Module.cuh"

__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    if (id >= N) // Check if the thread index is within bounds
    {
        return;
    }
    C[id] = add(A[id], B[id]);
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(void)
{
    // 1、设置GPU设备
    GPU_Init(); // 设置GPU设备

    // 2、分配主机内存和设备内存，并初始化
    int iElemCount = 512;                             // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float); // 字节数

    // （1）分配主机内存，并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount); // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        if (fpHost_A != NULL)
        {
            free(fpHost_A);
        }
        if (fpHost_B != NULL)
        {
            free(fpHost_B);
        }
        if (fpHost_C != NULL)
        {
            free(fpHost_C);
        }
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    // （2）分配设备内存，并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float **)&fpDevice_A, stBytesCount);
    cudaMalloc((float **)&fpDevice_B, stBytesCount);
    cudaMalloc((float **)&fpDevice_C, stBytesCount);
    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount); // 设备内存初始化为0
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        if (fpDevice_A != NULL)
        {
            cudaFree(fpDevice_A);
        }
        if (fpDevice_B != NULL)
        {
            cudaFree(fpDevice_B);
        }
        if (fpDevice_C != NULL)
        {
            cudaFree(fpDevice_C);
        }
        printf("fail to allocate memory\n");
        exit(-1);
    }

    // 3、初始化主机中数据
    srand(666); // 设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    // 4、数据从主机复制到设备
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);

    // 5、调用核函数在设备中进行计算
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / 32); // 17

    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount); // 调用核函数
    // cudaDeviceSynchronize();

    // 6、将计算得到的数据从设备传给主机
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    for (int i = 130; i < 140; i++) // 打印
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i + 1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 7、释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}
