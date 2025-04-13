///* 以二维数组为例，常用的线程模型是二维线程块和一维线程网格。
///* int iy=threadIdx.y+blockIdx.y*blockDim.y;
///* iy需要循环执行一行的运算
///! 不同于之前的matrixmanipulate.cu，这里是用一维网格对应一维数组


#include <stdio.h>
#include "./module/GPU_Module.cuh"

__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
        
    }
}

int main()
{
    GPU_Init();

    int matrix_x_num=16, matrix_y_num=8,matrix_total_num=matrix_x_num*matrix_y_num;
    size_t MatrixBytes = matrix_x_num * matrix_y_num * sizeof(int); // 字节数
    // （1）分配主机内存，并初始化
    //*每一个指针都指向一个二维数组，字节数都是 MatrixBytes
    int *pHost_add1, *pHost_add2, *pHost_sum;  
    pHost_add1 = (int *)malloc(MatrixBytes);
    pHost_add2 = (int *)malloc(MatrixBytes);
    pHost_sum = (int *)malloc(MatrixBytes);
    if(pHost_add1!=NULL && pHost_add2!=NULL && pHost_sum!=NULL)
    {
        for (int i = 0; i < matrix_total_num; i++)
            {
                pHost_add1[i] = i;
                pHost_add2[i] = i + 1;
            }
        memset(pHost_sum, 0, MatrixBytes);
    }
    else
    {
        if(pHost_add1!=NULL)
        {
            free(pHost_add1);
        }
        if(pHost_add2!=NULL)
        {
            free(pHost_add2);
        }
        if(pHost_sum!=NULL)
        {
            free(pHost_sum);
        }
        printf("Host memory allocation failed\n");
        exit(0);
    }

    // （2）分配设备内存
    int *pDevice_add1, *pDevice_add2, *pDevice_sum;
    cudaMalloc((void **)&pDevice_add1, MatrixBytes);
    cudaMalloc((void **)&pDevice_add2, MatrixBytes);
    cudaMalloc((void **)&pDevice_sum, MatrixBytes);
    if(pDevice_add1!=NULL && pDevice_add2!=NULL && pDevice_sum!=NULL)
    {
        printf("Device memory allocation success\n");
    }
    else
    {
        if(pDevice_add1!=NULL)
        {
            cudaFree(pDevice_add1);
        }
        if(pDevice_add2!=NULL)
        {
            cudaFree(pDevice_add2);
        }
        if(pDevice_sum!=NULL)
        {
            cudaFree(pDevice_sum);
        }
        printf("Device memory allocation failed\n");
        exit(0);
    }
    // （3）将主机内存数据拷贝到设备内存
    ErrorCheck(cudaMemcpy(pDevice_add1, pHost_add1, MatrixBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(pDevice_add2, pHost_add2, MatrixBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemset(pDevice_sum, 0, MatrixBytes), __FILE__, __LINE__);

    // （4）定义线程块和网格的维度
    dim3 block(4, 1, 1);
    dim3 grid((matrix_x_num+block.x-1)/block.x, 1,1);
    addMatrix<<<grid, block>>>(pDevice_add1, pDevice_add2, pDevice_sum, matrix_x_num, matrix_y_num);

    // (5) 将设备内存计算结果拷贝回主机内存
    ErrorCheck(cudaMemcpy(pHost_sum, pDevice_sum, MatrixBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__); 

    // (6) 打印部分结果进行验证
    //printf("Printing first 10 results for verification:\n");
    // Ensure loop doesn't go out of bounds if matrix_total_num < 10
    int print_count = (matrix_total_num < 10) ? matrix_total_num : 10; 
    for (int i = 0; i < print_count; i++)
    {
        printf("idx=%d, add1=%d, add2=%d, sum=%d\n", i, pHost_add1[i], pHost_add2[i], pHost_sum[i]);
    }

    // (7) 释放主机内存
    free(pHost_add1);
    free(pHost_add2);
    free(pHost_sum);

    // (8) 释放设备内存
    ErrorCheck(cudaFree(pDevice_add1), __FILE__, __LINE__); 
    ErrorCheck(cudaFree(pDevice_add2), __FILE__, __LINE__); // Corrected variable name
    ErrorCheck(cudaFree(pDevice_sum), __FILE__, __LINE__); 

    // (9) 重置设备
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__); 

}
