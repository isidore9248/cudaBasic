///* 以二维数组为例，常用的线程模型是二维线程块和一维线程网格。
///* int iy=threadIdx.y+blockIdx.y*blockDim.y;
///* int ix=threadIdx.x+blockIdx.x*blockDim.x;
///! 不同于之前的matrixmanipulate.cu，这里是用二维网格对应一维数组

#include <stdio.h>
#include "./module/GPU_Module.cuh"

__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
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
        ErrorCheck(cudaMemcpy(pDevice_add1, pHost_add1, MatrixBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(pDevice_add2, pHost_add2, MatrixBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(pDevice_sum, pHost_sum, MatrixBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }
    else
    {
        free(pHost_add1);free(pHost_add2);free(pHost_sum);
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

    // （3）初始化主机内存数据
    dim3 block(4, 4, 1); // 每个线程块的大小为4*4=16
    dim3 grid((matrix_x_num+block.x-1)/block.x, (matrix_y_num+block.y-1)/block.y);
    addMatrix<<<grid, block>>>(pDevice_add1, pDevice_add2, pDevice_sum, matrix_x_num, matrix_y_num);
    ErrorCheck(cudaMemcpy(pHost_sum, pDevice_sum, MatrixBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__); 
    //ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);


    for(int i=0;i<matrix_x_num;i++)
    {
        printf("id=%d, matrix_A=%d, matrix_B=%d, result=%d\n", i + 1,pHost_add1[i], pHost_add2[i], pHost_sum[i]);
    }

    free(pHost_add1);free(pHost_add2);free(pHost_sum);
    cudaFree(pDevice_add1);cudaFree(pDevice_add2);cudaFree(pDevice_sum);
    printf("Success!\n");
    return 0;
}