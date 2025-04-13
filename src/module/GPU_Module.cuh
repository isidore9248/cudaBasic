#pragma once
#include <stdio.h>

void GPU_Init(void)
{
    int GPUcount = 0;
    cudaError_t GPU_error = cudaGetDeviceCount(&GPUcount); // Get the number of GPUs available
    if (GPU_error != cudaSuccess || GPUcount == 0)         // Check for errors in getting the device count
    {
        printf("Error getting device count: %s\n", cudaGetErrorString(GPU_error));
        return;
    }

    GPU_error = cudaSetDevice(0); // Set the device to the first GPU
    if (GPU_error != cudaSuccess)
    {
        printf("Error setting device: %s\n", cudaGetErrorString(GPU_error));
        return;
    }

    printf("GPU Get OK\nGPU count: %d\n", GPUcount);
}

/// <summary>
/// 运行时API错误检查函数
/// </summary>
/// <param name="error_code">CUDA error code</param>
/// <param name="filename">File name where the error occurred</param>
/// <param name="lineNumber">Line number where the error occurred</param>
/// <returns>Returns the error code if there is an error, otherwise returns cudaSuccess</returns>
///* using example: ErrorCheck(cudaMemcpy(...), __FILE__, __LINE__);
cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}