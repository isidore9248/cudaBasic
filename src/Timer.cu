#include <stdio.h>
#include "module/GPU_Module.cuh"

__global__ void TimerFunc()
{
    for (int i = 0; i < 100; i++)
    {
        // printf("Hello World from thread %d\n", threadIdx.x + blockIdx.x * blockDim.x);
    }
}

int main()
{
    static float t_sum = 0.0f;

    for (int repeat = 0; repeat < 5; repeat++)
    {
        //* timer start & stop init
        cudaEvent_t start, stop;
        ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
        ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
        cudaEventQuery(start); // 此处不可用错误检测函数

        //* process---start
        TimerFunc<<<2, 4>>>(); // 调用核函数
        //* process---end

        ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
        float elapsed_time;
        ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
        printf("Time = %g ms.\n", elapsed_time);

        //! 通常不算入第一次的时间，因为第一次调用核函数时所消耗的时间比较多
        //! 但后续的调用时间会比较短
        if (repeat > 0)
        {
            t_sum += elapsed_time;
        }

        //* timer start & stop destroy
        ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
        ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    }

    printf("Average time = %g ms.\n", t_sum / 4);
}