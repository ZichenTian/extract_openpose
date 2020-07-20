#include <iostream>
#include <array>
#include "utils.h"

void nmsGpuFloatWrapper(
    float* dst, int* kernel, float* src, const float threshold, 
    const std::array<int, 4>& dstSize, const std::array<int, 4>& srcSize, const Point<float>& offset);

template <typename T>
void initSrcData(int n, int c, int h, int w, T* data) {
    const size_t size = n * c * h * w;      // 输入数据
    for(size_t i = 0; i < size; i++) {
        data[i] = i; 
    }
}

template <typename T>
void showDstData(int n, int c, int h, int w, const T* data) {
    const size_t size = n * c * h * w;      // 输出数据
    for(size_t i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void nmsWrapper(void) {

    /*********************** 参数设置 *******************************/
    const int batch = 1;            // 这些参数是我从srcBase.h里面扒的
    const int srcChannels = 57;     // 具体意思我不太清楚
    const int srcHeight = 368;
    const int srcWidth = 656;
    std::array<int, 4> srcSize = {batch, srcChannels, srcHeight, srcWidth};

    const int dstChannels = 18;
    const int dstHeight = 128;
    const int dstWidth = 3;
    std::array<int, 4> dstSize = {batch, dstChannels, dstHeight, dstWidth};

    Point<T> offset;              // 这个参数不知道该咋设
    offset.x = 0;
    offset.y = 0;

    const T threshold = 0.5;        // 这个参数是我乱设的，不清楚行不行

    const size_t srcLength = batch * srcChannels * srcHeight * srcWidth;
    const size_t kernelLength = batch * srcChannels * srcHeight * srcWidth; // same as srcSize
    const size_t dstLength = batch * dstChannels * dstHeight * dstWidth;

    /**************************** 存储空间分配 ***************************/

    T* srcDataCpu = new T[srcLength];
    initSrcData<T>(batch, srcChannels, srcHeight, srcWidth, srcDataCpu);

    T* srcDataCuda;
    int* kernelDataCuda;
    T* dstDataCuda;
    cudaMalloc((void**)&srcDataCuda, srcLength * sizeof(T));
    cudaMalloc((void**)&kernelDataCuda, kernelLength * sizeof(int));
    cudaMalloc((void**)&dstDataCuda, dstLength * sizeof(T));

    T* dstDataCpu = new T[dstLength];
    int* kernelDataCpu = nullptr;

    /*************************** NMS运算 ********************************/

    cudaMemcpy(srcDataCuda, srcDataCpu, srcLength * sizeof(T), cudaMemcpyHostToDevice);     // cpu copy to gpu

    nmsGpuFloatWrapper(dstDataCpu, kernelDataCuda, srcDataCpu, threshold, dstSize, srcSize, offset);       // compute
    cudaDeviceSynchronize();                                                                        // wait for compute finished

    cudaMemcpy(dstDataCpu, dstDataCuda, dstLength * sizeof(T), cudaMemcpyDeviceToHost);     // copy back to cpu

    showDstData(batch, dstChannels, dstHeight, dstWidth, dstDataCpu);

    /******************************* 释放空间 **********************************/

    cudaFree(srcDataCuda);
    cudaFree(kernelDataCuda);
    cudaFree(dstDataCuda);

    delete[] srcDataCpu;
    delete[] dstDataCpu;
}

int main(int argc, char* argv[]) {
    nmsWrapper<float>();
    return 0;
}
