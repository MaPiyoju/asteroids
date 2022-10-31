#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cmath>
#include<array>
#include <device_launch_parameters.h>
#include "Headers/deviceQuery.hpp"

////////////////////////////////////////////////////////////////////////////////
// Constants
#define PI 3.1415926536f

#ifndef _OPERATIONS_CU_
#define _OPERATIONS_CU_

__global__ void asteroidPositionKernel(float* theta, float factor, float* outX, float* outY, int numElems) {
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x< numElems) {
        float asteroidAngle = (PI / 180) * (theta[x]); // Convert asteroid's angle to radians
        float newX = std::cos(asteroidAngle) * factor;
        float newY = std::sin(asteroidAngle) * factor;

        outX[x] = newX;
        outY[x] = newY;
    }
}

extern "C" void cuda_asteroidPos(float* theta, float factor, float* outX, float* outY, int numElems) {
    cudaError_t error = cudaSuccess;

    std::array<int, 9> cudaInfo = getCudaInfo();
    int blocksPerGrid = (1000 + cudaInfo[3] - 1) / cudaInfo[3];

    asteroidPositionKernel<<< blocksPerGrid, cudaInfo[3] >>> (theta, factor, outX, outY, numElems);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nasteroidPositionKernel() failed to launch error = %d\n", error);
    }
}

__global__ void asteroidBoundsKernel(float* theta, float factor, float* outX, float* outY) {
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float asteroidAngle = (PI / 180) * (theta[x]); // Convert asteroid's angle to radians
    float newX = std::cos(asteroidAngle) * factor;
    float newY = std::sin(asteroidAngle) * factor;

    outX[x] = newX;
    outX[y] = newY;
}

extern "C" void cuda_asteroidBounds(float* theta, float factor, float* outX, float* outY) {
    cudaError_t error = cudaSuccess;

    std::array<int, 9> cudaInfo = getCudaInfo();
    int blocksPerGrid = (1000 + cudaInfo[3] - 1) / cudaInfo[3];

    asteroidBoundsKernel << < blocksPerGrid, cudaInfo[3] >> > (theta, factor, outX, outY);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nasteroidPositionKernel() failed to launch error = %d\n", error);
    }
}

#endif