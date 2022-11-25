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
#define gameW 1200
#define gameH 900

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

__global__ void asteroidBoundsKernel(float* posX, float* posY, float* w, float* h, float* outPosX, float* outPosY, int numElems) {
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < numElems) {
        outPosX[x] = 99999;
        outPosY[x] = 99999;

        // Check X positions
        if (posX[x] + w[x] < 0)
        {
            outPosX[x] = gameW;
            outPosY[x] = posY[x];
        }
        if (posX[x] - w[x] > gameW)
        {
            outPosX[x] = -w[x];
            outPosY[x] = posY[x];
        }

        // Check Y positions
        if (posY[x] + h[x] < 0)
        {
            outPosX[x] = posX[x];
            outPosY[x] = gameH + h[x];
        }
        if (posY[x] - h[x] > gameH)
        {
            outPosX[x] = posX[x];
            outPosY[x] = -h[x];
        }
    }
}


__global__ void asteroidCheckBulletKernel(float* posX, float* posY, float* w, float* h, float* bull_l, float* bull_r, float* bull_t, float* bull_b, int* out, int numElems) {
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    out[x] = 99999;

    if (x < numElems) {
        for (int i = 0; i < sizeof(bull_l); i++) {
            if (bull_r[i] >= posX[x] && bull_l[i] <= (posX[x] + w[x]) && bull_t[i] >= posY[x] && bull_b[i] <= (posY[x] + h[x]))
            {
                out[x] = i;
            }
        }
    }
}


extern "C" void cuda_asteroidPos(float* theta, float factor, float* outX, float* outY, int numElems) {
    cudaError_t error = cudaSuccess;
    
    std::array<int, 9> cudaInfo = getCudaInfo();

    int blocksPerGrid = (numElems + cudaInfo[3] - 1) / cudaInfo[3];

    asteroidPositionKernel<<< blocksPerGrid, numElems >>> (theta, factor, outX, outY, numElems);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nasteroidPositionKernel() failed to launch error = %d\n", error);
    }
}

extern "C" void cuda_asteroidBounds(float* posX, float* posY, float* w, float* h, float* outPosX, float* outPosY, int numElems) {
    cudaError_t error = cudaSuccess;

    std::array<int, 9> cudaInfo = getCudaInfo();
    int blocksPerGrid = (numElems + cudaInfo[3] - 1) / cudaInfo[3];

    asteroidBoundsKernel <<< blocksPerGrid, numElems >>> (posX, posY, w, h, outPosX, outPosY, numElems);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\asteroidBoundsKernel() failed to launch error = %d\n", error);
    }
}

extern "C" void cuda_asteroidCheckBullet(float* posX, float* posY, float* w, float* h, float* bull_l, float* bull_r, float* bull_t, float* bull_b, int* out, int numElems) {
    cudaError_t error = cudaSuccess;

    std::array<int, 9> cudaInfo = getCudaInfo();
    int blocksPerGrid = (numElems + cudaInfo[3] - 1) / cudaInfo[3];

    asteroidCheckBulletKernel <<< blocksPerGrid, numElems >>> (posX, posY, w, h, bull_l, bull_r, bull_t, bull_b, out, numElems);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\asteroidBoundsKernel() failed to launch error = %d\n", error);
    }
}

#endif