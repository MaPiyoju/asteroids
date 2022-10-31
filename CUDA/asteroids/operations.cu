//#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>
//#include <cmath>
//#include<array>
//#include <device_launch_parameters.h>
//
//////////////////////////////////////////////////////////////////////////////////
//// Constants
//#define PI 3.1415926536f
//
//__global__ void asteroidPositionKernel(float* outputData, int width, int height,
//    float theta, float factor) {
//    // calculate normalized texture coordinates
//    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    float u = (float)x - (float)width / 2;
//    float v = (float)y - (float)height / 2;
//    float tu = u * cosf(theta) - v * sinf(theta);
//    float tv = v * cosf(theta) + u * sinf(theta);
//
//    float asteroidAngle = (PI / 180) * (theta); // Convert asteroid's angle to radians
//    float newX = std::cos(asteroidAngle)* factor;
//    float newY = std::sin(asteroidAngle)* factor;
//
//    // read from texture and write to global memory
//    float newPos[2] = {
//        newX, newY
//    };
//
//    outputData = newPos;
//}
//
//extern "C" void cuda_asteroidPos(float* outputData, int width, int height,
//    float theta, float factor) {
//    cudaError_t error = cudaSuccess;
//
//    dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
//    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
//
//    asteroidPositionKernel <<< Dg, Db >>> (std::array<int, 2>*outputData, width, height,
//        theta, factor);
//
//    error = cudaGetLastError();
//
//    if (error != cudaSuccess) {
//        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
//    }
//}
