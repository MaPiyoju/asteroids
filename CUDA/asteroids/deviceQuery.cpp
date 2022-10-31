#include "./Headers/deviceQuery.hpp"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>
#include<array>


#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Get Cuda Info
////////////////////////////////////////////////////////////////////////////////
std::array<int, 9> getCudaInfo() {
  
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  } 

  std::array<int, 9> gpuInfo;

  int driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    // Assign properties

    // CUDA Cores ==> 0
    gpuInfo[0] = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    
    // Multiprocesor ==> 1
    gpuInfo[1] = deviceProp.multiProcessorCount;

    // Threads per multiprocessor ==> 2
    gpuInfo[2] = deviceProp.maxThreadsPerMultiProcessor;

    // Threads per block ==> 3
    gpuInfo[3] = deviceProp.maxThreadsPerBlock;

    // Total global memory ==> 4
    gpuInfo[4] = ((unsigned long long)deviceProp.totalGlobalMem);
    //static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),

    // Constant memory ==> 5
    gpuInfo[5] = deviceProp.totalConstMem;

    // Shared memory per block ==> 6
    gpuInfo[6] = deviceProp.sharedMemPerBlock;

    // Shared memory per multiproccessor ==> 7
    gpuInfo[7] = deviceProp.sharedMemPerMultiprocessor;

    // Warp size ==> 8
    gpuInfo[8] = deviceProp.warpSize;

    return gpuInfo;
}
