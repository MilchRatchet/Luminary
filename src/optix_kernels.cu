#include "math.cuh"
#include "utils.cuh"
#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory optix_params;

extern "C" __global__ void __raygen__example() {
  const uint3 idx = optixGetLaunchIndex();

  // printf does not work inside optix kernels
  printf("%d %d %d", idx.x, optix_params.width, optix_params.height);

  optix_params.ptrs.frame_output[idx.x] = get_RGBAhalf(255.0f, 0.0f, 0.0f, 0.0f);
}
