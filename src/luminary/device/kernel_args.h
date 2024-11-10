#ifndef LUMINARY_KERNEL_ARGS_H
#define LUMINARY_KERNEL_ARGS_H

#include "device_utils.h"

struct KernelArgsCloudComputeShapeNoise {
  uint32_t dim;
  uint8_t* tex;
} typedef KernelArgsCloudComputeShapeNoise;

struct KernelArgsCloudComputeDetailNoise {
  uint32_t dim;
  uint8_t* tex;
} typedef KernelArgsCloudComputeDetailNoise;

struct KernelArgsCloudComputeWeatherNoise {
  uint32_t dim;
  float seed;
  uint8_t* tex;
} typedef KernelArgsCloudComputeWeatherNoise;

#endif /* LUMINARY_KERNEL_ARGS_H */
