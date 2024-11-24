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

struct KernelArgsSkyComputeTransmittanceLUT {
  float4* dst_low;
  float4* dst_high;
} typedef KernelArgsSkyComputeTransmittanceLUT;

struct KernelArgsSkyComputeMultiscatteringLUT {
  DeviceTextureObject transmission_low_tex;
  DeviceTextureObject transmission_high_tex;
  float4* dst_low;
  float4* dst_high;
} typedef KernelArgsSkyComputeMultiscatteringLUT;

#endif /* LUMINARY_KERNEL_ARGS_H */
