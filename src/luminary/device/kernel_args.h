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

struct KernelArgsGenerateFinalImage {
  RGBF* src;
  RGBF color_correction;
  AGXCustomParams agx_params;
} typedef KernelArgsGenerateFinalImage;

struct KernelArgsConvertRGBFToXRGB8 {
  XRGB8* dst;
  uint32_t width;
  uint32_t height;
  LuminaryFilter filter;
} typedef KernelArgsConvertRGBFToXRGB8;

#endif /* LUMINARY_KERNEL_ARGS_H */
