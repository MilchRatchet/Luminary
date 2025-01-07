#ifndef LUMINARY_KERNEL_ARGS_H
#define LUMINARY_KERNEL_ARGS_H

#include "device_utils.h"

struct KernelArgsBSDFGenerateSSLUT {
  uint16_t* dst;
} typedef KernelArgsBSDFGenerateSSLUT;

struct KernelArgsBSDFGenerateGlossyLUT {
  uint16_t* dst;
  const uint16_t* src_energy_ss;
} typedef KernelArgsBSDFGenerateGlossyLUT;

struct KernelArgsBSDFGenerateDielectricLUT {
  uint16_t* dst;
  uint16_t* dst_inv;
} typedef KernelArgsBSDFGenerateDielectricLUT;

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
  uint32_t ld;
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

struct KernelArgsParticleGenerate {
  uint32_t count;
  uint32_t seed;
  float size;
  float size_variation;
  float4* vertex_buffer;
  Quad* quads;
} typedef KernelArgsParticleGenerate;

struct KernelArgsGenerateFinalImage {
  RGBF* src;
  RGBF color_correction;
  AGXCustomParams agx_params;
} typedef KernelArgsGenerateFinalImage;

struct KernelArgsConvertRGBFToARGB8 {
  ARGB8* dst;
  uint32_t width;
  uint32_t height;
  LuminaryFilter filter;
} typedef KernelArgsConvertRGBFToARGB8;

#endif /* LUMINARY_KERNEL_ARGS_H */
