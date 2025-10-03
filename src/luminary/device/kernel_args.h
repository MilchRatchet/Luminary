#ifndef LUMINARY_KERNEL_ARGS_H
#define LUMINARY_KERNEL_ARGS_H

#include "device_utils.h"

struct KernelArgsBufferAdd {
  float* dst;
  const float* src;
  uint32_t base_offset;
  uint32_t num_elements;
} typedef KernelArgsBufferAdd;

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

struct KernelArgsSkyComputeHDRI {
  float4* dst_color;
  float* dst_shadow;
  uint32_t dim;
  uint32_t ld_color;
  uint32_t ld_shadow;
  vec3 origin;
  uint32_t sample_count;
} typedef KernelArgsSkyComputeHDRI;

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

struct KernelArgsLightComputeIntensity {
  const uint32_t* mesh_ids;
  const uint32_t* triangle_ids;
  uint32_t lights_count;
  float* dst_intensities;
} typedef KernelArgsLightComputeIntensity;

struct KernelArgsLightGridGenerate {
  uint32_t count;
  vec3 bounds_min;
  vec3 bounds_span;
  uint32_t allocated_entries;
  float importance_threshold;
  MinHeapEntry* min_heap_data;
  uint8_t* dst_cache_point_num_entries;
} typedef KernelArgsLightGridGenerate;

struct KernelArgsGenerateFinalImage {
  RGBF* src;
  RGBF color_correction;
  AGXCustomParams agx_params;
  uint8_t undersampling;
} typedef KernelArgsGenerateFinalImage;

struct KernelArgsConvertRGBFToARGB8 {
  ARGB8* dst;
  uint32_t width;
  uint32_t height;
  LuminaryFilter filter;
  uint8_t undersampling;
} typedef KernelArgsConvertRGBFToARGB8;

struct KernelArgsCameraPostImageDownsample {
  const RGBF* src;
  uint32_t sw;
  uint32_t sh;
  RGBF* dst;
  uint32_t tw;
  uint32_t th;
} typedef KernelArgsCameraPostImageDownsample;

struct KernelArgsCameraPostImageDownsampleThreshold {
  const RGBF* src;
  uint32_t sw;
  uint32_t sh;
  RGBF* dst;
  uint32_t tw;
  uint32_t th;
  float threshold;
} typedef KernelArgsCameraPostImageDownsampleThreshold;

struct KernelArgsCameraPostImageUpsample {
  const RGBF* src;
  uint32_t sw;
  uint32_t sh;
  RGBF* dst;
  const RGBF* base;
  uint32_t tw;
  uint32_t th;
  float sa;
  float sb;
} typedef KernelArgsCameraPostImageUpsample;

struct KernelArgsCameraPostLensFlareGhosts {
  const RGBF* src;
  uint32_t sw;
  uint32_t sh;
  RGBF* dst;
  uint32_t tw;
  uint32_t th;
} typedef KernelArgsCameraPostLensFlareGhosts;

struct KernelArgsCameraPostLensFlareHalo {
  const RGBF* src;
  uint32_t sw;
  uint32_t sh;
  RGBF* dst;
  uint32_t tw;
  uint32_t th;
} typedef KernelArgsCameraPostLensFlareHalo;

struct KernelArgsOMMLevel0Format4 {
  uint32_t mesh_id;
  uint32_t triangle_count;
  uint8_t max_num_levels;
  uint8_t* dst;
  uint8_t* level_record;
  uint32_t* offset_record;
  uint32_t* dst_tri_work;
  uint32_t* work_counter;
} typedef KernelArgsOMMLevel0Format4;

struct KernelArgsOMMRefineFormat4 {
  uint32_t mesh_id;
  uint32_t triangle_count;
  uint8_t max_num_levels;
  uint8_t* dst;
  const uint8_t* src;
  uint8_t* level_record;
  uint32_t* offset_record;
  const uint32_t dst_level;
  const uint32_t* src_tri_work;
  uint32_t* dst_tri_work;
  uint32_t* work_counter;
} typedef KernelArgsOMMRefineFormat4;

struct KernelArgsOMMGatherArrayFormat4 {
  uint32_t triangle_count;
  uint8_t* dst;
  const uint8_t* src[OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL];
  const uint8_t* level_record;
  const uint32_t* offset_record;
  const OptixOpacityMicromapDesc* desc;
} typedef KernelArgsOMMGatherArrayFormat4;

struct KernelArgsMipmapGenerateLevel3DRGBA8 {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
  uint16_t depth;
} typedef KernelArgsMipmapGenerateLevel3DRGBA8;

struct KernelArgsMipmapGenerateLevel2DRGBA8 {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
} typedef KernelArgsMipmapGenerateLevel2DRGBA8;

struct KernelArgsMipmapGenerateLevel3DRGBA16 {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
  uint16_t depth;
} typedef KernelArgsMipmapGenerateLevel3DRGBA16;

struct KernelArgsMipmapGenerateLevel2DRGBA16 {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
} typedef KernelArgsMipmapGenerateLevel2DRGBA16;

struct KernelArgsMipmapGenerateLevel3DRGBAF {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
  uint16_t depth;
} typedef KernelArgsMipmapGenerateLevel3DRGBAF;

struct KernelArgsMipmapGenerateLevel2DRGBAF {
  CUtexObject src;
  CUsurfObject dst;
  uint16_t width;
  uint16_t height;
} typedef KernelArgsMipmapGenerateLevel2DRGBAF;

#endif /* LUMINARY_KERNEL_ARGS_H */
