#ifndef LUMINARY_OPTIX_KERNEL_H
#define LUMINARY_OPTIX_KERNEL_H

#include "device_utils.h"
#include "optix_shared.h"

struct Device typedef Device;

#define OPTIX_KERNEL_NUM_GROUPS (2 + OPTIX_KERNEL_FUNCTION_COUNT)

enum OptixKernelType {
  OPTIX_KERNEL_TYPE_RAYTRACE,
  OPTIX_KERNEL_TYPE_SHADING_GEOMETRY_GEO,
  OPTIX_KERNEL_TYPE_SHADING_GEOMETRY_SKY,
  OPTIX_KERNEL_TYPE_SHADING_VOLUME_GEO,
  OPTIX_KERNEL_TYPE_SHADING_VOLUME_SKY,
  OPTIX_KERNEL_TYPE_SHADING_PARTICLES_GEO,
  OPTIX_KERNEL_TYPE_SHADING_PARTICLES_SKY,
  OPTIX_KERNEL_TYPE_COUNT
} typedef OptixKernelType;

struct OptixKernel {
  bool available;
  OptixModule module;
  OptixProgramGroup groups[OPTIX_KERNEL_NUM_GROUPS];
  OptixPipeline pipeline;
  DEVICE char* records;
  OptixShaderBindingTable shaders;
} typedef OptixKernel;

LuminaryResult optix_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type);
LuminaryResult optix_kernel_execute(OptixKernel* kernel, Device* device);
LuminaryResult optix_kernel_destroy(OptixKernel** kernel);

#endif /* LUMINARY_OPTIX_KERNEL_H */
