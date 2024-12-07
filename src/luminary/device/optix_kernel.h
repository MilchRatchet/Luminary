#ifndef LUMINARY_OPTIX_KERNEL_H
#define LUMINARY_OPTIX_KERNEL_H

#include "device_utils.h"

struct Device typedef Device;

#define OPTIX_KERNEL_FUNCTION_COUNT 4
#define OPTIX_KERNEL_NUM_GROUPS (2 + OPTIX_KERNEL_FUNCTION_COUNT)

enum OptixKernelType {
  OPTIX_KERNEL_TYPE_RAYTRACE_GEOMETRY      = 0,
  OPTIX_KERNEL_TYPE_RAYTRACE_PARTICLES     = 1,
  OPTIX_KERNEL_TYPE_SHADING_GEOMETRY       = 2,
  OPTIX_KERNEL_TYPE_SHADING_VOLUME         = 3,
  OPTIX_KERNEL_TYPE_SHADING_PARTICLES      = 4,
  OPTIX_KERNEL_TYPE_SHADING_VOLUME_BRIDGES = 5,

  OPTIX_KERNEL_TYPE_COUNT
} typedef OptixKernelType;

struct OptixKernel {
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
