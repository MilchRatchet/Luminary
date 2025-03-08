#ifndef LUMINARY_OPTIX_KERNEL_H
#define LUMINARY_OPTIX_KERNEL_H

#include "device_utils.h"
#include "optix_shared.h"

struct Device typedef Device;

#define OPTIX_KERNEL_NUM_GROUPS (2 + OPTIX_KERNEL_FUNCTION_COUNT)

enum OptixKernelType {
  OPTIX_KERNEL_TYPE_RAYTRACE,
  OPTIX_KERNEL_TYPE_SHADING_GEOMETRY,
  OPTIX_KERNEL_TYPE_SHADING_VOLUME,
  OPTIX_KERNEL_TYPE_SHADING_PARTICLES,
  OPTIX_KERNEL_TYPE_COUNT
} typedef OptixKernelType;

struct OptixKernel {
  OptixModule module;
  OptixProgramGroup groups[OPTIX_KERNEL_NUM_GROUPS];
  OptixPipeline pipeline;
  DEVICE char* records;
  OptixShaderBindingTable shaders;
  bool use_new_scheduler;
} typedef OptixKernel;

LuminaryResult optix_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type);
LuminaryResult optix_kernel_execute(OptixKernel* kernel, Device* device);
LuminaryResult optix_kernel_destroy(OptixKernel** kernel);

#endif /* LUMINARY_OPTIX_KERNEL_H */
