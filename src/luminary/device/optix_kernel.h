#ifndef LUMINARY_OPTIX_KERNEL_H
#define LUMINARY_OPTIX_KERNEL_H

#include "device_utils.h"

struct Device typedef Device;

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
  OptixProgramGroup groups[OPTIXRT_NUM_GROUPS];
  OptixPipeline pipeline;
  DEVICE char* records;
  OptixShaderBindingTable shaders;
  DEVICE DeviceConstantMemory* params;
} typedef OptixKernel;

LuminaryResult optix_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type);
LuminaryResult optix_kernel_update_params(OptixKernel* kernel);
LuminaryResult optix_kernel_update_sample_id(OptixKernel* kernel);
LuminaryResult optix_kernel_execute(OptixKernel* kernel);
LuminaryResult optix_kernel_destroy(OptixKernel** kernel);

#endif /* LUMINARY_OPTIX_KERNEL_H */
