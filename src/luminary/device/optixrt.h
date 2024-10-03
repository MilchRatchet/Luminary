#ifndef OPTIXRT_H
#define OPTIXRT_H

#include "device_memory.h"
#include "device_utils.h"
#include "mesh.h"

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
  OptixShaderBindingTable shaders;
  DEVICE DeviceConstantMemory* params;
} typedef OptixKernel;

LuminaryResult optixrt_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type);
LuminaryResult optixrt_kernel_update_params(OptixKernel* kernel);
LuminaryResult optixrt_kernel_update_sample_id(OptixKernel* kernel);
LuminaryResult optixrt_kernel_execute(OptixKernel* kernel);
LuminaryResult optixrt_kernel_destroy(OptixKernel** kernel);

struct OptixBVH {
  OptixTraversableHandle traversable;
  DEVICE void* bvh_data;
} typedef OptixBVH;

enum OptixRTBVHType { OPTIX_RT_BVH_TYPE_DEFAULT = 0, OPTIX_RT_BVH_TYPE_SHADOW = 1 } typedef OptixRTBVHType;

LuminaryResult optixrt_bvh_create(OptixBVH** bvh, Device* device, const Mesh* mesh, OptixRTBVHType type);
LuminaryResult optixrt_bvh_destroy(OptixBVH** bvh);

#endif /* OPTIXRT_H */
