#ifndef OPTIXRT_H
#define OPTIXRT_H

#include "device.h"
#include "device_memory.h"
#include "device_utils.h"
#include "mesh.h"

struct OptixKernel {
  OptixPipeline pipeline;
  OptixShaderBindingTable shaders;
  DEVICE DeviceConstantMemory* params;
} typedef OptixKernel;

struct OptixBVH {
  size_t bvh_mem_size;
  OptixTraversableHandle traversable;
  DEVICE void* bvh_data;
} typedef OptixBVH;

enum OptixRTBVHType { OPTIX_RT_BVH_TYPE_DEFAULT = 0, OPTIX_RT_BVH_TYPE_SHADOW = 1 } typedef OptixRTBVHType;

LuminaryResult optixrt_compile_kernel(
  const OptixDeviceContext optix_ctx, const char* kernels_name, OptixKernel* kernel, RendererSettings settings);
LuminaryResult optixrt_build_bvh(
  OptixDeviceContext optix_ctx, OptixBVH* bvh, const Mesh* mesh, const OptixBuildInputDisplacementMicromap* dmm,
  const OptixBuildInputOpacityMicromap* omm, const OptixRTBVHType type);
LuminaryResult optixrt_init(Device* device, RendererSettings settings);
LuminaryResult optixrt_update_params(OptixKernel kernel);
LuminaryResult optixrt_execute(OptixKernel kernel);

#endif /* OPTIXRT_H */
