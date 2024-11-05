#ifndef LUMINARY_OPTIX_BVH_H
#define LUMINARY_OPTIX_BVH_H

#include "device_utils.h"

struct Device typedef Device;

struct OptixBVHInstanceCache {
  Device* device;
  uint32_t num_instances_allocated;
  uint32_t num_instances;
  DEVICE OptixInstance* instances;
} typedef OptixBVHInstanceCache;

DEVICE_CTX_FUNC LuminaryResult optix_bvh_instance_cache_create(OptixBVHInstanceCache** cache, Device* device);
DEVICE_CTX_FUNC LuminaryResult
  optix_bvh_instance_cache_update(OptixBVHInstanceCache* cache, const ARRAY MeshInstanceUpdate* instance_updates);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_instance_cache_destroy(OptixBVHInstanceCache** cache);

struct OptixBVH {
  bool allocated;
  bool fast_trace;
  OptixTraversableHandle traversable;
  DEVICE void* bvh_data;
} typedef OptixBVH;

enum OptixBVHType { OPTIX_BVH_TYPE_DEFAULT = 0, OPTIX_BVH_TYPE_SHADOW = 1 } typedef OptixRTBVHType;

DEVICE_CTX_FUNC LuminaryResult optix_bvh_create(OptixBVH** bvh);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_gas_build(OptixBVH* bvh, Device* device, const Mesh* mesh, OptixRTBVHType type);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_ias_build(OptixBVH* bvh, Device* device);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_destroy(OptixBVH** bvh);

#endif /* LUMINARY_OPTIX_BVH_H */
