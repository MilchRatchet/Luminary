#ifndef LUMINARY_OPTIX_BVH_H
#define LUMINARY_OPTIX_BVH_H

#include "device_light.h"
#include "device_particle.h"
#include "device_utils.h"

struct Device typedef Device;
struct OpacityMicromap typedef OpacityMicromap;
struct MeshInstanceProcessedUpdate typedef MeshInstanceProcessedUpdate;

enum OptixBVHType { OPTIX_BVH_TYPE_DEFAULT = 0, OPTIX_BVH_TYPE_SHADOW = 1, OPTIX_BVH_TYPE_COUNT } typedef OptixBVHType;

struct OptixBVHInstanceCache {
  uint32_t num_instances_allocated;
  uint32_t num_instances;
  DEVICE OptixInstance* instances[OPTIX_BVH_TYPE_COUNT];
} typedef OptixBVHInstanceCache;

LuminaryResult optix_bvh_instance_cache_create(OptixBVHInstanceCache** cache);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_instance_cache_update(
  OptixBVHInstanceCache* cache, Device* device, const ARRAY MeshInstanceProcessedUpdate* instance_updates, const ARRAY DeviceMesh** meshes);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_instance_cache_destroy(OptixBVHInstanceCache** cache);

struct OptixBVH {
  bool fast_trace;
  bool allocated_mask[OPTIX_BVH_TYPE_COUNT];
  OptixTraversableHandle traversable[OPTIX_BVH_TYPE_COUNT];
  DEVICE void* bvh_data[OPTIX_BVH_TYPE_COUNT];
} typedef OptixBVH;

DEVICE_CTX_FUNC LuminaryResult optix_bvh_create(OptixBVH** bvh);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_gas_build(OptixBVH* bvh, Device* device, const DeviceMesh* mesh);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_ias_build(OptixBVH* bvh, Device* device, const OptixBVHInstanceCache* cache);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_light_build(OptixBVH* bvh, Device* device, const DeviceLightTree* tree);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_particles_gas_build(OptixBVH* bvh, Device* device, const DeviceParticlesHandle* particles_handle);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_particles_ias_build(OptixBVH* bvh, Device* device, const DeviceParticlesHandle* particles_handle);
DEVICE_CTX_FUNC LuminaryResult optix_bvh_destroy(OptixBVH** bvh);

#endif /* LUMINARY_OPTIX_BVH_H */
