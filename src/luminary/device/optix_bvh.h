#ifndef LUMINARY_OPTIX_BVH_H
#define LUMINARY_OPTIX_BVH_H

#include "device_utils.h"

struct Device typedef Device;

struct OptixBVH {
  OptixTraversableHandle traversable;
  DEVICE void* bvh_data;
} typedef OptixBVH;

enum OptixBVHType { OPTIX_BVH_TYPE_DEFAULT = 0, OPTIX_BVH_TYPE_SHADOW = 1 } typedef OptixRTBVHType;

LuminaryResult optix_bvh_create(OptixBVH** bvh, Device* device, const Mesh* mesh, OptixRTBVHType type);
LuminaryResult optix_bvh_destroy(OptixBVH** bvh);

#endif /* LUMINARY_OPTIX_BVH_H */
