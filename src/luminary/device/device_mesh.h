#ifndef LUMINARY_DEVICE_MESH_H
#define LUMINARY_DEVICE_MESH_H

#include "device_utils.h"
#include "optix_bvh.h"

struct Device typedef Device;
struct OpacityMicromap typedef OpacityMicromap;

struct DeviceMesh {
  const Mesh* mesh;  // Non owning
  DEVICE DeviceTriangle* triangles;
  bool bvh_is_dirty;
  OptixBVH* bvh;
} typedef DeviceMesh;

struct Device typedef Device;

DEVICE_CTX_FUNC LuminaryResult device_mesh_create(DeviceMesh** device_mesh, Device* device, const Mesh* mesh);
DEVICE_CTX_FUNC LuminaryResult device_mesh_build_structures(DeviceMesh* device_mesh, OpacityMicromap* omm, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh);

#endif /* LUMINARY_DEVICE_MESH_H */
