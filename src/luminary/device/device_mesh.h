#ifndef LUMINARY_DEVICE_MESH_H
#define LUMINARY_DEVICE_MESH_H

#include "device_utils.h"
#include "optix_bvh.h"

struct DeviceMesh {
  DEVICE DeviceTriangle* triangles;
  OptixBVH* bvh;
} typedef DeviceMesh;

struct Device typedef Device;

DEVICE_CTX_FUNC LuminaryResult device_mesh_create(Device* device, DeviceMesh** device_mesh, const Mesh* mesh);
DEVICE_CTX_FUNC LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh);

#endif /* LUMINARY_DEVICE_MESH_H */
