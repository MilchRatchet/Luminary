#ifndef LUMINARY_DEVICE_MESH_H
#define LUMINARY_DEVICE_MESH_H

#include "device_utils.h"
#include "optix_bvh.h"

struct Device typedef Device;
struct OpacityMicromap typedef OpacityMicromap;

struct DeviceMesh {
  uint32_t id;
  uint32_t triangle_count;
  DEVICE DeviceTriangleVertex* vertices;
  DEVICE DeviceTriangleTexture* texture_triangles;
  OptixBVH* bvh;
  OpacityMicromap* omm;
} typedef DeviceMesh;

struct Device typedef Device;

LuminaryResult device_mesh_create(DeviceMesh** device_mesh);
DEVICE_CTX_FUNC LuminaryResult device_mesh_set(DeviceMesh* device_mesh, Device* device, const Mesh* mesh);
DEVICE_CTX_FUNC LuminaryResult device_mesh_process(DeviceMesh* device_mesh, Device* device, bool* data_has_changed);
DEVICE_CTX_FUNC LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh);

#endif /* LUMINARY_DEVICE_MESH_H */
