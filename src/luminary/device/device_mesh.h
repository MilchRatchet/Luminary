#ifndef LUMINARY_DEVICE_MESH_H
#define LUMINARY_DEVICE_MESH_H

#include "device_utils.h"

struct DeviceMesh {
  ARRAY DeviceTriangle* triangles;
  ARRAY uint32_t* meshlet_triangle_offsets;
  ARRAY uint16_t* meshlet_material_ids;
  // OptixBVH
} typedef DeviceMesh;

LuminaryResult device_mesh_create(DeviceMesh** device_mesh, const Mesh* mesh);
LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh);

#endif /* LUMINARY_DEVICE_MESH_H */
