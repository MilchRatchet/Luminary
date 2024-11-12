#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

// Traditional description through vertex buffer and index buffer which is required for OptiX RT.
// Both vertex buffer and index buffer have a stride of 16 bytes for each triplet
// but the counts only indicate the number of actual data entries.
struct TriangleGeomData {
  float* vertex_buffer;
  uint32_t* index_buffer;
  uint32_t vertex_count;
  uint32_t index_count;
  uint32_t triangle_count;
} typedef TriangleGeomData;

struct Mesh {
  uint32_t id;
  TriangleGeomData data;
  Triangle* triangles;
} typedef Mesh;

struct MeshInstance {
  uint32_t id;
  uint32_t mesh_id;
  vec3 translation;
  vec3 scale;
  Quaternion rotation;
  bool active; /* On deletion, instances will be marked as deleted and will be overwritten when new instances get added. */
} typedef MeshInstance;

LuminaryResult mesh_create(Mesh** mesh);
LuminaryResult mesh_destroy(Mesh** mesh);

LuminaryResult mesh_instance_get_default(MeshInstance* instance);

#endif /* LUMINARY_MESH_H */
