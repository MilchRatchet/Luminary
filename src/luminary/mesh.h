#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

#define MESH_ID_INVALID (0xFFFFFFFF)

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
  char* name;
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
LuminaryResult mesh_set_name(Mesh* mesh, const char* name);
LuminaryResult mesh_destroy(Mesh** mesh);

LuminaryResult mesh_instance_get_default(MeshInstance* instance);
LuminaryResult mesh_instance_check_for_dirty(const MeshInstance* input, const MeshInstance* old, bool* dirty);

LuminaryResult mesh_instance_from_public_api_instance(MeshInstance* mesh_instance, const LuminaryInstance* instance);
LuminaryResult mesh_instance_to_public_api_instance(LuminaryInstance* instance, const MeshInstance* mesh_instance);

#endif /* LUMINARY_MESH_H */
