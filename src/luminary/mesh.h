#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

struct BVH;
struct OptixBVH;
struct MeshletLightData;

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

/*
 * Subset of mesh in which all triangles have the same material. A meshlet may contain up to 65536 triangles, i.e., 16 bits for addressing.
 */
struct Meshlet {
  struct MeshletLightData* light_data;
  Triangle* triangles;
  bool has_textured_emission;
  float* normalized_emission;
  uint32_t* index_buffer;
  uint32_t triangle_count;
  uint16_t material_id;
} typedef Meshlet;

struct Mesh {
  TriangleGeomData* data;
  Triangle* triangles;
  ARRAY Meshlet* meshlets;
  struct OptixBVH* optix_bvh;
  struct OptixBVH* optix_bvh_shadow;
  struct OptixBVH* optix_bvh_light;
  struct BVH* bvh;
} typedef Mesh;

struct MeshInstance {
  uint32_t mesh_id;
  vec3 offset;
  vec3 scale;
  Quaternion rotation;
  bool deleted; /* On deletion, instances will be marked as deleted and will be overwritten when new instances get added. */
} typedef MeshInstance;

LuminaryResult mesh_create(Mesh** mesh);
LuminaryResult mesh_build_meshlets(Mesh* mesh);
LuminaryResult mesh_destroy(Mesh** mesh);

LuminaryResult mesh_instance_get_default(MeshInstance* instance);

#endif /* LUMINARY_MESH_H */
