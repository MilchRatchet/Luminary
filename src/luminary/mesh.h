#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

struct BVH;
struct OptixBVH;
struct LightData;

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

struct Meshlet {
  struct LightData* light_data;
  Triangle* triangles;
  uint32_t* index_buffer;
  uint16_t triangle_count;
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

struct MeshletInstance {
  uint32_t meshlet_id;
} typedef MeshletInstance;

struct MeshInstance {
  uint32_t mesh_id;
  ARRAY MeshletInstance* meshlet_instances;
} typedef MeshInstance;

LuminaryResult mesh_create(Mesh** mesh);
LuminaryResult mesh_build_meshlets(Mesh* mesh);
LuminaryResult mesh_destroy(Mesh** mesh);

#endif /* LUMINARY_MESH_H */
