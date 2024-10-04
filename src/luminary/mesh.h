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
  TriangleLight* triangle_lights;
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
