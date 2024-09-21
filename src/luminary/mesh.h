#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

struct BVH;
struct OptixBVH;
struct LightTree;

struct Mesh {
  float* vertex_buffer;
  uint32_t vertex_count;
  uint32_t* index_buffer;
  Triangle* triangles;
  uint32_t triangle_count;
  TriangleLight* lights;
  uint32_t light_count;
  struct BVH* bvh;
  struct OptixBVH* optix_bvh;
  struct LightTree* light_tree;
} typedef Mesh;

struct MeshInstance {
  uint32_t mesh_id;
} typedef MeshInstance;

LuminaryResult mesh_create(Mesh** mesh);
LuminaryResult mesh_destroy(Mesh** mesh);

#endif /* LUMINARY_MESH_H */
