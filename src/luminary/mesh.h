#ifndef LUMINARY_MESH_H
#define LUMINARY_MESH_H

#include "utils.h"

struct BVH;
struct OptixBVH;
struct LightData;

struct Mesh {
  TriangleGeomData* data;
  Triangle* triangles;
  struct LightData* light_data;
  struct BVH* bvh;
  struct OptixBVH* optix_bvh;
} typedef Mesh;

struct MeshInstance {
  uint32_t mesh_id;
} typedef MeshInstance;

LuminaryResult mesh_create(Mesh* mesh);
LuminaryResult mesh_destroy(Mesh* mesh);

#endif /* LUMINARY_MESH_H */
