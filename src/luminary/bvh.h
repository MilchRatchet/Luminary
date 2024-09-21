#ifndef LUMINARY_BVH_H
#define LUMINARY_BVH_H

#include "mesh.h"
#include "utils.h"

struct BVH {
  void* data;
  size_t size;
  TraversalTriangle* traversal_triangles;
  uint32_t triangle_count;
} typedef BVH;

LuminaryResult bvh_create(BVH** bvh, const Mesh* mesh);
LuminaryResult bvh_destroy(BVH** bvh);

#endif /* LUMINARY_BVH_H */
