#ifndef BVH_H
#define BVH_H

#include <stdint.h>
#include "primitives.h"
#include "mesh.h"

struct Node {
  int32_t triangles_address;
  uint32_t triangle_count;
  vec3 left_low;
  vec3 left_high;
  vec3 right_low;
  vec3 right_high;
} typedef Node;

Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, const int max_depth,
  int* nodes_length_out);

#endif /* BVH_H */
