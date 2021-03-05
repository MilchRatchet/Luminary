#ifndef BVH_H
#define BVH_H

#include <stdint.h>
#include "primitives.h"
#include "mesh.h"

struct compressed_vec3 {
  uint8_t x;
  uint8_t y;
  uint8_t z;
} typedef compressed_vec3;

struct Node {
  int32_t triangles_address;
  uint32_t triangle_count;
  vec3 p;
  int8_t ex;
  int8_t ey;
  int8_t ez;
  compressed_vec3 left_low;
  compressed_vec3 left_high;
  compressed_vec3 right_low;
  compressed_vec3 right_high;
} typedef Node;

Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, const int max_depth,
  int* nodes_length_out);

#endif /* BVH_H */
