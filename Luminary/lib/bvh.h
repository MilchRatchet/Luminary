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
  vec3 p;
  int32_t triangles_address;
  int8_t ex;
  int8_t ey;
  int8_t ez;
  int8_t _p;
  compressed_vec3 left_low;
  compressed_vec3 left_high;
  compressed_vec3 right_low;
  compressed_vec3 right_high;
  uint32_t triangle_count;
  uint32_t _p1;
  uint64_t _p2;
} typedef Node;

Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, const int max_depth,
  int* nodes_length_out);

#endif /* BVH_H */
