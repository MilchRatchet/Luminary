#ifndef BVH_H
#define BVH_H

#include <stdint.h>

#include "mesh.h"
#include "primitives.h"

struct compressed_vec3 {
  uint8_t x;
  uint8_t y;
  uint8_t z;
} typedef compressed_vec3;

struct Node2 {
  vec3 left_low;
  vec3 left_high;
  vec3 right_low;
  vec3 right_high;
  vec3 self_low;
  vec3 self_high;
  uint32_t triangle_count;
  uint32_t triangles_address;
  uint32_t child_address;
  float surface_area;
  float sah_cost[7];
  int decision[7];
  int cost_computed;
  uint8_t kind;
} typedef Node2;

struct Node8 {
  vec3 p;
  int8_t ex;
  int8_t ey;
  int8_t ez;
  uint8_t imask;
  int32_t child_node_base_index;
  int32_t triangle_base_index;
  uint8_t meta[8];
  uint8_t low_x[8];
  uint8_t low_y[8];
  uint8_t low_z[8];
  uint8_t high_x[8];
  uint8_t high_y[8];
  uint8_t high_z[8];
} typedef Node8;

Node2* build_bvh_structure(Triangle** triangles_io, unsigned int* triangles_length_io, unsigned int* nodes_length_out);
Node8* collapse_bvh(
  Node2* binary_nodes, const unsigned int binary_nodes_length, Triangle** triangles_io, const int triangles_length,
  unsigned int* nodes_length_out);
void sort_traversal_elements(Node8** nodes_io, const int nodes_length, Triangle** triangles_io, const int triangles_length);

#endif /* BVH_H */
