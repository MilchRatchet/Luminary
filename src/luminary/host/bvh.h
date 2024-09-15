#ifndef BVH_H
#define BVH_H

#include <stdint.h>

#include "structs.h"
#include "utils.h"

struct compressed_vec3 {
  uint8_t x;
  uint8_t y;
  uint8_t z;
} typedef compressed_vec3;

enum NodeType { NodeTypeNull = 0, NodeTypeInternal = 1, NodeTypeLeaf = 2 } typedef NodeType;

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
  NodeType type;
} typedef Node2;

void bvh_init(RaytraceInstance* instance);

#endif /* BVH_H */
