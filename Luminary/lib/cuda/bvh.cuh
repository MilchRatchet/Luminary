#ifndef CU_BVH_H
#define CU_BVH_H

#include "utils.cuh"
#include "math.cuh"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

struct traversal_result {
  unsigned int hit_id;
  float depth;
} typedef traversal_result;

__device__
traversal_result traverse_bvh(const vec3 origin, const vec3 ray) {
  float depth = device_scene.far_clip_distance;

  unsigned int hit_id = 0xffffffff;

  int node_address = 0;
  int node_key = 1;
  int bit_trail = 0;
  int mrpn_address = -1;

  while (node_address != -1) {
      while (device_scene.nodes[node_address].triangles_address == -1) {
          Node node = device_scene.nodes[node_address];

          const float decompression_x = __powf(2.0f, (float)node.ex);
          const float decompression_y = __powf(2.0f, (float)node.ey);
          const float decompression_z = __powf(2.0f, (float)node.ez);

          const vec3 left_high = decompress_vector(node.left_high, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 left_low = decompress_vector(node.left_low, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 right_high = decompress_vector(node.right_high, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 right_low = decompress_vector(node.right_low, node.p, decompression_x, decompression_y, decompression_z);

          const float L = ray_box_intersect(left_low, left_high, origin, ray);
          const float R = ray_box_intersect(right_low, right_high, origin, ray);
          const int R_is_closest = R < L;

          if (L < depth || R < depth) {

              node_key = node_key << 1;
              bit_trail = bit_trail << 1;

              if (L >= depth || R_is_closest) {
                  node_address = 2 * node_address + 2;
                  node_key = node_key ^ 0b1;
              }
              else {
                  node_address = 2 * node_address + 1;
              }

              if (L < depth && R < depth) {
                  bit_trail = bit_trail ^ 0b1;
                  if (R_is_closest) {
                      mrpn_address = node_address - 1;
                  }
                  else {
                      mrpn_address = node_address + 1;
                  }
              }
          } else {
              break;
          }
      }

      if (device_scene.nodes[node_address].triangles_address != -1) {
          Node node = device_scene.nodes[node_address];

          for (unsigned int i = 0; i < node.triangle_count; i++) {
              const float d = triangle_intersection(device_scene.triangles[node.triangles_address + i], origin, ray);

              if (d < depth) {
                  depth = d;
                  hit_id = node.triangles_address + i;
              }
          }
      }

      if (bit_trail == 0) {
          node_address = -1;
      }
      else {
          int num_levels = trailing_zeros(bit_trail);
          bit_trail = (bit_trail >> num_levels) ^ 0b1;
          node_key = (node_key >> num_levels) ^ 0b1;
          if (mrpn_address != -1) {
              node_address = mrpn_address;
              mrpn_address = -1;
          }
          else {
              node_address = node_key - 1;
          }
      }
  }

  traversal_result result;
  result.hit_id = hit_id;
  result.depth = depth;

  return result;
}

#endif /* CU_BVH_H */
