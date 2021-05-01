#ifndef CU_BVH_H
#define CU_BVH_H

#include "utils.cuh"
#include "math.cuh"
#include "minmax.cuh"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

struct traversal_result {
  unsigned int hit_id;
  float depth;
} typedef traversal_result;

__device__
bool bvh_ray_box_intersect(const vec3 low, const vec3 high, const vec3 inv_ray, const vec3 pso, const float depth, float& out_dist)
{
    vec3 lo;
    lo.x = low.x * inv_ray.x - pso.x;
    lo.y = low.y * inv_ray.y - pso.y;
    lo.z = low.z * inv_ray.z - pso.z;

    vec3 hi;
    hi.x = high.x * inv_ray.x - pso.x;
    hi.y = high.y * inv_ray.y - pso.y;
    hi.z = high.z * inv_ray.z - pso.z;

	const float slab_min = max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, eps);
	const float slab_max = min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, depth);

	out_dist = slabMin;

	return slabMin <= slabMax;
}

__device__
vec3 decompress_vector(const compressed_vec3 vector, const vec3 p, const float ex, const float ey, const float ez) {
    vec3 result;

    result.x = p.x + ex * (float)vector.x;
    result.y = p.y + ey * (float)vector.y;
    result.z = p.z + ez * (float)vector.z;

    return result;
}

__device__
traversal_result traverse_bvh(const vec3 origin, const vec3 ray, const Node* nodes, const Triangle* triangles) {
  float depth = device_scene.far_clip_distance;

  unsigned int hit_id = 0xffffffff;

  vec3 inv_ray;
  inv_ray.x = 1.0f / (fabsf(ray.x) > eps ? ray.x : copysignf(eps, ray.x));
  inv_ray.y = 1.0f / (fabsf(ray.y) > eps ? ray.y : copysignf(eps, ray.y));
  inv_ray.z = 1.0f / (fabsf(ray.z) > eps ? ray.z : copysignf(eps, ray.z));

  vec3 pso;
  pso.x = origin.x * inv_ray.x;
  pso.y = origin.y * inv_ray.y;
  pso.z = origin.z * inv_ray.z;

  int node_address = 0;
  int node_key = 1;
  int bit_trail = 0;
  int mrpn_address = -1;

  Node node;

  while (node_address != -1) {
      while (true) {
          node = nodes[node_address];

          if (node.triangles_address != -1) break;

          const float decompression_x = exp2f((float)node.ex);
          const float decompression_y = exp2f((float)node.ey);
          const float decompression_z = exp2f((float)node.ez);

          const vec3 left_high = decompress_vector(node.left_high, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 left_low = decompress_vector(node.left_low, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 right_high = decompress_vector(node.right_high, node.p, decompression_x, decompression_y, decompression_z);
          const vec3 right_low = decompress_vector(node.right_low, node.p, decompression_x, decompression_y, decompression_z);

          float L,R;
          const bool L_hit = bvh_ray_box_intersect(left_low, left_high, inv_ray, pso, depth, L);
          const bool R_hit = bvh_ray_box_intersect(right_low, right_high, inv_ray, pso, depth, R);
          const int R_is_closest = (R_hit) && (R < L);

          if (L_hit || R_hit) {
              node_key = node_key << 1;
              bit_trail = bit_trail << 1;

              if (!L_hit || R_is_closest) {
                  node_address = 2 * node_address + 2;
                  node_key = node_key ^ 0b1;
              }
              else {
                  node_address = 2 * node_address + 1;
              }

              if (L_hit && R_hit) {
                  bit_trail = bit_trail ^ 0b1;
                  if (R_is_closest) {
                      mrpn_address = node_address - 1;
                  }
                  else {
                      mrpn_address = node_address + 1;
                  }
              }
          } else {
            if (bit_trail == 0) {
                node_address = 0;
                break;
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
      }

      if (node.triangles_address != -1) {
          for (unsigned int i = 0; i < node.triangle_count; i++) {
              const float d = triangle_intersection(triangles[node.triangles_address + i], origin, ray);

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
