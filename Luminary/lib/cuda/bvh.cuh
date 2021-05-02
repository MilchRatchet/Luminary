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
int bvh_ray_box_intersect(const vec3 low, const vec3 high, const vec3 inv_ray, const vec3 pso, const float depth, float& out_dist)
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

	out_dist = slab_min;

	return slab_min <= slab_max;
}

__device__
float bvh_triangle_intersection(const float4* triangles, const vec3 origin, const vec3 ray) {
    const float4 v1 = __ldg(triangles);
    const float4 v2 = __ldg(triangles + 1);
    const float4 v3 = __ldg(triangles + 2);

    vec3 vertex;
    vertex.x = v1.x;
    vertex.y = v1.y;
    vertex.z = v1.z;

    vec3 edge1;
    edge1.x = v2.x;
    edge1.y = v2.y;
    edge1.z = v2.z;

    vec3 edge2;
    edge2.x = v3.x;
    edge2.y = v3.y;
    edge2.z = v3.z;

    const vec3 h = cross_product(ray, edge2);
    const float a = dot_product(edge1, h);

    if (__builtin_expect(a > -0.00000001f && a < 0.00000001f, 0)) return FLT_MAX;

    const float f = 1.0f / a;
    const vec3 s = vec_diff(origin, vertex);
    const float u = f * dot_product(s, h);

    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    const vec3 q = cross_product(s, edge1);
    const float v = f * dot_product(ray, q);

    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    const float t = f * dot_product(edge2, q);

    if (t > -eps) {
        return t;
    } else {
        return FLT_MAX;
    }
}

__device__
vec3 bvh_decompress_vector(const unsigned char x, const unsigned char y, const unsigned char z, const float4 p, const float ex, const float ey, const float ez) {
    vec3 result;

    result.x = p.x + ex * (float)x;
    result.y = p.y + ey * (float)y;
    result.z = p.z + ez * (float)z;

    return result;
}

__device__
unsigned char get_8bit(const unsigned int input, const unsigned int bitshift) {
    return (input >> bitshift) & 0x000000FF;
}

__device__
traversal_result traverse_bvh(const vec3 origin, const vec3 ray, const Node* nodes, const Traversal_Triangle* triangles) {
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

  while (node_address != -1) {
      while (true) {
          const float4 p = __ldg((float4*)((int*)(nodes + node_address)));
          const uint4 data = __ldg((uint4*)(((int*)(nodes + node_address)) + 4));

          if (!signbit(p.w)) break;

          const float decompression_x = exp2f((float)((char)get_8bit(data.x, 0)));
          const float decompression_y = exp2f((float)((char)get_8bit(data.x, 8)));
          const float decompression_z = exp2f((float)((char)get_8bit(data.x, 16)));

          const vec3 left_low = bvh_decompress_vector(get_8bit(data.y, 0), get_8bit(data.y, 8), get_8bit(data.y, 16), p, decompression_x, decompression_y, decompression_z);
          const vec3 left_high = bvh_decompress_vector(get_8bit(data.y, 24), get_8bit(data.z, 0), get_8bit(data.z, 8), p, decompression_x, decompression_y, decompression_z);
          const vec3 right_low = bvh_decompress_vector(get_8bit(data.z, 16), get_8bit(data.z, 24), get_8bit(data.w, 0), p, decompression_x, decompression_y, decompression_z);
          const vec3 right_high = bvh_decompress_vector(get_8bit(data.w, 8), get_8bit(data.w, 16), get_8bit(data.w, 24), p, decompression_x, decompression_y, decompression_z);

          float L,R;
          const int L_hit = bvh_ray_box_intersect(left_low, left_high, inv_ray, pso, depth, L);
          const int R_hit = bvh_ray_box_intersect(right_low, right_high, inv_ray, pso, depth, R);

          if (__builtin_expect(L_hit || R_hit, 1)) {
              node_key = node_key << 1;
              bit_trail = bit_trail << 1;
              const int R_is_closest = (R_hit) && (R < L);
              node_address *= 2;

              if (!L_hit || R_is_closest) {
                  node_address += 2;
                  node_key = node_key ^ 0b1;
              }
              else {
                  node_address += 1;
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
                break;
            }
            else {
                const int num_levels = trailing_zeros(bit_trail);
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

      const int triangles_address = __ldg(((int*)(nodes + node_address) + 3));

      if (triangles_address != -1) {
        const int triangles_count = __ldg(((int*)(nodes + node_address) + 8));
        for (unsigned int i = 0; i < triangles_count; i++) {
            const float d = bvh_triangle_intersection((float4*)(triangles + triangles_address + i), origin, ray);

            if (d < depth) {
                depth = d;
                hit_id = triangles_address + i;
            }
        }
      }

      if (bit_trail == 0) {
          break;
      }
      else {
          const int num_levels = trailing_zeros(bit_trail);
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
