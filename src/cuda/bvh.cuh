#ifndef CU_BVH_H
#define CU_BVH_H

#include <cuda_runtime_api.h>

#include "intrinsics.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

struct traversal_result {
  unsigned int hit_id;
  float depth;
} typedef traversal_result;

__device__ unsigned char get_8bit(const unsigned int input, const unsigned int bitshift) {
  return (input >> bitshift) & 0x000000FF;
}

#define STACK_SIZE_SM 10
#define STACK_SIZE 32

#define STACK_POP(X)                                  \
  {                                                   \
    stack_ptr--;                                      \
    if (stack_ptr < STACK_SIZE_SM)                    \
      X = traversal_stack_sm[threadIdx.x][stack_ptr]; \
    else                                              \
      X = traversal_stack[stack_ptr - STACK_SIZE_SM]; \
  }

#define STACK_PUSH(X)                                 \
  {                                                   \
    if (stack_ptr < STACK_SIZE_SM)                    \
      traversal_stack_sm[threadIdx.x][stack_ptr] = X; \
    else                                              \
      traversal_stack[stack_ptr - STACK_SIZE_SM] = X; \
    stack_ptr++;                                      \
  }

__device__ void traverse_bvh(const bool check_alpha) {
  const uint16_t trace_task_count = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];
  uint16_t offset                 = 0;

  uint2 traversal_stack[STACK_SIZE];
  __shared__ uint2 traversal_stack_sm[THREADS_PER_BLOCK][STACK_SIZE_SM];

  unsigned int hit_id;
  float depth;

  vec3 inv_ray;
  vec3 origin;
  vec3 ray;

  float cost = 0.0f;

  uint2 node_task     = make_uint2(0, 0);
  uint2 triangle_task = make_uint2(0, 0);

  unsigned int stack_ptr = 0;
  unsigned int octant;

  while (1) {
    if (stack_ptr == 0 && node_task.y <= 0x00ffffff && triangle_task.y == 0) {
      if (offset >= trace_task_count)
        break;

      const TraceTask task = load_trace_task_essentials(device_trace_tasks + get_task_address(offset));
      const float2 result  = __ldcs((float2*) (device.trace_results + get_task_address(offset)));

      node_task     = make_uint2(0, 0x80000000);
      triangle_task = make_uint2(0, 0);

      depth  = result.x;
      hit_id = __float_as_uint(result.y);

      origin = task.origin;
      ray    = task.ray;

      inv_ray.x = 1.0f / (fabsf(task.ray.x) > eps * eps ? task.ray.x : copysignf(eps * eps, task.ray.x));
      inv_ray.y = 1.0f / (fabsf(task.ray.y) > eps * eps ? task.ray.y : copysignf(eps * eps, task.ray.y));
      inv_ray.z = 1.0f / (fabsf(task.ray.z) > eps * eps ? task.ray.z : copysignf(eps * eps, task.ray.z));

      octant    = ((task.ray.x < 0.0f) ? 0b100 : 0) | ((task.ray.y < 0.0f) ? 0b10 : 0) | ((task.ray.z < 0.0f) ? 0b1 : 0);
      octant    = 0b111 - octant;
      stack_ptr = 0;

      cost = 0.0f;
    }

    if (node_task.y > 0x00ffffff) {
      const unsigned int hits                  = node_task.y;
      const unsigned int imask                 = node_task.y;
      const unsigned int child_bit_index       = __bfind(hits);
      const unsigned int child_node_base_index = node_task.x;

      node_task.y ^= (1 << child_bit_index);

      if (node_task.y > 0x00ffffff) {
        STACK_PUSH(node_task);
      }

      cost += 1.0f;

      const unsigned int slot_index       = (child_bit_index - 24) ^ octant;
      const unsigned int inverse_octant4  = octant * 0x01010101u;
      const unsigned int relative_index   = __popc(imask & ~(0xffffffff << slot_index));
      const unsigned int child_node_index = child_node_base_index + relative_index;

      float4* data_ptr = (float4*) (device_scene.nodes + child_node_index);

      const float4 data0 = __ldg(data_ptr + 0);
      const float4 data1 = __ldg(data_ptr + 1);
      const float4 data2 = __ldg(data_ptr + 2);
      const float4 data3 = __ldg(data_ptr + 3);
      const float4 data4 = __ldg(data_ptr + 4);

      const vec3 p = get_vector(data0.x, data0.y, data0.z);

      float3 e;
      e.x = *((char*) &data0.w + 0);
      e.y = *((char*) &data0.w + 1);
      e.z = *((char*) &data0.w + 2);

      node_task.x           = __float_as_uint(data1.x);
      triangle_task.x       = __float_as_uint(data1.y);
      triangle_task.y       = 0;
      unsigned int hit_mask = 0;

      vec3 scaled_inv_ray;
      scaled_inv_ray.x = exp2f(e.x) * inv_ray.x;
      scaled_inv_ray.y = exp2f(e.y) * inv_ray.y;
      scaled_inv_ray.z = exp2f(e.z) * inv_ray.z;

      const float diag_shift = (p.x - origin.x) * inv_ray.x;

      const float shifted_eps   = eps * eps - diag_shift;
      const float shifted_depth = depth - diag_shift;

      const float shifted_origin_y = (p.y - origin.y) * inv_ray.y - diag_shift;
      const float shifted_origin_z = (p.z - origin.z) * inv_ray.z - diag_shift;

      {
        const unsigned int meta4       = __float_as_int(data1.z);
        const unsigned int is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
        const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
        const unsigned int bit_index4  = (meta4 ^ (inverse_octant4 & inner_mask4)) & 0x1f1f1f1f;
        const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

        const unsigned int low_x  = __uslctf(__float_as_uint(data2.x), __float_as_uint(data3.z), inv_ray.x);
        const unsigned int high_x = __uslctf(__float_as_uint(data3.z), __float_as_uint(data2.x), inv_ray.x);
        const unsigned int low_y  = __uslctf(__float_as_uint(data2.z), __float_as_uint(data4.x), inv_ray.y);
        const unsigned int high_y = __uslctf(__float_as_uint(data4.x), __float_as_uint(data2.z), inv_ray.y);
        const unsigned int low_z  = __uslctf(__float_as_uint(data3.x), __float_as_uint(data4.z), inv_ray.z);
        const unsigned int high_z = __uslctf(__float_as_uint(data4.z), __float_as_uint(data3.x), inv_ray.z);

        float min_x[4];
        float max_x[4];
        float min_y[4];
        float max_y[4];
        float min_z[4];
        float max_z[4];

        min_x[0] = get_8bit(low_x, 0) * scaled_inv_ray.x;
        min_x[1] = get_8bit(low_x, 8) * scaled_inv_ray.x;
        min_x[2] = get_8bit(low_x, 16) * scaled_inv_ray.x;
        min_x[3] = get_8bit(low_x, 24) * scaled_inv_ray.x;

        max_x[0] = get_8bit(high_x, 0) * scaled_inv_ray.x;
        max_x[1] = get_8bit(high_x, 8) * scaled_inv_ray.x;
        max_x[2] = get_8bit(high_x, 16) * scaled_inv_ray.x;
        max_x[3] = get_8bit(high_x, 24) * scaled_inv_ray.x;

        min_y[0] = get_8bit(low_y, 0) * scaled_inv_ray.y + shifted_origin_y;
        min_y[1] = get_8bit(low_y, 8) * scaled_inv_ray.y + shifted_origin_y;
        min_y[2] = get_8bit(low_y, 16) * scaled_inv_ray.y + shifted_origin_y;
        min_y[3] = get_8bit(low_y, 24) * scaled_inv_ray.y + shifted_origin_y;

        max_y[0] = get_8bit(high_y, 0) * scaled_inv_ray.y + shifted_origin_y;
        max_y[1] = get_8bit(high_y, 8) * scaled_inv_ray.y + shifted_origin_y;
        max_y[2] = get_8bit(high_y, 16) * scaled_inv_ray.y + shifted_origin_y;
        max_y[3] = get_8bit(high_y, 24) * scaled_inv_ray.y + shifted_origin_y;

        min_z[0] = get_8bit(low_z, 0) * scaled_inv_ray.z + shifted_origin_z;
        min_z[1] = get_8bit(low_z, 8) * scaled_inv_ray.z + shifted_origin_z;
        min_z[2] = get_8bit(low_z, 16) * scaled_inv_ray.z + shifted_origin_z;
        min_z[3] = get_8bit(low_z, 24) * scaled_inv_ray.z + shifted_origin_z;

        max_z[0] = get_8bit(high_z, 0) * scaled_inv_ray.z + shifted_origin_z;
        max_z[1] = get_8bit(high_z, 8) * scaled_inv_ray.z + shifted_origin_z;
        max_z[2] = get_8bit(high_z, 16) * scaled_inv_ray.z + shifted_origin_z;
        max_z[3] = get_8bit(high_z, 24) * scaled_inv_ray.z + shifted_origin_z;

        // We don't use fmax_fmax/fmin_fmin here because we are ALU bound on Ampere
        float slab_min, slab_max;

        slab_min = fmaxf(fmaxf(min_x[0], fmaxf(min_y[0], min_z[0])), shifted_eps);
        slab_max = fminf(fminf(max_x[0], fminf(max_y[0], max_z[0])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[1], fmaxf(min_y[1], min_z[1])), shifted_eps);
        slab_max = fminf(fminf(max_x[1], fminf(max_y[1], max_z[1])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[2], fmaxf(min_y[2], min_z[2])), shifted_eps);
        slab_max = fminf(fminf(max_x[2], fminf(max_y[2], max_z[2])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[3], fmaxf(min_y[3], min_z[3])), shifted_eps);
        slab_max = fminf(fminf(max_x[3], fminf(max_y[3], max_z[3])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));
      }

      {
        const unsigned int meta4       = __float_as_int(data1.w);
        const unsigned int is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
        const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
        const unsigned int bit_index4  = (meta4 ^ (inverse_octant4 & inner_mask4)) & 0x1f1f1f1f;
        const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

        const unsigned int low_x  = __uslctf(__float_as_uint(data2.y), __float_as_uint(data3.w), inv_ray.x);
        const unsigned int high_x = __uslctf(__float_as_uint(data3.w), __float_as_uint(data2.y), inv_ray.x);
        const unsigned int low_y  = __uslctf(__float_as_uint(data2.w), __float_as_uint(data4.y), inv_ray.y);
        const unsigned int high_y = __uslctf(__float_as_uint(data4.y), __float_as_uint(data2.w), inv_ray.y);
        const unsigned int low_z  = __uslctf(__float_as_uint(data3.y), __float_as_uint(data4.w), inv_ray.z);
        const unsigned int high_z = __uslctf(__float_as_uint(data4.w), __float_as_uint(data3.y), inv_ray.z);

        float min_x[4];
        float max_x[4];
        float min_y[4];
        float max_y[4];
        float min_z[4];
        float max_z[4];

        min_x[0] = get_8bit(low_x, 0) * scaled_inv_ray.x;
        min_x[1] = get_8bit(low_x, 8) * scaled_inv_ray.x;
        min_x[2] = get_8bit(low_x, 16) * scaled_inv_ray.x;
        min_x[3] = get_8bit(low_x, 24) * scaled_inv_ray.x;

        max_x[0] = get_8bit(high_x, 0) * scaled_inv_ray.x;
        max_x[1] = get_8bit(high_x, 8) * scaled_inv_ray.x;
        max_x[2] = get_8bit(high_x, 16) * scaled_inv_ray.x;
        max_x[3] = get_8bit(high_x, 24) * scaled_inv_ray.x;

        min_y[0] = get_8bit(low_y, 0) * scaled_inv_ray.y + shifted_origin_y;
        min_y[1] = get_8bit(low_y, 8) * scaled_inv_ray.y + shifted_origin_y;
        min_y[2] = get_8bit(low_y, 16) * scaled_inv_ray.y + shifted_origin_y;
        min_y[3] = get_8bit(low_y, 24) * scaled_inv_ray.y + shifted_origin_y;

        max_y[0] = get_8bit(high_y, 0) * scaled_inv_ray.y + shifted_origin_y;
        max_y[1] = get_8bit(high_y, 8) * scaled_inv_ray.y + shifted_origin_y;
        max_y[2] = get_8bit(high_y, 16) * scaled_inv_ray.y + shifted_origin_y;
        max_y[3] = get_8bit(high_y, 24) * scaled_inv_ray.y + shifted_origin_y;

        min_z[0] = get_8bit(low_z, 0) * scaled_inv_ray.z + shifted_origin_z;
        min_z[1] = get_8bit(low_z, 8) * scaled_inv_ray.z + shifted_origin_z;
        min_z[2] = get_8bit(low_z, 16) * scaled_inv_ray.z + shifted_origin_z;
        min_z[3] = get_8bit(low_z, 24) * scaled_inv_ray.z + shifted_origin_z;

        max_z[0] = get_8bit(high_z, 0) * scaled_inv_ray.z + shifted_origin_z;
        max_z[1] = get_8bit(high_z, 8) * scaled_inv_ray.z + shifted_origin_z;
        max_z[2] = get_8bit(high_z, 16) * scaled_inv_ray.z + shifted_origin_z;
        max_z[3] = get_8bit(high_z, 24) * scaled_inv_ray.z + shifted_origin_z;

        // We don't use fmax_fmax/fmin_fmin here because we are ALU bound on Ampere
        float slab_min, slab_max;

        slab_min = fmaxf(fmaxf(min_x[0], fmaxf(min_y[0], min_z[0])), shifted_eps);
        slab_max = fminf(fminf(max_x[0], fminf(max_y[0], max_z[0])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[1], fmaxf(min_y[1], min_z[1])), shifted_eps);
        slab_max = fminf(fminf(max_x[1], fminf(max_y[1], max_z[1])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[2], fmaxf(min_y[2], min_z[2])), shifted_eps);
        slab_max = fminf(fminf(max_x[2], fminf(max_y[2], max_z[2])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

        slab_min = fmaxf(fmaxf(min_x[3], fmaxf(min_y[3], min_z[3])), shifted_eps);
        slab_max = fminf(fminf(max_x[3], fminf(max_y[3], max_z[3])), shifted_depth);

        if (slab_min <= slab_max)
          asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));
      }

      node_task.y     = (hit_mask & 0xff000000) | (*((unsigned char*) &data0.w + 3));
      triangle_task.y = hit_mask & 0x00ffffff;
    }

    const int active_threads = __popc(__activemask());

    while (triangle_task.y != 0) {
      const int threshold        = active_threads * 0.2f;
      const int triangle_threads = __popc(__activemask());

      if (triangle_threads < threshold) {
        STACK_PUSH(triangle_task);
        break;
      }

      cost += 0.5f;

      const int triangle_index = __bfind(triangle_task.y);

      const TraversalTriangle triangle = load_traversal_triangle(triangle_index + triangle_task.x);

      if (check_alpha) {
        UV coords;
        const float d = bvh_triangle_intersection_uv(triangle, origin, ray, coords);

        if (d < depth) {
          bool opaque = true;

          if (triangle.albedo_tex) {
            const UV tex_coords = load_triangle_tex_coords(triangle_index + triangle_task.x, coords);
            const float4 albedo = tex2D<float4>(device.albedo_atlas[triangle.albedo_tex], tex_coords.u, 1.0f - tex_coords.v);

            if (albedo.w <= device_scene.material.alpha_cutoff) {
              opaque = false;
            }
          }

          if (opaque) {
            depth  = d;
            hit_id = triangle_index + triangle_task.x;
          }
        }
      }
      else {
        const float d = bvh_triangle_intersection(triangle, origin, ray);

        if (d < depth) {
          depth  = d;
          hit_id = triangle_index + triangle_task.x;
        }
      }

      triangle_task.y ^= (1 << triangle_index);
    }

    if (node_task.y <= 0x00ffffff) {
      if (stack_ptr > 0) {
        STACK_POP(node_task);

        if (node_task.y <= 0x00ffffff) {
          triangle_task = node_task;
          node_task     = make_uint2(0, 0);
        }
      }
      else {
        if (device_shading_mode == SHADING_HEAT && hit_id >= TRIANGLE_ID_LIMIT) {
          hit_id = 0;
        }

        float2 result;
        result.x = (device_shading_mode == SHADING_HEAT) ? cost : depth;
        result.y = __uint_as_float(hit_id);

        __stcs((float2*) (device.trace_results + get_task_address(offset)), result);

        offset++;
      }
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_trace_tasks() {
  traverse_bvh(false);
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_trace_tasks_alpha_cutoff() {
  traverse_bvh(true);
}

#endif /* CU_BVH_H */
