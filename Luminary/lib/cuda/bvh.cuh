#ifndef CU_BVH_H
#define CU_BVH_H

#include "utils.cuh"
#include "math.cuh"
#include "minmax.cuh"
#include "memory.cuh"
#include "ocean.cuh"
#include <cuda_runtime_api.h>

struct traversal_result {
  unsigned int hit_id;
  float depth;
} typedef traversal_result;

__device__
float bvh_triangle_intersection(const float4* triangles, const vec3 origin, const vec3 ray) {
    const float4 v1 = __ldg(triangles);
    const float4 v2 = __ldg(triangles + 1);
    const float v3 = __ldg((float*)(triangles + 2));

    vec3 vertex = get_vector(v1.x, v1.y, v1.z);
    vec3 edge1 = get_vector(v1.w, v2.x, v2.y);
    vec3 edge2 = get_vector(v2.z, v2.w, v3);

    const vec3 h = cross_product(ray, edge2);
    const float a = dot_product(edge1, h);

    if (__builtin_expect(a > -0.00000001f && a < 0.00000001f, 0)) return FLT_MAX;

    const float f = 1.0f / a;
    const vec3 s = sub_vector(origin, vertex);
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
unsigned char get_8bit(const unsigned int input, const unsigned int bitshift) {
    return (input >> bitshift) & 0x000000FF;
}

__device__
unsigned int sign_extend_s8x4(unsigned int a)
{
    unsigned int result;
    asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(result) : "r"(a));
    return result;
}

__device__
unsigned int __bfind(unsigned int a)
{
    unsigned int result;
    asm volatile("bfind.u32 %0, %1; " : "=r"(result) : "r"(a));
    return result;
}

#define STACK_SIZE_SM 9
#define STACK_SIZE 32

#define STACK_POP(X)                                                    \
    {                                                                   \
        packed_stptr_invoct.x--;                                        \
        if (packed_stptr_invoct.x < STACK_SIZE_SM)                      \
            X = traversal_stack_sm[threadIdx.x][packed_stptr_invoct.x]; \
        else                                                            \
            X = traversal_stack[packed_stptr_invoct.x - STACK_SIZE_SM]; \
    }

#define STACK_PUSH(X)                                                   \
    {                                                                   \
        if (packed_stptr_invoct.x < STACK_SIZE_SM)                      \
            traversal_stack_sm[threadIdx.x][packed_stptr_invoct.x] = X; \
        else                                                            \
            traversal_stack[packed_stptr_invoct.x - STACK_SIZE_SM] = X; \
        packed_stptr_invoct.x++;                                        \
    }

__global__ __launch_bounds__(THREADS_PER_BLOCK,10)
void process_trace_tasks() {
    const uint16_t trace_task_count = device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4];
    uint16_t offset = 0;

	uint2 traversal_stack[STACK_SIZE];
    __shared__ uint2 traversal_stack_sm[THREADS_PER_BLOCK][STACK_SIZE_SM];

    unsigned int hit_id;
    float depth;

    vec3 inv_ray;
    vec3 origin;
    vec3 ray;

    uint2 node_task = make_uint2(0, 0);
    uint2 triangle_task = make_uint2(0, 0);

    ushort2 packed_stptr_invoct;
    packed_stptr_invoct.x = 0;

    while (1) {
        if (packed_stptr_invoct.x == 0 && node_task.y <= 0x00ffffff && triangle_task.y == 0) {
            if (offset >= trace_task_count) break;

            const TraceTask task = load_trace_task_essentials(device_trace_tasks + get_task_address(offset));
            const float2 result = __ldcs((float2*)(device_trace_results + get_task_address(offset)));

            node_task = make_uint2(0, 0x80000000);
            triangle_task = make_uint2(0, 0);

            depth = result.x;
            hit_id = float_as_uint(result.y);

            origin = task.origin;
            ray = task.ray;

            inv_ray.x = 1.0f / (fabsf(task.ray.x) > eps ? task.ray.x : copysignf(eps, task.ray.x));
            inv_ray.y = 1.0f / (fabsf(task.ray.y) > eps ? task.ray.y : copysignf(eps, task.ray.y));
            inv_ray.z = 1.0f / (fabsf(task.ray.z) > eps ? task.ray.z : copysignf(eps, task.ray.z));

            packed_stptr_invoct.y = (((task.ray.x < 0.0f) ? 1 : 0) << 2) | (((task.ray.y < 0.0f) ? 1 : 0) << 1) | (((task.ray.z < 0.0f) ? 1 : 0) << 0);
            packed_stptr_invoct.y = 0b111 - packed_stptr_invoct.y;
            packed_stptr_invoct.x = 0;
        }

        int lost_loop_iterations = 0;
        const int starting_threads = __popc(__activemask());

        while (1) {
            if (node_task.y > 0x00ffffff) {
                const unsigned int hits = node_task.y;
                const unsigned int imask = node_task.y;
                const unsigned int child_bit_index = __bfind(hits);
                const unsigned int child_node_base_index = node_task.x;

                node_task.y &= ~(1 << child_bit_index);

                if (node_task.y > 0x00ffffff) {
                    STACK_PUSH(node_task);
                }

                const unsigned int slot_index = (child_bit_index - 24) ^ (unsigned int)packed_stptr_invoct.y;
                const unsigned int inverse_octant4 = (unsigned int)packed_stptr_invoct.y * 0x01010101u;
                const unsigned int relative_index = __popc(imask & ~(0xffffffff << slot_index));
                const unsigned int child_node_index = child_node_base_index + relative_index;

                float4* data_ptr = (float4*)(device_scene.nodes + child_node_index);

                const float4 data0 = __ldg(data_ptr + 0);
                const float4 data1 = __ldg(data_ptr + 1);
                const float4 data2 = __ldg(data_ptr + 2);
                const float4 data3 = __ldg(data_ptr + 3);
                const float4 data4 = __ldg(data_ptr + 4);

                vec3 p;
                p.x = data0.x;
                p.y = data0.y;
                p.z = data0.z;

                int3 e;
                e.x = *((char*)&data0.w + 0);
                e.y = *((char*)&data0.w + 1);
                e.z = *((char*)&data0.w + 2);

                node_task.x = float_as_uint(data1.x);
                triangle_task.x = float_as_uint(data1.y);
                triangle_task.y = 0;
                unsigned int hit_mask = 0;

                vec3 scaled_inv_ray;
                scaled_inv_ray.x = uint_as_float((e.x + 127) << 23) * inv_ray.x;
                scaled_inv_ray.y = uint_as_float((e.y + 127) << 23) * inv_ray.y;
                scaled_inv_ray.z = uint_as_float((e.z + 127) << 23) * inv_ray.z;

                vec3 shifted_origin;
                shifted_origin.x = (p.x - origin.x) * inv_ray.x;
                shifted_origin.y = (p.y - origin.y) * inv_ray.y;
                shifted_origin.z = (p.z - origin.z) * inv_ray.z;

                {
                    const unsigned int meta4 = float_as_int(data1.z);
                    const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                    const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                    const unsigned int bit_index4 = (meta4 ^ (inverse_octant4 & inner_mask4)) & 0x1f1f1f1f;
                    const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

                    const unsigned int low_x = (inv_ray.x < 0.0f) ? float_as_uint(data3.z) : float_as_uint(data2.x);
                    const unsigned int high_x = (inv_ray.x < 0.0f) ? float_as_uint(data2.x) : float_as_uint(data3.z);
                    const unsigned int low_y = (inv_ray.y < 0.0f) ? float_as_uint(data4.x) : float_as_uint(data2.z);
                    const unsigned int high_y = (inv_ray.y < 0.0f) ? float_as_uint(data2.z) : float_as_uint(data4.x);
                    const unsigned int low_z = (inv_ray.z < 0.0f) ? float_as_uint(data4.z) : float_as_uint(data3.x);
                    const unsigned int high_z = (inv_ray.z < 0.0f) ? float_as_uint(data3.x) : float_as_uint(data4.z);

                    float min_x[4];
                    float max_x[4];
                    float min_y[4];
                    float max_y[4];
                    float min_z[4];
                    float max_z[4];

                    min_x[0] = get_8bit(low_x, 0) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[1] = get_8bit(low_x, 8) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[2] = get_8bit(low_x, 16) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[3] = get_8bit(low_x, 24) * scaled_inv_ray.x + shifted_origin.x;

                    max_x[0] = get_8bit(high_x, 0) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[1] = get_8bit(high_x, 8) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[2] = get_8bit(high_x, 16) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[3] = get_8bit(high_x, 24) * scaled_inv_ray.x + shifted_origin.x;

                    min_y[0] = get_8bit(low_y, 0) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[1] = get_8bit(low_y, 8) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[2] = get_8bit(low_y, 16) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[3] = get_8bit(low_y, 24) * scaled_inv_ray.y + shifted_origin.y;

                    max_y[0] = get_8bit(high_y, 0) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[1] = get_8bit(high_y, 8) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[2] = get_8bit(high_y, 16) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[3] = get_8bit(high_y, 24) * scaled_inv_ray.y + shifted_origin.y;

                    min_z[0] = get_8bit(low_z, 0) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[1] = get_8bit(low_z, 8) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[2] = get_8bit(low_z, 16) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[3] = get_8bit(low_z, 24) * scaled_inv_ray.z + shifted_origin.z;

                    max_z[0] = get_8bit(high_z, 0) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[1] = get_8bit(high_z, 8) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[2] = get_8bit(high_z, 16) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[3] = get_8bit(high_z, 24) * scaled_inv_ray.z + shifted_origin.z;

                    float slab_min, slab_max;
                    int intersection;

                    slab_min = fmaxf(fmax_fmax(min_x[0], min_y[0], min_z[0]), eps);
                    slab_max = fminf(fmin_fmin(max_x[0], max_y[0], max_z[0]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[1], min_y[1], min_z[1]), eps);
                    slab_max = fminf(fmin_fmin(max_x[1], max_y[1], max_z[1]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[2], min_y[2], min_z[2]), eps);
                    slab_max = fminf(fmin_fmin(max_x[2], max_y[2], max_z[2]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[3], min_y[3], min_z[3]), eps);
                    slab_max = fminf(fmin_fmin(max_x[3], max_y[3], max_z[3]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));
                }

                {
                    const unsigned int meta4 = float_as_int(data1.w);
                    const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                    const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                    const unsigned int bit_index4 = (meta4 ^ (inverse_octant4 & inner_mask4)) & 0x1f1f1f1f;
                    const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

                    const unsigned int low_x = (inv_ray.x < 0.0f) ? float_as_uint(data3.w) : float_as_uint(data2.y);
                    const unsigned int high_x = (inv_ray.x < 0.0f) ? float_as_uint(data2.y) : float_as_uint(data3.w);
                    const unsigned int low_y = (inv_ray.y < 0.0f) ? float_as_uint(data4.y) : float_as_uint(data2.w);
                    const unsigned int high_y = (inv_ray.y < 0.0f) ? float_as_uint(data2.w) : float_as_uint(data4.y);
                    const unsigned int low_z = (inv_ray.z < 0.0f) ? float_as_uint(data4.w) : float_as_uint(data3.y);
                    const unsigned int high_z = (inv_ray.z < 0.0f) ? float_as_uint(data3.y) : float_as_uint(data4.w);

                    float min_x[4];
                    float max_x[4];
                    float min_y[4];
                    float max_y[4];
                    float min_z[4];
                    float max_z[4];

                    min_x[0] = get_8bit(low_x, 0) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[1] = get_8bit(low_x, 8) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[2] = get_8bit(low_x, 16) * scaled_inv_ray.x + shifted_origin.x;
                    min_x[3] = get_8bit(low_x, 24) * scaled_inv_ray.x + shifted_origin.x;

                    max_x[0] = get_8bit(high_x, 0) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[1] = get_8bit(high_x, 8) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[2] = get_8bit(high_x, 16) * scaled_inv_ray.x + shifted_origin.x;
                    max_x[3] = get_8bit(high_x, 24) * scaled_inv_ray.x + shifted_origin.x;

                    min_y[0] = get_8bit(low_y, 0) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[1] = get_8bit(low_y, 8) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[2] = get_8bit(low_y, 16) * scaled_inv_ray.y + shifted_origin.y;
                    min_y[3] = get_8bit(low_y, 24) * scaled_inv_ray.y + shifted_origin.y;

                    max_y[0] = get_8bit(high_y, 0) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[1] = get_8bit(high_y, 8) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[2] = get_8bit(high_y, 16) * scaled_inv_ray.y + shifted_origin.y;
                    max_y[3] = get_8bit(high_y, 24) * scaled_inv_ray.y + shifted_origin.y;

                    min_z[0] = get_8bit(low_z, 0) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[1] = get_8bit(low_z, 8) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[2] = get_8bit(low_z, 16) * scaled_inv_ray.z + shifted_origin.z;
                    min_z[3] = get_8bit(low_z, 24) * scaled_inv_ray.z + shifted_origin.z;

                    max_z[0] = get_8bit(high_z, 0) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[1] = get_8bit(high_z, 8) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[2] = get_8bit(high_z, 16) * scaled_inv_ray.z + shifted_origin.z;
                    max_z[3] = get_8bit(high_z, 24) * scaled_inv_ray.z + shifted_origin.z;

                    float slab_min, slab_max;
                    int intersection;

                    slab_min = fmaxf(fmax_fmax(min_x[0], min_y[0], min_z[0]), eps);
                    slab_max = fminf(fmin_fmin(max_x[0], max_y[0], max_z[0]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[1], min_y[1], min_z[1]), eps);
                    slab_max = fminf(fmin_fmin(max_x[1], max_y[1], max_z[1]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[2], min_y[2], min_z[2]), eps);
                    slab_max = fminf(fmin_fmin(max_x[2], max_y[2], max_z[2]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));

                    slab_min = fmaxf(fmax_fmax(min_x[3], min_y[3], min_z[3]), eps);
                    slab_max = fminf(fmin_fmin(max_x[3], max_y[3], max_z[3]), depth);
                    intersection = slab_min <= slab_max;

                    if (intersection)
                        asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(hit_mask) : "r"(child_bits4), "r"(bit_index4), "r"(hit_mask));
                }

                node_task.y = (hit_mask & 0xff000000) | (*((unsigned char*)&data0.w + 3));
                triangle_task.y = hit_mask & 0x00ffffff;
            }

            const int active_threads = __popc(__activemask());

            while (triangle_task.y != 0) {
                const int threshold = active_threads * 0.2f;
                const int triangle_threads = __popc(__activemask());

                if (triangle_threads < threshold) {
                    STACK_PUSH(triangle_task);
                    break;
                }

                const int triangle_index = __bfind(triangle_task.y);

                const float d = bvh_triangle_intersection((float4*)(device_scene.traversal_triangles + triangle_index + triangle_task.x), origin, ray);

                if (d < depth) {
                    depth = d;
                    hit_id = triangle_index + triangle_task.x;
                }

                triangle_task.y &= ~(1 << triangle_index);
            }

            if (node_task.y <= 0x00ffffff) {
                if (packed_stptr_invoct.x > 0) {
                    STACK_POP(node_task);

                    if (node_task.y <= 0x00ffffff) {
                        triangle_task = node_task;
                        node_task = make_uint2(0, 0);
                    }
                } else {
                    float2 result;
                    result.x = depth;
                    result.y = uint_as_float(hit_id);

                    __stcs((float2*)(device_trace_results + get_task_address(offset)), result);

                    offset++;

                    break;
                }
            }

            const int Nd = 4;
            const int Nw = 16;
            lost_loop_iterations += starting_threads - __popc(__activemask()) - Nd;
            if (lost_loop_iterations >= Nw) break;
        }
    }
}

#endif /* CU_BVH_H */
