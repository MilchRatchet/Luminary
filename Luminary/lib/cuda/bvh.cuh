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
int bvh_ray_box_intersect(const unsigned char lx, const unsigned char ly, const unsigned char lz,
                          const unsigned char hx, const unsigned char hy, const unsigned char hz,
                          const vec3 scaled_inv_ray, const vec3 shifted_origin, const float depth, float& out_dist)
{
    vec3 lo;
    lo.x = (float)lx * scaled_inv_ray.x + shifted_origin.x;
    lo.y = (float)ly * scaled_inv_ray.y + shifted_origin.y;
    lo.z = (float)lz * scaled_inv_ray.z + shifted_origin.z;

    vec3 hi;
    hi.x = (float)hx * scaled_inv_ray.x + shifted_origin.x;
    hi.y = (float)hy * scaled_inv_ray.y + shifted_origin.y;
    hi.z = (float)hz * scaled_inv_ray.z + shifted_origin.z;

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

#define STACK_POP(X) {                                               \
    stack_ptr--;                                                     \
    if (stack_ptr < stack_size_sm)                                   \
        X = traversal_stack_sm[threadIdx.x][stack_ptr];              \
    else                                                             \
        X = traversal_stack[stack_ptr - stack_size_sm];              \
}

#define STACK_PUSH(X) {                                              \
    if (stack_ptr < stack_size_sm)                                   \
        traversal_stack_sm[threadIdx.x][stack_ptr] = X;              \
    else                                                             \
        traversal_stack[stack_ptr - stack_size_sm] = X;              \
    stack_ptr++;                                                     \
}

__device__
traversal_result traverse_bvh(const vec3 origin, const vec3 ray, const Node8* nodes, const Traversal_Triangle* triangles) {
    float depth = device_scene.far_clip_distance;

    const int stack_size = 32;
	uint2 traversal_stack[stack_size];

    const int stack_size_sm = 12;
    __shared__ uint2 traversal_stack_sm[128][stack_size_sm];

    char stack_ptr = 0;

    unsigned int hit_id = 0xffffffff;

    vec3 inv_ray;
    inv_ray.x = 1.0f / (fabsf(ray.x) > eps ? ray.x : copysignf(eps, ray.x));
    inv_ray.y = 1.0f / (fabsf(ray.y) > eps ? ray.y : copysignf(eps, ray.y));
    inv_ray.z = 1.0f / (fabsf(ray.z) > eps ? ray.z : copysignf(eps, ray.z));

    uint2 node_task = make_uint2(0, 0b10000000000000000000000000000000);
    uint2 triangle_task = make_uint2(0, 0);

    unsigned int inverse_octant = (((ray.x < 0.0f) ? 1 : 0) << 2) | (((ray.y < 0.0f) ? 1 : 0) << 1) | (((ray.z < 0.0f) ? 1 : 0) << 0);
    inverse_octant = 7 - inverse_octant;

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

            {
                const unsigned int slot_index = (child_bit_index - 24) ^ inverse_octant;
                const unsigned int inverse_octant4 = inverse_octant * 0x01010101u;
                const unsigned int relative_index = __popc(imask & ~(0xffffffff << slot_index));
                const unsigned int child_node_index = child_node_base_index + relative_index;

                float4* data_ptr = (float4*)(nodes + child_node_index);

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

                    for (int i = 0; i < 4; i++) {
                        const float slab_min = fmaxf(fmax_fmax(min_x[i], min_y[i], min_z[i]), eps);
                        const float slab_max = fminf(fmin_fmin(max_x[i], max_y[i], max_z[i]), depth);

                        const int intersection = slab_min <= slab_max;

                        if (intersection) {
                            const unsigned int child_bits = get_8bit(child_bits4, i * 8);
                            const unsigned int bit_index = get_8bit(bit_index4, i * 8);
                            hit_mask |= child_bits << bit_index;
                        }
                    }
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

                    for (int i = 0; i < 4; i++) {
                        const float slab_min = fmaxf(fmax_fmax(min_x[i], min_y[i], min_z[i]), eps);
                        const float slab_max = fminf(fmin_fmin(max_x[i], max_y[i], max_z[i]), depth);

                        const int intersection = slab_min <= slab_max;

                        if (intersection) {
                            const unsigned int child_bits = get_8bit(child_bits4, i * 8);
                            const unsigned int bit_index = get_8bit(bit_index4, i * 8);
                            hit_mask |= child_bits << bit_index;
                        }
                    }
                }

                node_task.y = (hit_mask & 0xff000000) | (*((unsigned char*)&data0.w + 3));
                triangle_task.y = hit_mask & 0x00ffffff;
            }
        }
        else {
            triangle_task = node_task;
            node_task = make_uint2(0, 0);
        }

        while (triangle_task.y != 0) {
            int triangle_index = __bfind(triangle_task.y);

            const float d = bvh_triangle_intersection((float4*)(triangles + triangle_index + triangle_task.x), origin, ray);

            if (d < depth) {
                depth = d;
                hit_id = triangle_index + triangle_task.x;
            }

            triangle_task.y &= ~(1 << triangle_index);
        }

        if (node_task.y <= 0x00ffffff) {
            if (stack_ptr > 0) {
                STACK_POP(node_task);
            } else {
                break;
            }
        }
    }

    traversal_result result;
    result.hit_id = hit_id;
    result.depth = depth;

    return result;
}

#endif /* CU_BVH_H */
