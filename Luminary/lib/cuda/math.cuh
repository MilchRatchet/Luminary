#ifndef CU_MATH_H
#define CU_MATH_H

#include "utils.cuh"
#include <float.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

__device__
vec3 cross_product(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.y*b.z - a.z*b.y;
    result.y = a.z*b.x - a.x*b.z;
    result.z = a.x*b.y - a.y*b.x;

    return result;
}

__device__
float dot_product(const vec3 a, const vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
float lerp(const float a, const float b, const float t) {
    return a + t * (b - a);
}

__device__
vec3 vec_diff(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;

    return result;
}

__device__
vec3 reflect_vector(const vec3 v, const vec3 n) {
    vec3 result;

    const float dot = dot_product(v, n);

    result.x = v.x - 2.0f * dot * n.x;
    result.y = v.y - 2.0f * dot * n.y;
    result.z = v.z - 2.0f * dot * n.z;

    return result;
}

__device__
vec3 scale_vector(vec3 vector, const float scale) {
    vector.x *= scale;
    vector.y *= scale;
    vector.z *= scale;

    return vector;
}

__device__
float get_length(const vec3 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

__device__
vec3 normalize_vector(vec3 vector) {
    const float inv_length = 1.0f / get_length(vector);

    return scale_vector(vector, inv_length);
}

__device__
vec3 get_coordinates_in_triangle(const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 point) {
    const vec3 diff = vec_diff(point, vertex);
    const float d00 = dot_product(edge1,edge1);
    const float d01 = dot_product(edge1,edge2);
    const float d11 = dot_product(edge2,edge2);
    const float d20 = dot_product(diff,edge1);
    const float d21 = dot_product(diff,edge2);
    const float denom = 1.0f / (d00 * d11 - d01 * d01);
    vec3 result;
    result.x = (d11 * d20 - d01 * d21) * denom;
    result.y = (d00 * d21 - d01 * d20) * denom;
    return result;
}

__device__
vec3 lerp_normals(const vec3 vertex_normal, const vec3 edge1_normal, const vec3 edge2_normal, const float lambda, const float mu) {
    vec3 result;

    result.x = vertex_normal.x + lambda * edge1_normal.x + mu * edge2_normal.x;
    result.y = vertex_normal.y + lambda * edge1_normal.y + mu * edge2_normal.y;
    result.z = vertex_normal.z + lambda * edge1_normal.z + mu * edge2_normal.z;

    return normalize_vector(result);
}

__device__
UV lerp_uv(const UV vertex_texture, const UV edge1_texture, const UV edge2_texture, const float lambda, const float mu) {
    UV result;

    result.u = vertex_texture.u + lambda * edge1_texture.u + mu * edge2_texture.u;
    result.v = vertex_texture.v + lambda * edge1_texture.v + mu * edge2_texture.v;

    return result;
}

__device__
vec3 sample_ray_from_angles_and_vector(const float theta, const float phi, const vec3 basis) {
    vec3 u1, u2;
    if (basis.z < -1.0f + 2.0f * eps) {
        u1.x = 0.0f;
        u1.y = -1.0f;
        u1.z = 0.0f;
        u2.x = -1.0f;
        u2.y = 0.0f;
        u2.z = 0.0f;
    }
    else
    {
        const float a = 1.0f/(1.0f+basis.z);
        const float b = -basis.x*basis.y*a;
        u1.x = 1.0f - basis.x*basis.x*a;
        u1.y = b;
        u1.z = -basis.x;
        u2.x = b;
        u2.y = 1.0f-basis.y*basis.y*a;
        u2.z = -basis.y;
    }

    const float c1 = sinf(theta) * cosf(phi);
    const float c2 = sinf(theta) * sinf(phi);
    const float c3 = cosf(theta);

    vec3 result;

    result.x = c1 * u1.x + c2 * u2.x + c3 * basis.x;
    result.y = c1 * u1.y + c2 * u2.y + c3 * basis.y;
    result.z = c1 * u1.z + c2 * u2.z + c3 * basis.z;

    return normalize_vector(result);
}

__device__
int trailing_zeros(const unsigned int n) {
    return __clz(__brev(n));
}

__device__
Quaternion inverse_quaternion(const Quaternion q) {
    Quaternion result;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;
    result.w = q.w;
    return result;
}

__device__
Quaternion get_rotation_to_z_canonical(const vec3 v) {
    Quaternion res;
    if (v.z < -1.0f + eps) {
        res.x = 1.0f;
        res.y = 0.0f;
        res.z = 0.0f;
        res.w = 0.0f;
        return res;
    }
    res.x = v.y;
    res.y = -v.x;
    res.z = 0.0f;
    res.w = 1.0f + v.z;

    const float norm = 1.0f / sqrtf(res.x * res.x + res.y * res.y + res.w * res.w);

    res.x *= norm;
    res.y *= norm;
    res.w *= norm;

    return res;
}

__device__
vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
    vec3 result;

    vec3 u;
    u.x = q.x;
    u.y = q.y;
    u.z = q.z;

    const float s = q.w;

    const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
    const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

    vec3 cross;

    cross.x = u.y*v.z - u.z*v.y;
    cross.y = u.z*v.x - u.x*v.z;
    cross.z = u.x*v.y - u.y*v.x;

    result.x = 2.0f * dot_uv * u.x + ((s*s)-dot_uu) * v.x + 2.0f * s * cross.x;
    result.y = 2.0f * dot_uv * u.y + ((s*s)-dot_uu) * v.y + 2.0f * s * cross.y;
    result.z = 2.0f * dot_uv * u.z + ((s*s)-dot_uu) * v.z + 2.0f * s * cross.z;

    return result;
}

#endif /* CU_MATH_H */
