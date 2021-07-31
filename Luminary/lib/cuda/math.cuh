#ifndef CU_MATH_H
#define CU_MATH_H

#include "utils.cuh"
#include <float.h>
#include <cuda_runtime_api.h>

__device__
vec3 cross_product(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.y*b.z - a.z*b.y;
    result.y = a.z*b.x - a.x*b.z;
    result.z = a.x*b.y - a.y*b.x;

    return result;
}

__device__
float fractf(const float x) {
    return x - floorf(x);
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
vec3 get_vector(const float x, const float y, const float z) {
    vec3 result;

    result.x = x;
    result.y = y;
    result.z = z;

    return result;
}

__device__
vec3 add_vector(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;

    return result;
}

__device__
vec3 sub_vector(const vec3 a, const vec3 b) {
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

__device__ __host__
vec3 scale_vector(vec3 vector, const float scale) {
    vector.x *= scale;
    vector.y *= scale;
    vector.z *= scale;

    return vector;
}

__device__ __host__
float get_length(const vec3 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

__device__
vec3 normalize_vector(vec3 vector) {
    const float scale = rnorm3df(vector.x, vector.y, vector.z);

    vector.x *= scale;
    vector.y *= scale;
    vector.z *= scale;

    return vector;
}

__device__
vec3 get_coordinates_in_triangle(const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 point) {
    const vec3 diff = sub_vector(point, vertex);
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
vec3 lerp_normals(const vec3 vertex_normal, const vec3 edge1_normal, const vec3 edge2_normal, const float lambda, const float mu, const vec3 face_normal) {
    vec3 result;

    result.x = vertex_normal.x + lambda * edge1_normal.x + mu * edge2_normal.x;
    result.y = vertex_normal.y + lambda * edge1_normal.y + mu * edge2_normal.y;
    result.z = vertex_normal.z + lambda * edge1_normal.z + mu * edge2_normal.z;

    const float length = get_length(result);

    return (length < eps) ? face_normal : scale_vector(result, 1.0f / length);
}

__device__
UV get_UV(const float u, const float v) {
    UV result;

    result.u = u;
    result.v = v;

    return result;
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

__device__
RGBF get_color(const float r, const float g, const float b) {
    RGBF result;

    result.r = r;
    result.g = g;
    result.b = b;

    return result;
}

__device__
RGBF add_color(const RGBF a, const RGBF b) {
    RGBF result;

    result.r = a.r + b.r;
    result.g = a.g + b.g;
    result.b = a.b + b.b;

    return result;
}

__device__
RGBF mul_color(const RGBF a, const RGBF b) {
    RGBF result;

    result.r = a.r * b.r;
    result.g = a.g * b.g;
    result.b = a.b * b.b;

    return result;
}

__device__
float linearRGB_to_SRGB(const float value) {
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    else {
        return 1.055f * powf(value, 0.416666666667f) - 0.055f;
    }
}

__device__
RGBF aces_tonemap(RGBF pixel) {
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;

  pixel.r = 1.25f * (pixel.r * (a * pixel.r + b)) / (pixel.r * (c * pixel.r + d) + e);
  pixel.g = 1.25f * (pixel.g * (a * pixel.g + b)) / (pixel.g * (c * pixel.g + d) + e);
  pixel.b = 1.25f * (pixel.b * (a * pixel.b + b)) / (pixel.b * (c * pixel.b + d) + e);

  return pixel;
}

__device__
float luminance(const RGBF v) {
    return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

__device__
RGBF reinhard_tonemap(RGBF pixel) {
  const float factor = 1.0f / (1.0f + luminance(pixel));
  pixel.r *= factor;
  pixel.g *= factor;
  pixel.b *= factor;

  return pixel;
}

__device__
RGBAF saturate_albedo(RGBAF color, float change) {
    const float max_value = fmaxf(color.r, fmaxf(color.g, color.b));
    const float min_value = fminf(color.r, fminf(color.g, color.b));
    const float diff = 0.01f + max_value - min_value;
    color.r = fmaxf(0.0f, color.r - change * ((max_value - color.r) / diff) * min_value);
    color.g = fmaxf(0.0f, color.g - change * ((max_value - color.g) / diff) * min_value);
    color.b = fmaxf(0.0f, color.b - change * ((max_value - color.b) / diff) * min_value);

    return color;
}

#endif /* CU_MATH_H */
