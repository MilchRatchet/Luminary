#ifndef CU_MATH_H
#define CU_MATH_H

#include <cuda_runtime_api.h>
#include <float.h>

#include "intrinsics.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ vec3 cross_product(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;

  return result;
}

__device__ float fractf(const float x) {
  return x - floorf(x);
}

__device__ float dot_product(const vec3 a, const vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float lerp(const float a, const float b, const float t) {
  return a + t * (b - a);
}

/*
 * Linearly remaps a value from one range to another.
 * @param value Value to map.
 * @param src_low Low of source range.
 * @param src_high High of source range.
 * @param dst_low Low of destination range.
 * @param dst_high High of destination range.
 * @result Remapped value.
 */
__device__ float remap(const float value, const float src_low, const float src_high, const float dst_low, const float dst_high) {
  return (value - src_low) / (src_high - src_low) * (dst_high - dst_low) + dst_low;
}

__device__ float remap01(const float value, const float src_low, const float src_high) {
  return __saturatef(remap(value, src_low, src_high, 0.0f, 1.0f));
}

__device__ float step(const float edge, const float x) {
  return (x < edge) ? 0.0f : 1.0f;
}

__device__ float smoothstep(const float x, const float edge0, const float edge1) {
  float t = remap01(x, edge0, edge1);
  // Surprisingly, this is almost equivalent on [0,1]
  // https://twitter.com/lisyarus/status/1600173486802014209
  // return 0.5f * (1.0f - cosf(PI * x));
  return t * t * (3.0f - 2.0f * t);
}

__device__ vec3 get_vector(const float x, const float y, const float z) {
  vec3 result;

  result.x = x;
  result.y = y;
  result.z = z;

  return result;
}

__device__ __host__ vec3 add_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;

  return result;
}

__device__ vec3 sub_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;

  return result;
}

__device__ vec3 mul_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x * b.x;
  result.y = a.y * b.y;
  result.z = a.z * b.z;

  return result;
}

__device__ vec3 min_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = fminf(a.x, b.x);
  result.y = fminf(a.y, b.y);
  result.z = fminf(a.z, b.z);

  return result;
}

__device__ vec3 max_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = fmaxf(a.x, b.x);
  result.y = fmaxf(a.y, b.y);
  result.z = fmaxf(a.z, b.z);

  return result;
}

__device__ vec3 inv_vector(const vec3 a) {
  vec3 result;

  result.x = 1.0f / a.x;
  result.y = 1.0f / a.y;
  result.z = 1.0f / a.z;

  return result;
}

__device__ vec3 add_vector_const(const vec3 x, const float y) {
  return add_vector(x, get_vector(y, y, y));
}

__device__ vec3 fract_vector(const vec3 x) {
  return get_vector(fractf(x.x), fractf(x.y), fractf(x.z));
}

__device__ vec3 floor_vector(const vec3 x) {
  return get_vector(floorf(x.x), floorf(x.y), floorf(x.z));
}

__device__ vec3 normalize_vector(vec3 vector) {
  const float scale = rnorm3df(vector.x, vector.y, vector.z);

  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

__device__ vec3 scale_vector(vec3 vector, const float scale) {
  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

__device__ vec3 reflect_vector(const vec3 ray, const vec3 normal) {
  const float dot   = dot_product(ray, normal);
  const vec3 result = sub_vector(ray, scale_vector(normal, 2.0f * dot));

  return normalize_vector(result);
}

__device__ float get_length(const vec3 vector) {
  return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

__device__ float2 get_coordinates_in_triangle(const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 point) {
  const vec3 diff   = sub_vector(point, vertex);
  const float d00   = dot_product(edge1, edge1);
  const float d01   = dot_product(edge1, edge2);
  const float d11   = dot_product(edge2, edge2);
  const float d20   = dot_product(diff, edge1);
  const float d21   = dot_product(diff, edge2);
  const float denom = 1.0f / (d00 * d11 - d01 * d01);

  return make_float2((d11 * d20 - d01 * d21) * denom, (d00 * d21 - d01 * d20) * denom);
}

__device__ vec3
  lerp_normals(const vec3 vertex_normal, const vec3 edge1_normal, const vec3 edge2_normal, const float2 coords, const vec3 face_normal) {
  vec3 result;

  result.x = vertex_normal.x + coords.x * edge1_normal.x + coords.y * edge2_normal.x;
  result.y = vertex_normal.y + coords.x * edge1_normal.y + coords.y * edge2_normal.y;
  result.z = vertex_normal.z + coords.x * edge1_normal.z + coords.y * edge2_normal.z;

  const float length = get_length(result);

  return (length < eps) ? face_normal : scale_vector(result, 1.0f / length);
}

__device__ UV get_UV(const float u, const float v) {
  UV result;

  result.u = u;
  result.v = v;

  return result;
}

__device__ UV lerp_uv(const UV vertex_texture, const UV edge1_texture, const UV edge2_texture, const float2 coords) {
  UV result;

  result.u = vertex_texture.u + coords.x * edge1_texture.u + coords.y * edge2_texture.u;
  result.v = vertex_texture.v + coords.x * edge1_texture.v + coords.y * edge2_texture.v;

  return result;
}

/*
 * Uses a orthonormal basis which is built as described in
 * T. Duff, J. Burgess, P. Christensen, C. Hery, A. Kensler, M. Liani, R. Villemin, _Building an Orthonormal Basis, Revisited_
 */
__device__ vec3 sample_hemisphere_basis(const float altitude, const float azimuth, const vec3 basis) {
  vec3 u1, u2;
  // Orthonormal basis building
  {
    const float sign = copysignf(1.0f, basis.z);
    const float a    = -1.0f / (sign + basis.z);
    const float b    = basis.x * basis.y * a;
    u1               = get_vector(1.0f + sign * basis.x * basis.x * a, sign * b, -sign * basis.x);
    u2               = get_vector(b, sign + basis.y * basis.y * a, -basis.y);
  }

  const float c1 = sinf(altitude) * cosf(azimuth);
  const float c2 = sinf(altitude) * sinf(azimuth);
  const float c3 = cosf(altitude);

  vec3 result;

  result.x = c1 * u1.x + c2 * u2.x + c3 * basis.x;
  result.y = c1 * u1.y + c2 * u2.y + c3 * basis.y;
  result.z = c1 * u1.z + c2 * u2.z + c3 * basis.z;

  return normalize_vector(result);
}

__device__ Mat3x3 create_basis(const vec3 basis) {
  const float sign = copysignf(1.0f, basis.z);
  const float a    = -1.0f / (sign + basis.z);
  const float b    = basis.x * basis.y * a;
  const vec3 u1    = get_vector(1.0f + sign * basis.x * basis.x * a, sign * b, -sign * basis.x);
  const vec3 u2    = get_vector(b, sign + basis.y * basis.y * a, -basis.y);

  Mat3x3 mat;

  mat.f11 = u1.x;
  mat.f21 = u1.y;
  mat.f31 = u1.z;
  mat.f12 = u2.x;
  mat.f22 = u2.y;
  mat.f32 = u2.z;
  mat.f13 = basis.x;
  mat.f23 = basis.y;
  mat.f33 = basis.z;

  return mat;
}

/*
 * This function samples a ray on the unit sphere.
 *
 * @param alpha Number between [-1,1]. To sample only the upper hemisphere use [0,1] and vice versa.
 * @param beta Number between [0,1].
 * @result Unit vector on the sphere given by the two parameters.
 */
__device__ vec3 sample_ray_sphere(const float alpha, const float beta) {
  if (fabsf(alpha) > 1.0f - eps) {
    return get_vector(0.0f, 0.0f, copysignf(1.0f, alpha));
  }

  const float a = sqrtf(1.0f - alpha * alpha);
  const float b = 2.0f * PI * beta;

  // How this works:
  // Standard way is a = acosf(alpha) and then return (sinf(a) * cosf(b), sinf(a) * sinf(b), cosf(a)).
  // What we can do instead is sample a point uniformly on the disk, i.e., x and y such that
  // sqrtf(x * x + y * y) = 1 through x = cosf(2.0f * PI * beta) and y = sinf(2.0f * PI * beta).
  // We have that z = alpha. We need a factor 'a' so that the direction, obtained by using the point on the disk
  // scaled by 'a' and using the z value alpha, is normalized. Hence: 1.0f = sqrtf(a * a * (x * x + y * y) + alpha * alpha).
  // Rearranging the terms yields: a = sqrtf((1.0f - alpha * alpha) / (x * x + y * y)).
  // Since (x * x + y * y) = 1, we obtain our formula.
  return get_vector(a * cosf(b), a * sinf(b), alpha);
}

__device__ int trailing_zeros(const unsigned int n) {
  return __clz(__brev(n));
}

__device__ Quaternion inverse_quaternion(const Quaternion q) {
  Quaternion result;
  result.x = -q.x;
  result.y = -q.y;
  result.z = -q.z;
  result.w = q.w;
  return result;
}

__device__ Quaternion get_rotation_to_z_canonical(const vec3 v) {
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

__device__ __host__ vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
  vec3 result;

  vec3 u;
  u.x = q.x;
  u.y = q.y;
  u.z = q.z;

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  vec3 cross;

  cross.x = u.y * v.z - u.z * v.y;
  cross.y = u.z * v.x - u.x * v.z;
  cross.z = u.x * v.y - u.y * v.x;

  result.x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x;
  result.y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y;
  result.z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z;

  return result;
}

__device__ vec4 transform_vec4(const Mat4x4 m, const vec4 p) {
  vec4 res;

  res.x = m.f11 * p.x + m.f12 * p.y + m.f13 * p.z + m.f14 * p.w;
  res.y = m.f21 * p.x + m.f22 * p.y + m.f23 * p.z + m.f24 * p.w;
  res.z = m.f31 * p.x + m.f32 * p.y + m.f33 * p.z + m.f34 * p.w;
  res.w = m.f41 * p.x + m.f42 * p.y + m.f43 * p.z + m.f44 * p.w;

  return res;
}

__device__ vec3 transform_vec4_3(const Mat4x4 m, const vec3 p) {
  vec3 res;

  res.x = m.f11 * p.x + m.f12 * p.y + m.f13 * p.z;
  res.y = m.f21 * p.x + m.f22 * p.y + m.f23 * p.z;
  res.z = m.f31 * p.x + m.f32 * p.y + m.f33 * p.z;

  return res;
}

__device__ vec3 transform_vec3(const Mat3x3 m, const vec3 p) {
  vec3 res;

  res.x = m.f11 * p.x + m.f12 * p.y + m.f13 * p.z;
  res.y = m.f21 * p.x + m.f22 * p.y + m.f23 * p.z;
  res.z = m.f31 * p.x + m.f32 * p.y + m.f33 * p.z;

  return res;
}

/*
 * Computes the distance to the first intersection of a ray with a sphere. To check for any hit use sphere_ray_hit.
 * This implementation is based on Chapter 7 in Ray Tracing Gems I, "Precision Improvements for Ray/Sphere Intersection".
 *
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @result Value t such that origin + t * ray is a point on the sphere.
 */
__device__ float sphere_ray_intersection(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
  const vec3 diff = sub_vector(origin, p);
  const float dot = dot_product(diff, ray);
  const float r2  = r * r;
  const float c   = dot_product(diff, diff) - r2;
  const vec3 k    = sub_vector(diff, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return FLT_MAX;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t0 = c / q;

  if (t0 >= 0.0f)
    return t0;

  const float t1 = q;
  return (t1 >= 0.0f) ? t1 : FLT_MAX;
}

/*
 * Computes the distance to the first intersection of a ray with a sphere with (0,0,0) as its center.
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result Value t such that origin + t * ray is a point on the sphere.
 */
__device__ float sph_ray_int_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return FLT_MAX;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);
  const float c  = dot_product(origin, origin) - r2;
  const float t0 = c / q;

  if (t0 >= 0.0f)
    return t0;

  const float t1 = q;
  return (t1 >= 0.0f) ? t1 : FLT_MAX;
}

/*
 * Computes whether a ray hits a sphere. To compute the distance see sphere_ray_intersection.
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @result 1 if the ray hits the sphere, 0 else.
 */
__device__ bool sphere_ray_hit(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
  const vec3 diff = sub_vector(origin, p);
  const float dot = dot_product(diff, ray);
  const float r2  = r * r;
  const float c   = dot_product(diff, diff) - r2;
  const vec3 k    = sub_vector(diff, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return false;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t0 = c / q;

  return (t0 >= 0.0f);
}

/*
 * Computes whether a ray hits a sphere with (0,0,0) as its center. To compute the distance see sph_ray_int_p0.
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result 1 if the ray hits the sphere, 0 else.
 */
__device__ bool sph_ray_hit_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return false;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);
  const float c  = dot_product(origin, origin) - r2;
  const float t0 = c / q;

  return (t0 >= 0.0f);
}

/*
 * Computes the distance to the last intersection of a ray with a sphere. To compute the first hit use sphere_ray_intersection.
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @result Value t such that origin + t * ray is a point on the sphere.
 */
__device__ float sphere_ray_intersect_back(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
  const vec3 diff = sub_vector(origin, p);
  const float dot = dot_product(diff, ray);
  const float r2  = r * r;
  const float c   = dot_product(diff, diff) - r2;
  const vec3 k    = sub_vector(diff, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return FLT_MAX;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t1 = q;

  if (t1 >= 0.0f)
    return t1;

  const float t0 = c / q;
  return (t0 >= 0.0f) ? t0 : FLT_MAX;
}

/*
 * Computes the distance to the last intersection of a ray with a sphere with (0,0,0) as its center.
 * @param ray Normalized ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result Value t such that origin + t * ray is a point on the sphere.
 */
__device__ float sph_ray_int_back_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = r2 - dot_product(k, k);

  if (d < 0.0f)
    return FLT_MAX;

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);
  const float c  = dot_product(origin, origin) - r2;
  const float t1 = q;

  if (t1 >= 0.0f)
    return t1;

  const float t0 = c / q;
  return (t0 >= 0.0f) ? t0 : FLT_MAX;
}

__device__ __host__ vec3 angles_to_direction(const float altitude, const float azimuth) {
  vec3 dir;
  dir.x = cosf(azimuth) * cosf(altitude);
  dir.y = sinf(altitude);
  dir.z = sinf(azimuth) * cosf(altitude);

  return dir;
}

// PBRT v3 Chapter "Specular Reflection and Transmission", Refract() function
__device__ vec3 refract_ray(const vec3 ray, const vec3 normal, const float index_ratio) {
  const float dot = -dot_product(normal, ray);

  const float b = 1.0f - index_ratio * index_ratio * (1.0f - dot * dot);

  if (b < 0.0f) {
    // Total reflection
    return reflect_vector(ray, normal);
  }
  else {
    return normalize_vector(add_vector(scale_vector(ray, index_ratio), scale_vector(normal, index_ratio * dot - sqrtf(b))));
  }
}

__device__ RGBF get_color(const float r, const float g, const float b) {
  RGBF result;

  result.r = r;
  result.g = g;
  result.b = b;

  return result;
}

__device__ RGBAF get_RGBAF(const float r, const float g, const float b, const float a) {
  RGBAF result;

  result.r = r;
  result.g = g;
  result.b = b;
  result.a = a;

  return result;
}

__device__ RGBF add_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r + b.r;
  result.g = a.g + b.g;
  result.b = a.b + b.b;

  return result;
}

__device__ RGBF sub_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r - b.r;
  result.g = a.g - b.g;
  result.b = a.b - b.b;

  return result;
}

__device__ RGBF mul_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r * b.r;
  result.g = a.g * b.g;
  result.b = a.b * b.b;

  return result;
}

__device__ RGBF scale_color(const RGBF a, const float b) {
  RGBF result;

  result.r = a.r * b;
  result.g = a.g * b;
  result.b = a.b * b;

  return result;
}

__device__ RGBF fma_color(const RGBF a, const float b, const RGBF c) {
  return add_color(c, scale_color(a, b));
}

__device__ RGBF min_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = fminf(a.r, b.r);
  result.g = fminf(a.g, b.g);
  result.b = fminf(a.b, b.b);

  return result;
}

__device__ RGBF max_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = fmaxf(a.r, b.r);
  result.g = fmaxf(a.g, b.g);
  result.b = fmaxf(a.b, b.b);

  return result;
}

__device__ RGBF inv_color(const RGBF a) {
  RGBF result;

  result.r = 1.0f / a.r;
  result.g = 1.0f / a.g;
  result.b = 1.0f / a.b;

  return result;
}

__device__ int color_any(const RGBF a) {
  return (a.r > 0.0f || a.g > 0.0f || a.b > 0.0f);
}

__device__ float RGBF_avg(const RGBF a) {
  return (a.r + a.g + a.b) * (1.0f / 3.0f);
}

__device__ RGBF opaque_color(const RGBAF a) {
  return get_color(a.r, a.g, a.b);
}

__device__ RGBAF RGBAF_set(const float r, const float g, const float b, const float a) {
  RGBAF result;

  result.r = r;
  result.g = g;
  result.b = b;
  result.a = a;

  return result;
}

__device__ float get_dithering(const int x, const int y) {
  const float dither = 2.0f * white_noise() - 1.0f;

  return copysignf(1.0f - sqrtf(1.0f - fabsf(dither)), dither);
}

//
// Linear RGB / sRGB conversion taken from https://www.shadertoy.com/view/lsd3zN
// which is based os D3DX implementations
//

__device__ float linearRGB_to_SRGB(const float value) {
  if (value <= 0.0031308f) {
    return 12.92f * value;
  }
  else {
    return 1.055f * powf(value, 0.416666666667f) - 0.055f;
  }
}

__device__ float SRGB_to_linearRGB(const float value) {
  if (value <= 0.04045f) {
    return value / 12.92f;
  }
  else {
    return powf((value + 0.055f) / 1.055f, 2.4f);
  }
}

__device__ float luminance(const RGBF v) {
  return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

__device__ RGBAF saturate_albedo(RGBAF color, float change) {
  const float max_value = fmaxf(color.r, fmaxf(color.g, color.b));
  const float min_value = fminf(color.r, fminf(color.g, color.b));
  const float diff      = 0.01f + max_value - min_value;
  color.r               = fmaxf(0.0f, color.r - change * ((max_value - color.r) / diff) * min_value);
  color.g               = fmaxf(0.0f, color.g - change * ((max_value - color.g) / diff) * min_value);
  color.b               = fmaxf(0.0f, color.b - change * ((max_value - color.b) / diff) * min_value);

  return color;
}

__device__ RGBF filter_gray(RGBF color) {
  const float value = luminance(color);

  return get_color(value, value, value);
}

__device__ RGBF filter_sepia(RGBF color) {
  return get_color(
    color.r * 0.393f + color.g * 0.769f + color.b * 0.189f, color.r * 0.349f + color.g * 0.686f + color.b * 0.168f,
    color.r * 0.272f + color.g * 0.534f + color.b * 0.131f);
}

__device__ RGBF filter_gameboy(RGBF color, int x, int y) {
  const float value = 2550.0f * luminance(color);

  const float dither = get_dithering(x, y);

  const int tone = (int) ((32.0f + value - dither * 64.0f) / 64.0f);

  switch (tone) {
    case 0:
      return get_color(15.0f / 255.0f, 56.0f / 255.0f, 15.0f / 255.0f);
    case 1:
      return get_color(48.0f / 255.0f, 98.0f / 255.0f, 48.0f / 255.0f);
    case 2:
      return get_color(139.0f / 255.0f, 172.0f / 255.0f, 15.0f / 255.0f);
    case 3:
    default:
      return get_color(155.0f / 255.0f, 188.0f / 255.0f, 15.0f / 255.0f);
  }
}

__device__ RGBF filter_2bitgray(RGBF color, int x, int y) {
  const float value = 2550.0f * luminance(color);

  const float dither = get_dithering(x, y);

  const int tone = (int) ((32.0f + value - dither * 64.0f) / 64.0f);

  switch (tone) {
    case 0:
      return get_color(0.0f, 0.0f, 0.0f);
    case 1:
      return get_color(85.0f / 255.0f, 85.0f / 255.0f, 85.0f / 255.0f);
    case 2:
      return get_color(170.0f / 255.0f, 170.0f / 255.0f, 170.0f / 255.0f);
    case 3:
    default:
      return get_color(1.0f, 1.0f, 1.0f);
  }
}

__device__ RGBF filter_crt(RGBF color, int x, int y) {
  color = scale_color(color, 1.5f);

  const int row = y % 3;

  switch (row) {
    case 0:
      color.r = 0.0f;
      color.g = 0.0f;
      break;
    case 1:
      color.g = 0.0f;
      color.b = 0.0f;
      break;
    case 2:
      color.r = 0.0f;
      color.b = 0.0f;
      break;
  }

  return color;
}

__device__ RGBF filter_blackwhite(RGBF color, int x, int y) {
  const float value = 2550.0f * luminance(color);

  const float dither = get_dithering(x, y);

  const int tone = (int) ((64.0f + value - dither * 128.0f) / 128.0f);

  switch (tone) {
    case 0:
      return get_color(0.0f, 0.0f, 0.0f);
    case 1:
    default:
      return get_color(1.0f, 1.0f, 1.0f);
  }
}

__device__ float henyey_greenstein_phase_function(const float cos_angle, const float g) {
  return (1.0f - g * g) / (4.0f * PI * powf(1.0f + g * g - 2.0f * g * cos_angle, 1.5f));
}

__device__ float draine_phase_function(const float cos_angle, const float g, const float alpha) {
  return henyey_greenstein_phase_function(cos_angle, g)
         * ((1.0f + alpha * cos_angle * cos_angle) / (1.0f + (alpha / 3.0f) * (1.0f + 2.0f * g * g)));
}

struct JendersieEonParams {
  float g_hg;
  float g_d;
  float alpha;
  float w_d;
} typedef JendersieEonParams;

__device__ JendersieEonParams jendersie_eon_phase_parameters(const float diameter, const float ms_factor) {
  JendersieEonParams params;

  // Renaming to a shorter name.
  const float d = diameter;

  // TODO: Allow precomputation of these so that the actual functions take these parameters as arguments.

  if (d >= 5.0f && d <= 50.0f) {
    params.g_hg  = expf(-0.0990567f / (d - 1.67154f));
    params.g_d   = expf(-(2.20679f / (d + 3.91029f)) - 0.428934f);
    params.alpha = expf(3.62489f - (8.29288f / (d + 5.52825f)));
    params.w_d   = expf(-(0.599085f / (d - 0.641583f)) - 0.665888f);
  }
  else if (d >= 1.5f && d < 5.0f) {
    params.g_hg  = 0.0604931f * logf(logf(d)) + 0.940256f;
    params.g_d   = 0.500411f - (0.081287f / (-2.0f * logf(d) + tanf(logf(d)) + 1.27551f));
    params.alpha = 7.30354f * logf(d) + 6.31675f;
    params.w_d   = 0.026914f * (logf(d) - cosf(5.68947f * (logf(logf(d)) - 0.0292149f))) + 0.376475f;
  }
  else if (d >= 0.1f && d < 1.5f) {
    params.g_hg = 0.862f - 0.143f * logf(d) * logf(d);
    params.g_d  = 0.379685f
                   * cosf(
                     1.19692f * cosf(((logf(d) - 0.238604f) * (logf(d) + 1.00667f)) / (0.507522f - 0.15677f * logf(d))) + 1.37932f * logf(d)
                     + 0.0625835f)
                 + 0.344213f;
    params.alpha = 250.0f;
    params.w_d   = 0.146209f * cosf(3.38707f * logf(d) + 2.11193f) + 0.316072f + 0.0778917f * logf(d);
  }
  else if (d < 0.1f) {
    params.g_hg  = 13.8f * d * d;
    params.g_d   = 1.1456f * d * sinf(9.29044f * d);
    params.alpha = 250.0f;
    params.w_d   = 0.252977f - 312.983f * powf(d, 4.3f);
  }

  params.g_hg *= ms_factor;
  params.g_d *= ms_factor;

  return params;
}

/*
 * Fog phase function from [JenE23].
 *
 * [JenE23] J. Jendersie and E. d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks, 2023.
 *
 * @param diameter Diameter of water droplets in [5,50] in micrometer.
 */
__device__ float jendersie_eon_phase_function(const float cos_angle, const float diameter, const float ms_factor = 1.0f) {
  JendersieEonParams params = jendersie_eon_phase_parameters(diameter, ms_factor);

  const float phase_hg = henyey_greenstein_phase_function(cos_angle, params.g_hg);
  const float phase_d  = draine_phase_function(cos_angle, params.g_d, params.alpha);

  return (1.0f - params.w_d) * phase_hg + params.w_d * phase_d;
}

/*
 * Uses a orthonormal basis which is built as described in
 * T. Duff, J. Burgess, P. Christensen, C. Hery, A. Kensler, M. Liani, R. Villemin, _Building an Orthonormal Basis, Revisited_
 */
__device__ vec3 phase_sample_basis(const float alpha, const float beta, const vec3 basis) {
  const float sign = copysignf(1.0f, basis.z);
  const float a    = -1.0f / (sign + basis.z);
  const float b    = basis.x * basis.y * a;
  const vec3 u1    = get_vector(1.0f + sign * basis.x * basis.x * a, sign * b, -sign * basis.x);
  const vec3 u2    = get_vector(b, sign + basis.y * basis.y * a, -basis.y);

  const vec3 s = sample_ray_sphere(alpha, beta);

  vec3 result;
  result.x = s.x * u1.x + s.y * u2.x + s.z * basis.x;
  result.y = s.x * u1.y + s.y * u2.y + s.z * basis.y;
  result.z = s.x * u1.z + s.y * u2.z + s.z * basis.z;

  return normalize_vector(result);
}

__device__ float henyey_greenstein_phase_sample(const float g, const float r) {
  const float g2 = g * g;

  const float t = (1.0f - g2) / (1.0f - g + 2.0f * g * r);

  // Note that the Jendersie-Eon paper supplement claims only t, but t^2 is correct according to all other sources.
  return (1.0f + g2 - t * t) / (2.0f * g);
}

__device__ float draine_phase_sample(const float g, const float alpha, const float r) {
  const float g2 = g * g;
  const float g4 = g2 * g2;

  const float t0 = alpha - alpha * g2;
  const float t1 = alpha * g4 - alpha;
  const float t2 = -3.0f * (4.0f * (g4 - g2) + t1 * (1.0f + g2));
  const float t3 = g * (2.0f * r - 1.0f);
  const float t4 = 3.0f * g2 * (1.0f + t3) + alpha * (2.0f + g2 * (1.0f + (1.0f + 2.0f * g2) * t3));
  const float t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
  const float t6 = t0 * 4.0f * (g4 - g2);
  const float t7 = powf(t5 + sqrtf(t5 * t5 - t6 * t6 * t6), 1.0f / 3.0f);
  const float t8 = 2.0f * ((t1 + (t6 / t7) + t7) / t0);
  const float t9 = sqrtf(6.0f * (1.0f + g2) + t8);

  const float h = sqrtf(6.0f * (1.0f + g2) - t8 + 8.0f * t4 / (t0 * t9)) - t9;

  return 0.5f * g + ((1.0f / (2.0f * g)) - (1.0f / (8.0f * g)) * (h * h));
}

__device__ float4 texture_load(const DeviceTexture tex, const UV uv) {
  float4 v = tex2D<float4>(tex.tex, uv.u, 1.0f - uv.v);

  v.x = powf(v.x, tex.gamma);
  v.y = powf(v.y, tex.gamma);
  v.z = powf(v.z, tex.gamma);
  // Gamma is never applied to the alpha of a texture according to PNG standard.

  return v;
}

/*
 * This function perfectly importance samples the phase function from [JenE23]. The details of this are presented in the
 * supplement of the paper.
 *
 * [JenE23] J. Jendersie and E. d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks, 2023.
 *
 * @param diameter Diameter of water droplets in [5,50] in micrometer.
 */
__device__ vec3 jendersie_eon_phase_sample(const vec3 ray, const float diameter, const float ms_factor = 1.0f) {
  const float r = white_noise();

  JendersieEonParams params = jendersie_eon_phase_parameters(diameter, ms_factor);

  float u;
  if (white_noise() < params.w_d) {
    u = draine_phase_sample(params.g_d, params.alpha, r);
  }
  else {
    u = henyey_greenstein_phase_sample(params.g_hg, r);
  }

  return phase_sample_basis(u, white_noise(), ray);
}

__device__ float bvh_triangle_intersection(const TraversalTriangle triangle, const vec3 origin, const vec3 ray) {
  const vec3 h  = cross_product(ray, triangle.edge2);
  const float a = dot_product(triangle.edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, triangle.vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, triangle.edge1);
  const float v = f * dot_product(ray, q);

  if (v < 0.0f || u < 0.0f || u + v > 1.0f)
    return FLT_MAX;

  const float t = f * dot_product(triangle.edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

__device__ float bvh_triangle_intersection_uv(const TraversalTriangle triangle, const vec3 origin, const vec3 ray, float2& coords) {
  const vec3 h  = cross_product(ray, triangle.edge2);
  const float a = dot_product(triangle.edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, triangle.vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, triangle.edge1);
  const float v = f * dot_product(ray, q);

  coords = make_float2(u, v);

  //  The third check is inverted to catch NaNs since NaNs always return false, the not will turn it into a true
  if (v < 0.0f || u < 0.0f || !(u + v <= 1.0f))
    return FLT_MAX;

  const float t = f * dot_product(triangle.edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

__device__ float clampf(const float x, const float a, const float b) {
  return fminf(b, fmaxf(a, x));
}

/*
 * Computes solid angle when sampling a triangle using the formula in "The Solid Angle of a Plane Triangle" (1983) by A. Van Oosterom and J.
 * Strackee.
 * @param triangle Triangle.
 * @param origin Point from which you sample.
 * @param normal Normalized normal of surface from which you sample.
 * @result Solid angle of triangle.
 */
__device__ float sample_triangle_solid_angle(const TriangleLight triangle, const vec3 origin) {
  vec3 a       = sub_vector(triangle.vertex, origin);
  const vec3 b = normalize_vector(add_vector(triangle.edge1, a));
  const vec3 c = normalize_vector(add_vector(triangle.edge2, a));
  a            = normalize_vector(a);

  const float num   = fabsf(dot_product(a, cross_product(b, c)));
  const float denom = 1.0f + dot_product(a, b) + dot_product(a, c) + dot_product(b, c);

  return 4.0f * atan2f(num, denom);
}

/*
 * A more numerically stable version of normalize(b-a).
 * Whether this is actually is any better, I don't know, I just pretend.
 */
__device__ vec3 vector_direction_stable(vec3 a, vec3 b) {
  const float len_a = get_length(a);
  const float len_b = get_length(b);

  if (len_a == 0.0f && len_b == 0.0f) {
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  const float max_len = fminf(len_a, len_b);

  a = scale_vector(a, 1.0f / max_len);
  b = scale_vector(b, 1.0f / max_len);

  vec3 d = sub_vector(a, b);

  return normalize_vector(d);
}

/*
 * Surface sample a triangle and collect a weight as if it was solid angle sampled.
 * @param triangle Triangle.
 * @param origin Point to sample from.
 * @param area Solid angle of the triangle.
 * @result Normalized direction to the point on the triangle.
 *
 * Robust triangle sampling.
 */
__device__ vec3 sample_triangle(const TriangleLight triangle, const vec3 origin, float& area, uint32_t& seed) {
  float r1 = sqrtf(white_noise_offset_restir(seed++));
  float r2 = white_noise_offset_restir(seed++);

  // We use solid angle when we shouldn't. However, I don't want to change everything now just because
  // there is this little bias in the triangle light sampling weight. Instead, I would like to solid
  // angle sample the triangle lights in the future. I once did a test which failed due to numerical issues.
  // The NaNs are controllable but the directions that are computed are just not reliable enough, too many
  // end up missing the target light. Given how much we need to compensate even for surface sampling,
  // it is reasonable to think that this may never be possible unless double precision numbers become
  // useful on end user GPUs.
  area = sample_triangle_solid_angle(triangle, origin);

  if (isnan(area) || area < 1e-7f) {
    area = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  // Map random numbers uniformly into [0.025,0.975].
  r1 = 0.025f + 0.95f * r1;
  r2 = 0.025f + 0.95f * r2;

  float u = 1.0f - r1;
  float v = r1 * r2;

  const vec3 p = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, u), scale_vector(triangle.edge2, v)));

  return vector_direction_stable(p, origin);
}

/*
 * Samples the solid angle of a sphere.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @param origin Point to sample from.
 * @result Normalized direction to the point on the sphere.
 */
__device__ vec3 sample_sphere(const vec3 p, const float r, const vec3 origin, float& area, uint32_t& seed) {
  float r1 = white_noise_offset_restir(seed++);
  float r2 = white_noise_offset_restir(seed++);

  vec3 dir      = sub_vector(p, origin);
  const float d = get_length(dir);

  if (d < r) {
    return normalize_vector(sample_ray_sphere(2.0f * r1 - 1.0f, r2));
  }

  // Map random numbers uniformly into [0.0,0.999].
  r1 = 0.999f * r1;
  r2 = 0.999f * r2;

  dir = scale_vector(dir, 1.0f / d);

  const float angle = asinf(__saturatef(r / d));

  area = 2.0f * PI * angle * angle;

  const float u = sqrtf(r1) * angle;
  const float v = 2.0f * PI * r2;

  return normalize_vector(sample_hemisphere_basis(u, v, dir));
}

/*
 * Computes solid angle when sampling a sphere.
 * @param p Center of sphere.
 * @param r Radius of sphere.
 * @param origin Point from which you sample.
 * @param normal Normalized normal of surface from which you sample.
 * @result Solid angle of sphere.
 */
__device__ float sample_sphere_solid_angle(const vec3 p, const float r, const vec3 origin) {
  vec3 dir      = sub_vector(p, origin);
  const float d = get_length(dir);

  if (d < r)
    return 2.0f * PI;

  const float a = asinf(r / d);

  return 2.0f * PI * a * a;
}

__device__ int material_is_mirror(const float roughness, const float metallic) {
  return (roughness < 0.9f && metallic > 0.9f);
}

// Computes tangent space for use with normal mapping without precomputation
// http://www.thetenthplanet.de/archives/1180
__device__ Mat3x3 cotangent_frame(vec3 normal, vec3 e1, vec3 e2, UV t1, UV t2) {
  e1 = normalize_vector(e1);
  e2 = normalize_vector(e2);

  const float abs1 = __frsqrt_rn(t1.u * t1.u + t1.v * t1.v);
  if (!isinf(abs1)) {
    t1.u *= abs1;
    t1.v *= abs1;
  }
  const float abs2 = __frsqrt_rn(t2.u * t2.u + t2.v * t2.v);
  if (!isinf(abs2)) {
    t2.u *= abs2;
    t2.v *= abs2;
  }

  const vec3 a1 = cross_product(e2, normal);
  const vec3 a2 = cross_product(normal, e1);

  vec3 T = add_vector(scale_vector(a1, t1.u), scale_vector(a2, t2.u));
  vec3 B = add_vector(scale_vector(a1, t1.v), scale_vector(a2, t2.v));

  const float invmax = __frsqrt_rn(fmaxf(dot_product(T, T), dot_product(B, B)));

  T = scale_vector(T, invmax);
  B = scale_vector(B, invmax);

  Mat3x3 mat;
  mat.f11 = T.x;
  mat.f21 = T.y;
  mat.f31 = T.z;
  mat.f12 = B.x;
  mat.f22 = B.y;
  mat.f32 = B.z;
  mat.f13 = normal.x;
  mat.f23 = normal.y;
  mat.f33 = normal.z;

  return mat;
}

/////////////////
// RGBAhalf functions
/////////////////

__device__ RGBAhalf RGBF_to_RGBAhalf(const RGBF a) {
  RGBAhalf result;

  result.rg   = __floats2half2_rn(a.r, a.g);
  result.ba.x = __float2half(a.b);
  result.ba.y = (__half) 0.0f;

  return result;
}

__device__ RGBF RGBAhalf_to_RGBF(const RGBAhalf a) {
  RGBF result;

  const float2 rg = __half22float2(a.rg);

  result.r = rg.x;
  result.g = rg.y;
  result.b = __half2float(a.ba.x);

  return result;
}

__device__ RGBAhalf RGBAF_to_RGBAhalf(const RGBAF a) {
  RGBAhalf result;

  result.rg = __floats2half2_rn(a.r, a.g);
  result.ba = __floats2half2_rn(a.b, a.a);

  return result;
}

__device__ RGBAF RGBAhalf_to_RGBAF(const RGBAhalf a) {
  RGBAF result;

  const float2 rg = __half22float2(a.rg);
  const float2 ba = __half22float2(a.ba);

  result.r = rg.x;
  result.g = rg.y;
  result.b = ba.x;
  result.a = ba.y;

  return result;
}

__device__ RGBAhalf scale_RGBAhalf(const RGBAhalf a, const __half b) {
  RGBAhalf result;

  __half2 s = make_half2(b, b);

  result.rg = __hmul2(a.rg, s);
  result.ba = __hmul2(a.ba, s);

  return result;
}

__device__ RGBAhalf add_RGBAhalf(const RGBAhalf a, const RGBAhalf b) {
  RGBAhalf result;

  result.rg = __hadd2(a.rg, b.rg);
  result.ba = __hadd2(a.ba, b.ba);

  return result;
}

__device__ RGBAhalf sub_RGBAhalf(const RGBAhalf a, const RGBAhalf b) {
  RGBAhalf result;

  result.rg = __hsub2(a.rg, b.rg);
  result.ba = __hsub2(a.ba, b.ba);

  return result;
}

__device__ RGBAhalf mul_RGBAhalf(const RGBAhalf a, const RGBAhalf b) {
  RGBAhalf result;

  result.rg = __hmul2(a.rg, b.rg);
  result.ba = __hmul2(a.ba, b.ba);

  return result;
}

__device__ RGBAhalf max_RGBAhalf(const RGBAhalf a, const RGBAhalf b) {
  RGBAhalf result;

  result.rg = __hmax2(a.rg, b.rg);
  result.ba = __hmax2(a.ba, b.ba);

  return result;
}

__device__ RGBAhalf min_RGBAhalf(const RGBAhalf a, const RGBAhalf b) {
  RGBAhalf result;

  result.rg = __hmin2(a.rg, b.rg);
  result.ba = __hmin2(a.ba, b.ba);

  return result;
}

__device__ float hmax_RGBAhalf(const RGBAhalf a) {
  const __half2 max = __hmax2(a.rg, a.ba);

  return fmaxf(__high2float(max), __low2float(max));
}

__device__ RGBAhalf fma_RGBAhalf(const RGBAhalf a, const __half b, const RGBAhalf c) {
  RGBAhalf result;

  __half2 s = make_half2(b, b);

  result.rg = __hfma2(a.rg, s, c.rg);
  result.ba = __hfma2(a.ba, s, c.ba);

  return result;
}

__device__ RGBAhalf sqrt_RGBAhalf(const RGBAhalf a) {
  RGBAhalf result;

  result.rg = h2sqrt(a.rg);
  result.ba = h2sqrt(a.ba);

  return result;
}

__device__ bool any_RGBAhalf(const RGBAhalf a) {
  __half2 zero = make_half2(0.0f, 0.0f);

  return !(__hbeq2(a.rg, zero) && __hbeq2(a.ba, zero));
}

/*
 * This bounds the RGBAhalf to values of magnitude less than 6500.0f
 * This is slow!
 */
__device__ RGBAhalf bound_RGBAhalf(const RGBAhalf a) {
  RGBAhalf result;

  int inf_r = __hisinf(a.rg.x) | __hisnan(a.rg.x);
  int inf_g = __hisinf(a.rg.y) | __hisnan(a.rg.y);
  int inf_b = __hisinf(a.ba.x) | __hisnan(a.ba.x);
  int inf_a = __hisinf(a.ba.y) | __hisnan(a.ba.y);

  result.rg.x = (inf_r) ? ((__half) (6500.0f * (float) inf_r)) : a.rg.x;
  result.rg.y = (inf_g) ? ((__half) (6500.0f * (float) inf_g)) : a.rg.y;
  result.ba.x = (inf_b) ? ((__half) (6500.0f * (float) inf_b)) : a.ba.x;
  result.ba.y = (inf_a) ? ((__half) (6500.0f * (float) inf_a)) : a.ba.y;

  return result;
}

__device__ int infnan_RGBAhalf(const RGBAhalf a) {
  int inf_r = __hisinf(a.rg.x) || __hisnan(a.rg.x);
  int inf_g = __hisinf(a.rg.y) || __hisnan(a.rg.y);
  int inf_b = __hisinf(a.ba.x) || __hisnan(a.ba.x);
  int inf_a = __hisinf(a.ba.y) || __hisnan(a.ba.y);

  return inf_r || inf_g || inf_b || inf_a;
}

__device__ RGBAhalf get_RGBAhalf(const float r, const float g, const float b, const float a) {
  RGBAhalf result;

  result.rg = make_half2((__half) r, (__half) g);
  result.ba = make_half2((__half) b, (__half) a);

  return result;
}

__device__ RGBF rgb_to_hsv(const RGBF rgb) {
  const float max_value = fmaxf(rgb.r, fmaxf(rgb.g, rgb.b));
  const float min_value = fminf(rgb.r, fminf(rgb.g, rgb.b));

  const float v = max_value;
  const float s = (max_value - min_value) / max_value;

  float h = 0.0f;
  if (s != 0.0f) {
    const float delta = max_value - min_value;
    if (max_value == rgb.r) {
      h = (rgb.g - rgb.b) / delta;
    }
    else if (max_value == rgb.g) {
      h = 2.0f + (rgb.b - rgb.r) / delta;
    }
    else {
      h = 4.0f + (rgb.r - rgb.g) / delta;
    }

    h *= 1.0f / 6.0f;

    if (h < 0.0f) {
      h += 1.0f;
    }
  }

  return get_color(h, s, v);
}

//
// Taken from the shader toy https://www.shadertoy.com/view/lsS3Wc by Inigo Quilez
//
__device__ RGBF hsv_to_rgb(RGBF hsv) {
  const float s = hsv.g;
  const float v = hsv.b;

  if (s == 0.0f) {
    return get_color(v, v, v);
  }

  const float h = hsv.r * 6.0f;

  RGBF hue = get_color(h, h, h);
  hue      = add_color(hue, get_color(0.0f, 4.0f, 2.0f));
  hue.r    = fmodf(hue.r, 6.0f);
  hue.g    = fmodf(hue.g, 6.0f);
  hue.b    = fmodf(hue.b, 6.0f);
  hue      = add_color(hue, get_color(-3.0f, -3.0f, -3.0f));
  hue.r    = fabsf(hue.r);
  hue.g    = fabsf(hue.g);
  hue.b    = fabsf(hue.b);
  hue      = add_color(hue, get_color(-1.0f, -1.0f, -1.0f));
  hue.r    = __saturatef(hue.r);
  hue.g    = __saturatef(hue.g);
  hue.b    = __saturatef(hue.b);

  return scale_color(add_color(scale_color(get_color(1.0f, 1.0f, 1.0f), 1.0f - s), scale_color(hue, s)), v);
}

#endif /* CU_MATH_H */
