#ifndef CU_MATH_H
#define CU_MATH_H

#include <float.h>

#include "intrinsics.cuh"
#include "random.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Math
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION float difference_of_products(const float a, const float b, const float c, const float d) {
  const float cd = c * d;

  const float err = fmaf(-c, d, cd);
  const float dop = fmaf(a, b, -cd);

  return dop + err;
}

LUMINARY_FUNCTION vec3 cross_product(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = difference_of_products(a.y, b.z, a.z, b.y);
  result.y = difference_of_products(a.z, b.x, a.x, b.z);
  result.z = difference_of_products(a.x, b.y, a.y, b.x);

  return result;
}

LUMINARY_FUNCTION float fractf(const float x) {
  return x - floorf(x);
}

LUMINARY_FUNCTION float dot_product(const vec3 a, const vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

LUMINARY_FUNCTION float lerp(const float a, const float b, const float t) {
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
LUMINARY_FUNCTION float remap(const float value, const float src_low, const float src_high, const float dst_low, const float dst_high) {
  return (value - src_low) / (src_high - src_low) * (dst_high - dst_low) + dst_low;
}

LUMINARY_FUNCTION float remap01(const float value, const float src_low, const float src_high) {
  return __saturatef(remap(value, src_low, src_high, 0.0f, 1.0f));
}

LUMINARY_FUNCTION float step(const float edge, const float x) {
  return (x < edge) ? 0.0f : 1.0f;
}

LUMINARY_FUNCTION float smoothstep(const float x, const float edge0, const float edge1) {
  float t = remap01(x, edge0, edge1);
  // Surprisingly, this is almost equivalent on [0,1]
  // https://twitter.com/lisyarus/status/1600173486802014209
  // return 0.5f * (1.0f - cosf(PI * x));
  return t * t * (3.0f - 2.0f * t);
}

// (exp(x) - 1)/x with cancellation of rounding errors.
LUMINARY_FUNCTION float expm1_over_x(const float x) {
  const float u = expf(x);

  if (u == 1.0f) {
    return 1.0f;
  }

  const float y = u - 1.0f;
  const float z = (fabsf(x) < 1.0f) ? logf(u) : x;

  return y / z;
}

////////////////////////////////////////////////////////////////////
// Vector API
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION vec3 get_vector(const float x, const float y, const float z) {
  vec3 result;

  result.x = x;
  result.y = y;
  result.z = z;

  return result;
}

LUMINARY_FUNCTION vec3 add_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;

  return result;
}

LUMINARY_FUNCTION vec3 sub_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;

  return result;
}

LUMINARY_FUNCTION vec3 mul_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x * b.x;
  result.y = a.y * b.y;
  result.z = a.z * b.z;

  return result;
}

LUMINARY_FUNCTION vec3 min_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = fminf(a.x, b.x);
  result.y = fminf(a.y, b.y);
  result.z = fminf(a.z, b.z);

  return result;
}

LUMINARY_FUNCTION vec3 max_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = fmaxf(a.x, b.x);
  result.y = fmaxf(a.y, b.y);
  result.z = fmaxf(a.z, b.z);

  return result;
}

LUMINARY_FUNCTION vec3 inv_vector(const vec3 a) {
  vec3 result;

  result.x = 1.0f / a.x;
  result.y = 1.0f / a.y;
  result.z = 1.0f / a.z;

  return result;
}

LUMINARY_FUNCTION vec3 add_vector_const(const vec3 x, const float y) {
  return add_vector(x, get_vector(y, y, y));
}

LUMINARY_FUNCTION vec3 fract_vector(const vec3 x) {
  return get_vector(fractf(x.x), fractf(x.y), fractf(x.z));
}

LUMINARY_FUNCTION vec3 floor_vector(const vec3 x) {
  return get_vector(floorf(x.x), floorf(x.y), floorf(x.z));
}

LUMINARY_FUNCTION vec3 normalize_vector(vec3 vector) {
  const float scale = rnorm3df(vector.x, vector.y, vector.z);

  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

LUMINARY_FUNCTION vec3 scale_vector(vec3 vector, const float scale) {
  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

LUMINARY_FUNCTION vec3 reflect_vector(const vec3 V, const vec3 normal) {
  const float dot   = dot_product(V, normal);
  const vec3 result = sub_vector(scale_vector(normal, 2.0f * dot), V);

  return normalize_vector(result);
}

LUMINARY_FUNCTION float get_length(const vec3 vector) {
  return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

LUMINARY_FUNCTION float2 get_coordinates_in_triangle(const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 point) {
  const vec3 diff   = sub_vector(point, vertex);
  const float d00   = dot_product(edge1, edge1);
  const float d01   = dot_product(edge1, edge2);
  const float d11   = dot_product(edge2, edge2);
  const float d20   = dot_product(diff, edge1);
  const float d21   = dot_product(diff, edge2);
  const float denom = 1.0f / (d00 * d11 - d01 * d01);

  return make_float2((d11 * d20 - d01 * d21) * denom, (d00 * d21 - d01 * d20) * denom);
}

LUMINARY_FUNCTION vec3
  lerp_normals(const vec3 vertex_normal, const vec3 edge1_normal, const vec3 edge2_normal, const float2 coords, const vec3 face_normal) {
  vec3 result;

  result.x = vertex_normal.x + coords.x * edge1_normal.x + coords.y * edge2_normal.x;
  result.y = vertex_normal.y + coords.x * edge1_normal.y + coords.y * edge2_normal.y;
  result.z = vertex_normal.z + coords.x * edge1_normal.z + coords.y * edge2_normal.z;

  const float length = get_length(result);

  return (length < eps) ? face_normal : scale_vector(result, 1.0f / length);
}

LUMINARY_FUNCTION UV get_uv(const float u, const float v) {
  UV result;

  result.u = u;
  result.v = v;

  return result;
}

LUMINARY_FUNCTION UV add_uv(const UV a, const UV b) {
  UV uv;

  uv.u = a.u + b.u;
  uv.v = a.v + b.v;

  return uv;
}

LUMINARY_FUNCTION UV lerp_uv(const UV vertex_texture, const UV vertex1_texture, const UV vertex2_texture, const float2 coords) {
  UV result;

  result.u = vertex_texture.u + coords.x * (vertex1_texture.u - vertex_texture.u) + coords.y * (vertex2_texture.u - vertex_texture.u);
  result.v = vertex_texture.v + coords.x * (vertex1_texture.v - vertex_texture.v) + coords.y * (vertex2_texture.v - vertex_texture.v);

  return result;
}

LUMINARY_FUNCTION UV uv_sub(const UV a, const UV b) {
  UV uv;

  uv.u = a.u - b.u;
  uv.v = a.v - b.v;

  return uv;
}

LUMINARY_FUNCTION UV uv_scale(const UV a, const float b) {
  UV uv;

  uv.u = a.u * b;
  uv.v = a.v * b;

  return uv;
}

/*
 * Uses a orthonormal basis which is built as described in
 * T. Duff, J. Burgess, P. Christensen, C. Hery, A. Kensler, M. Liani, R. Villemin, _Building an Orthonormal Basis, Revisited_
 */
LUMINARY_FUNCTION vec3 sample_hemisphere_basis(const float altitude, const float azimuth, const vec3 basis) {
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

LUMINARY_FUNCTION Mat3x3 create_basis(const vec3 basis) {
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
 * This function samples a ray on the unit sphere. Up is Z+.
 *
 * @param alpha Number between [-1,1]. To sample only the upper hemisphere use [0,1] and vice versa.
 * @param beta Number between [0,1].
 * @result Unit vector on the sphere given by the two parameters.
 */
LUMINARY_FUNCTION vec3 sample_ray_sphere(const float alpha, const float beta) {
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

LUMINARY_FUNCTION int trailing_zeros(const unsigned int n) {
  return __clz(__brev(n));
}

LUMINARY_FUNCTION Quaternion normalize_quaternion(const Quaternion q) {
  const float length = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);

  Quaternion res;

  res.x = q.x / length;
  res.y = q.y / length;
  res.z = q.z / length;
  res.w = q.w / length;

  return res;
}

LUMINARY_FUNCTION Quaternion quaternion_inverse(const Quaternion q) {
  Quaternion result;
  result.x = -q.x;
  result.y = -q.y;
  result.z = -q.z;
  result.w = q.w;
  return result;
}

LUMINARY_FUNCTION Quaternion quaternion_rotation_to_z_canonical(const vec3 v) {
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

  const float norm = rnorm3df(res.x, res.y, res.w);

  res.x *= norm;
  res.y *= norm;
  res.w *= norm;

  return res;
}

LUMINARY_FUNCTION vec3 quaternion_apply(const Quaternion q, const vec3 v) {
  const vec3 u  = get_vector(q.x, q.y, q.z);
  const float s = q.w;

  const float dot_uv = dot_product(u, v);
  const float dot_uu = dot_product(u, u);

  const vec3 cross = cross_product(u, v);

  vec3 result;
  result = scale_vector(u, 2.0f * dot_uv);
  result = add_vector(result, scale_vector(v, s * s - dot_uu));
  result = add_vector(result, scale_vector(cross, 2.0f * s));

  return result;
}

LUMINARY_FUNCTION vec3 quaternion16_apply(const Quaternion16 q, const vec3 v) {
  Quaternion quat;
  quat.x = (q.x * (1.0f / 0x7FFF)) - 1.0f;
  quat.y = (q.y * (1.0f / 0x7FFF)) - 1.0f;
  quat.z = (q.z * (1.0f / 0x7FFF)) - 1.0f;
  quat.w = (q.w * (1.0f / 0x7FFF)) - 1.0f;

  return quaternion_apply(quat, v);
}

LUMINARY_FUNCTION vec3 quaternion16_apply_inv(const Quaternion16 q, const vec3 v) {
  Quaternion quat;
  quat.x = 1.0f - (q.x * (1.0f / 0x7FFF));
  quat.y = 1.0f - (q.y * (1.0f / 0x7FFF));
  quat.z = 1.0f - (q.z * (1.0f / 0x7FFF));
  quat.w = (q.w * (1.0f / 0x7FFF)) - 1.0f;

  return quaternion_apply(quat, v);
}

LUMINARY_FUNCTION vec3 transform_vec4_3_position(const Mat3x4 m, const vec3 p) {
  vec3 res;

  res.x = m.f11 * p.x + m.f12 * p.y + m.f13 * p.z + m.f14;
  res.y = m.f21 * p.x + m.f22 * p.y + m.f23 * p.z + m.f24;
  res.z = m.f31 * p.x + m.f32 * p.y + m.f33 * p.z + m.f34;

  return res;
}

LUMINARY_FUNCTION vec3 transform_vec3(const Mat3x3 m, const vec3 p) {
  vec3 res;

  res.x = m.f11 * p.x + m.f12 * p.y + m.f13 * p.z;
  res.y = m.f21 * p.x + m.f22 * p.y + m.f23 * p.z;
  res.z = m.f31 * p.x + m.f32 * p.y + m.f33 * p.z;

  return res;
}

////////////////////////////////////////////////////////////////////
// Transformation API
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION vec3 transform_apply_rotation(const DeviceTransform trans, const vec3 v) {
  return quaternion16_apply(trans.rotation, v);
}

LUMINARY_FUNCTION vec3 transform_apply_rotation_inv(const DeviceTransform trans, const vec3 v) {
  return quaternion16_apply_inv(trans.rotation, v);
}

LUMINARY_FUNCTION vec3 transform_apply_absolute(const DeviceTransform trans, const vec3 v) {
  return add_vector(v, trans.translation);
}

LUMINARY_FUNCTION vec3 transform_apply_absolute_inv(const DeviceTransform trans, const vec3 v) {
  return sub_vector(v, trans.translation);
}

LUMINARY_FUNCTION vec3 transform_apply_relative(const DeviceTransform trans, const vec3 v) {
  return mul_vector(transform_apply_rotation(trans, v), trans.scale);
}

LUMINARY_FUNCTION vec3 transform_apply_relative_inv(const DeviceTransform trans, const vec3 v) {
  return transform_apply_rotation_inv(trans, mul_vector(v, inv_vector(trans.scale)));
}

LUMINARY_FUNCTION vec3 transform_apply(const DeviceTransform trans, const vec3 v) {
  return transform_apply_absolute(trans, transform_apply_relative(trans, v));
}

LUMINARY_FUNCTION vec3 transform_apply_inv(const DeviceTransform trans, const vec3 v) {
  return transform_apply_relative_inv(trans, transform_apply_absolute_inv(trans, v));
}

////////////////////////////////////////////////////////////////////
// Matrix API
////////////////////////////////////////////////////////////////////

struct mat3 {
  vec3 col0;
  vec3 col1;
  vec3 col2;
} typedef mat3;

LUMINARY_FUNCTION mat3 mat3_get(const vec3 col0, const vec3 col1, const vec3 col2) {
  mat3 result;

  result.col0 = col0;
  result.col1 = col1;
  result.col2 = col2;

  return result;
}

LUMINARY_FUNCTION mat3 mat3_transpose(const mat3 mat) {
  mat3 result;

  result.col0.x = mat.col0.x;
  result.col0.y = mat.col1.x;
  result.col0.z = mat.col2.x;

  result.col1.x = mat.col0.y;
  result.col1.y = mat.col1.y;
  result.col1.z = mat.col2.y;

  result.col2.x = mat.col0.z;
  result.col2.y = mat.col1.z;
  result.col2.z = mat.col2.z;

  return result;
}

LUMINARY_FUNCTION mat3 mat3_lerp(const mat3 a, const mat3 b, const float t) {
  mat3 result;

  result.col0 = add_vector(scale_vector(a.col0, 1.0f - t), scale_vector(b.col0, t));
  result.col1 = add_vector(scale_vector(a.col1, 1.0f - t), scale_vector(b.col1, t));
  result.col2 = add_vector(scale_vector(a.col2, 1.0f - t), scale_vector(b.col2, t));

  return result;
}

LUMINARY_FUNCTION vec3 mat3_mul_vec(const mat3 a, const vec3 b) {
  vec3 result;

  result.x = a.col0.x * b.x + a.col1.x * b.y + a.col2.x * b.z;
  result.y = a.col0.y * b.x + a.col1.y * b.y + a.col2.y * b.z;
  result.z = a.col0.z * b.x + a.col1.z * b.y + a.col2.z * b.z;

  return result;
}

LUMINARY_FUNCTION mat3 mat3_mul_mat(const mat3 a, const mat3 b) {
  mat3 result;

  result.col0 = mat3_mul_vec(a, b.col0);
  result.col1 = mat3_mul_vec(a, b.col1);
  result.col2 = mat3_mul_vec(a, b.col2);

  return result;
}

LUMINARY_FUNCTION mat3 mat3_identity() {
  mat3 result;

  result.col0 = get_vector(1.0f, 0.0f, 0.0f);
  result.col1 = get_vector(0.0f, 1.0f, 0.0f);
  result.col2 = get_vector(0.0f, 0.0f, 1.0f);

  return result;
}

LUMINARY_FUNCTION mat3 mat3_scale(const mat3 mat, const float scale) {
  mat3 result;

  result.col0 = scale_vector(mat.col0, scale);
  result.col1 = scale_vector(mat.col1, scale);
  result.col2 = scale_vector(mat.col2, scale);

  return result;
}

LUMINARY_FUNCTION float mat3_determinant(const mat3 mat) {
  float det = 0.0f;

  det += mat.col0.x * (mat.col1.y * mat.col2.z - mat.col2.y * mat.col1.z);
  det -= mat.col1.x * (mat.col0.y * mat.col2.z - mat.col2.y * mat.col0.z);
  det += mat.col2.x * (mat.col0.y * mat.col1.z - mat.col1.y * mat.col0.z);

  return det;
}

LUMINARY_FUNCTION mat3 mat3_inverse(const mat3 mat) {
  const float determinant = mat3_determinant(mat);

  if (determinant == 0.0f)
    return mat3_identity();

  mat3 result;

  result.col0.x = mat.col1.y * mat.col2.z - mat.col2.y * mat.col1.z;
  result.col0.y = mat.col2.y * mat.col0.z - mat.col0.y * mat.col2.z;
  result.col0.z = mat.col0.y * mat.col1.z - mat.col1.y * mat.col0.z;
  result.col1.x = mat.col2.x * mat.col1.z - mat.col1.x * mat.col2.z;
  result.col1.y = mat.col0.x * mat.col2.z - mat.col2.x * mat.col0.z;
  result.col1.z = mat.col1.x * mat.col0.z - mat.col0.x * mat.col1.z;
  result.col2.x = mat.col1.x * mat.col2.y - mat.col2.x * mat.col1.y;
  result.col2.y = mat.col2.x * mat.col0.y - mat.col0.x * mat.col2.y;
  result.col2.z = mat.col0.x * mat.col1.y - mat.col1.x * mat.col0.y;

  return mat3_scale(result, 1.0f / determinant);
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
LUMINARY_FUNCTION float sphere_ray_intersection(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
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
LUMINARY_FUNCTION float sph_ray_int_p0(const vec3 ray, const vec3 origin, const float r) {
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
LUMINARY_FUNCTION bool sphere_ray_hit(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
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
LUMINARY_FUNCTION bool sph_ray_hit_p0(const vec3 ray, const vec3 origin, const float r) {
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
LUMINARY_FUNCTION float sphere_ray_intersect_back(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
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
LUMINARY_FUNCTION float sph_ray_int_back_p0(const vec3 ray, const vec3 origin, const float r) {
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

LUMINARY_FUNCTION vec3 angles_to_direction(const float altitude, const float azimuth) {
  vec3 dir;
  dir.x = cosf(azimuth) * cosf(altitude);
  dir.y = sinf(altitude);
  dir.z = sinf(azimuth) * cosf(altitude);

  return dir;
}

LUMINARY_FUNCTION void direction_to_angles(const vec3 dir, float& azimuth, float& altitude) {
  altitude = asinf(dir.y);
  azimuth  = atan2f(dir.z, dir.x);

  if (azimuth < 0.0f)
    azimuth += 2.0f * PI;
}

// PBRT v3 Chapter "Specular Reflection and Transmission", Refract() function
LUMINARY_FUNCTION vec3 refract_vector(const vec3 V, const vec3 normal, const float index_ratio, bool& total_reflection) {
  if (index_ratio < eps) {
    total_reflection = false;
    return scale_vector(V, -1.0f);
  }

  const float dot = fabsf(dot_product(normal, V));

  const float b = 1.0f - index_ratio * index_ratio * (1.0f - dot * dot);

  total_reflection = b < 0.0f;

  if (total_reflection) {
    // Total reflection
    return reflect_vector(V, normal);
  }
  else {
    return normalize_vector(sub_vector(scale_vector(normal, index_ratio * dot - sqrtf(b)), scale_vector(V, index_ratio)));
  }
}

// Shift origin vector to avoid self intersection.
LUMINARY_FUNCTION vec3 shift_origin_vector(const vec3 origin, const vec3 V, const vec3 L, const bool is_refraction, const float length) {
  if (length == 0.0f)
    return origin;

  const vec3 shift_vector = (is_refraction) ? L : V;
  return add_vector(origin, scale_vector(shift_vector, length));
}

LUMINARY_FUNCTION RGBF get_color(const float r, const float g, const float b) {
  RGBF result;

  result.r = r;
  result.g = g;
  result.b = b;

  return result;
}

LUMINARY_FUNCTION RGBF splat_color(const float v) {
  RGBF result;

  result.r = v;
  result.g = v;
  result.b = v;

  return result;
}

LUMINARY_FUNCTION RGBAF get_RGBAF(const float r, const float g, const float b, const float a) {
  RGBAF result;

  result.r = r;
  result.g = g;
  result.b = b;
  result.a = a;

  return result;
}

LUMINARY_FUNCTION RGBF add_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r + b.r;
  result.g = a.g + b.g;
  result.b = a.b + b.b;

  return result;
}

LUMINARY_FUNCTION RGBF sub_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r - b.r;
  result.g = a.g - b.g;
  result.b = a.b - b.b;

  return result;
}

LUMINARY_FUNCTION RGBF mul_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r * b.r;
  result.g = a.g * b.g;
  result.b = a.b * b.b;

  return result;
}

LUMINARY_FUNCTION RGBF scale_color(const RGBF a, const float b) {
  RGBF result;

  result.r = a.r * b;
  result.g = a.g * b;
  result.b = a.b * b;

  return result;
}

LUMINARY_FUNCTION RGBF fma_color(const RGBF a, const float b, const RGBF c) {
  return add_color(c, scale_color(a, b));
}

LUMINARY_FUNCTION RGBF min_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = fminf(a.r, b.r);
  result.g = fminf(a.g, b.g);
  result.b = fminf(a.b, b.b);

  return result;
}

LUMINARY_FUNCTION RGBF max_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = fmaxf(a.r, b.r);
  result.g = fmaxf(a.g, b.g);
  result.b = fmaxf(a.b, b.b);

  return result;
}

LUMINARY_FUNCTION RGBF inv_color(const RGBF a) {
  RGBF result;

  result.r = 1.0f / a.r;
  result.g = 1.0f / a.g;
  result.b = 1.0f / a.b;

  return result;
}

LUMINARY_FUNCTION int color_any(const RGBF a) {
  return (a.r > 0.0f || a.g > 0.0f || a.b > 0.0f);
}

LUMINARY_FUNCTION float RGBF_avg(const RGBF a) {
  return (a.r + a.g + a.b) * (1.0f / 3.0f);
}

LUMINARY_FUNCTION RGBF opaque_color(const RGBAF a) {
  return get_color(a.r, a.g, a.b);
}

LUMINARY_FUNCTION RGBAF transparent_color(const RGBF a, const float alpha) {
  RGBAF result;

  result.r = a.r;
  result.g = a.g;
  result.b = a.b;
  result.a = alpha;

  return result;
}

LUMINARY_FUNCTION RGBAF RGBAF_set(const float r, const float g, const float b, const float a) {
  RGBAF result;

  result.r = r;
  result.g = g;
  result.b = b;
  result.a = a;

  return result;
}

LUMINARY_FUNCTION RGBAF RGBAF_add(const RGBAF a, const RGBAF b) {
  RGBAF result;

  result.r = a.r + b.r;
  result.g = a.g + b.g;
  result.b = a.b + b.b;
  result.a = a.a + b.a;

  return result;
}

LUMINARY_FUNCTION RGBAF RGBAF_scale(const RGBAF a, const float b) {
  RGBAF result;

  result.r = a.r * b;
  result.g = a.g * b;
  result.b = a.b * b;
  result.a = a.a * b;

  return result;
}

// Unused, maybe useful in the future
LUMINARY_FUNCTION float color_decompress_impl(const uint32_t value, const uint32_t exponent_bits, const uint32_t mantissa_bits) {
  const uint32_t mantissa_mask = ((1 << mantissa_bits) - 1);

  const uint32_t source_bits = value;

  const uint32_t exponent_offset = ((1 << 7) - 1) - ((1 << (exponent_bits - 1)) - 1);

  const uint32_t exponent = (source_bits >> mantissa_bits) + exponent_offset;

  const uint32_t bits = (exponent << 23) | ((source_bits & mantissa_mask) << (23 - mantissa_bits));

  return __uint_as_float(bits);
}

LUMINARY_FUNCTION uint32_t color_compress_impl(const float value, const uint32_t exponent_bits, const uint32_t mantissa_bits) {
  if (value <= 0.0f)
    return 0;

  const uint32_t exponent_mask = ((1 << exponent_bits) - 1);
  const uint32_t mantissa_mask = ((1 << mantissa_bits) - 1);

  const uint32_t source_bits = __float_as_uint(value);

  const uint32_t exponent_offset = ((1 << 7) - 1) - ((1 << (exponent_bits - 1)) - 1);

  // Add half an ulp to round instead of truncate.
  const uint32_t mantissa = (((source_bits & 0x7FFFFF) + (1 << (23 - mantissa_bits - 1))) >> (23 - mantissa_bits));

  // If the mantissa overflowed, we need to increment the exponent and shift the mantissa.
  const uint32_t mantissa_overflow = mantissa >> mantissa_bits;

  const uint32_t exponent = min(max(source_bits >> 23, exponent_offset) - exponent_offset + mantissa_overflow, exponent_mask);

  const uint32_t bits = (exponent << mantissa_bits) | (mantissa & mantissa_mask);

  return bits;
}

//
// Linear RGB / sRGB conversion taken from https://www.shadertoy.com/view/lsd3zN
// which is based os D3DX implementations
//

LUMINARY_FUNCTION float linearRGB_to_SRGB(const float value) {
  if (value <= 0.0031308f) {
    return 12.92f * value;
  }
  else {
    return 1.055f * powf(value, 0.416666666667f) - 0.055f;
  }
}

LUMINARY_FUNCTION float SRGB_to_linearRGB(const float value) {
  if (value <= 0.04045f) {
    return value / 12.92f;
  }
  else {
    return powf((value + 0.055f) / 1.055f, 2.4f);
  }
}

LUMINARY_FUNCTION float luminance(const RGBF v) {
  return 0.212655f * v.r + 0.715158f * v.g + 0.072187f * v.b;
}

LUMINARY_FUNCTION float color_importance(const RGBF color) {
  return __fmax_fmax(color.r, color.g, color.b);
}

LUMINARY_FUNCTION RGBAF saturate_albedo(RGBAF color, float change) {
  const float max_value = fmaxf(color.r, fmaxf(color.g, color.b));
  const float min_value = fminf(color.r, fminf(color.g, color.b));
  const float diff      = 0.01f + max_value - min_value;
  color.r               = fmaxf(0.0f, color.r - change * ((max_value - color.r) / diff) * min_value);
  color.g               = fmaxf(0.0f, color.g - change * ((max_value - color.g) / diff) * min_value);
  color.b               = fmaxf(0.0f, color.b - change * ((max_value - color.b) / diff) * min_value);

  return color;
}

LUMINARY_FUNCTION RGBF filter_gray(const RGBF color) {
  const float value = luminance(color);

  return get_color(value, value, value);
}

LUMINARY_FUNCTION RGBF filter_sepia(const RGBF color) {
  return get_color(
    color.r * 0.393f + color.g * 0.769f + color.b * 0.189f, color.r * 0.349f + color.g * 0.686f + color.b * 0.168f,
    color.r * 0.272f + color.g * 0.534f + color.b * 0.131f);
}

LUMINARY_FUNCTION RGBF filter_gameboy(const RGBF color, const uint32_t x, const uint32_t y) {
  const float value  = 4.0f * luminance(color);
  const float dither = random_dither_mask(x, y);

  const int tone = (int) (value + dither);

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

LUMINARY_FUNCTION RGBF filter_2bitgray(const RGBF color, const uint32_t x, const uint32_t y) {
  const float value  = 4.0f * luminance(color);
  const float dither = random_dither_mask(x, y);

  const int tone = (int) (value + dither);

  switch (tone) {
    case 0:
      return get_color(0.0f, 0.0f, 0.0f);
    case 1:
      return get_color(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    case 2:
      return get_color(2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f);
    case 3:
    default:
      return get_color(1.0f, 1.0f, 1.0f);
  }
}

LUMINARY_FUNCTION RGBF filter_crt(RGBF color, int x, int y) {
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

LUMINARY_FUNCTION RGBF filter_blackwhite(const RGBF color, const uint32_t x, const uint32_t y) {
  const float value  = 2.0f * luminance(color);
  const float dither = random_dither_mask(x, y);

  const int tone = (int) (value + dither);

  switch (tone) {
    case 0:
      return get_color(0.0f, 0.0f, 0.0f);
    case 1:
    default:
      return get_color(1.0f, 1.0f, 1.0f);
  }
}

LUMINARY_FUNCTION float henyey_greenstein_phase_function(const float cos_angle, const float g) {
  const float g2         = g * g;
  const float denom_term = 1.0f + g2 - 2.0f * g * cos_angle;
  const float pow15      = denom_term * sqrtf(denom_term);

  return (1.0f - g * g) / (4.0f * PI * pow15);
}

LUMINARY_FUNCTION float draine_phase_function(const float cos_angle, const float g, const float alpha) {
  return henyey_greenstein_phase_function(cos_angle, g)
         * ((1.0f + alpha * cos_angle * cos_angle) / (1.0f + (alpha / 3.0f) * (1.0f + 2.0f * g * g)));
}

struct JendersieEonParams {
  float g_hg;
  float g_d;
  float alpha;
  float w_d;
} typedef JendersieEonParams;

LUMINARY_FUNCTION JendersieEonParams jendersie_eon_phase_parameters(const float diameter) {
  JendersieEonParams params;

  // Renaming to a shorter name.
  const float d = diameter;

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

  return params;
}

/*
 * Fog phase function from [JenE23].
 *
 * [JenE23] J. Jendersie and E. d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks, 2023.
 *
 * @param diameter Diameter of water droplets in [5,50] in micrometer.
 */
LUMINARY_FUNCTION float jendersie_eon_phase_function(const float cos_angle, const JendersieEonParams params, const float ms_factor = 1.0f) {
  const float phase_hg = henyey_greenstein_phase_function(cos_angle, params.g_hg * ms_factor);
  const float phase_d  = draine_phase_function(cos_angle, params.g_d * ms_factor, params.alpha);

  return (1.0f - params.w_d) * phase_hg + params.w_d * phase_d;
}

/*
 * Uses a orthonormal basis which is built as described in
 * J. Frisvad, _Building an Orthonormal Basis from a 3D Unit Vector Without Normalization_, 2012
 * with an improved threshold given in
 * N. Max, _Improved accuracy when building an orthonormal basis_, 2017.
 * The common branchless version has a discontinuity at z=0 which makes it unsuitable for
 * sampling with low discrepancy random numbers.
 */
LUMINARY_FUNCTION vec3 phase_sample_basis(const float alpha, const float beta, const vec3 basis) {
  vec3 u1, u2;

  if (basis.z < -0.9999805689f) {
    u1 = get_vector(0.0f, -1.0f, 0.0f);
    u2 = get_vector(-1.0f, 0.0f, 0.0f);
  }
  else {
    const float a = 1.0f / (1.0f + basis.z);
    const float b = -basis.x * basis.y * a;

    u1 = get_vector(1.0f - basis.x * basis.x * a, b, -basis.x);
    u2 = get_vector(b, 1.0f - basis.y * basis.y * a, -basis.y);
  }

  const vec3 s = sample_ray_sphere(alpha, beta);

  vec3 result;
  result.x = s.x * u1.x + s.y * u2.x + s.z * basis.x;
  result.y = s.x * u1.y + s.y * u2.y + s.z * basis.y;
  result.z = s.x * u1.z + s.y * u2.z + s.z * basis.z;

  return normalize_vector(result);
}

LUMINARY_FUNCTION float henyey_greenstein_phase_sample(const float g, const float r) {
  const float g2 = g * g;

  const float t = (1.0f - g2) / (1.0f - g + 2.0f * g * r);

  // Note that the Jendersie-Eon paper supplement claims only t, but t^2 is correct according to all other sources.
  return (1.0f + g2 - t * t) / (2.0f * g);
}

LUMINARY_FUNCTION float draine_phase_sample(const float g, const float alpha, const float r) {
  const float g2 = g * g;
  const float g4 = g2 * g2;

  const float t0 = alpha - alpha * g2;
  const float t1 = alpha * g4 - alpha;
  const float t2 = -3.0f * (4.0f * (g4 - g2) + t1 * (1.0f + g2));
  const float t3 = g * (2.0f * r - 1.0f);
  const float t4 = 3.0f * g2 * (1.0f + t3) + alpha * (2.0f + g2 * (1.0f + (1.0f + 2.0f * g2) * t3));
  const float t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
  const float t6 = t0 * 4.0f * (g4 - g2);
  const float t7 = cbrtf(t5 + sqrtf(t5 * t5 - t6 * t6 * t6));
  const float t8 = 2.0f * ((t1 + (t6 / t7) + t7) / t0);
  const float t9 = sqrtf(6.0f * (1.0f + g2) + t8);

  const float h = sqrtf(6.0f * (1.0f + g2) - t8 + 8.0f * t4 / (t0 * t9)) - t9;

  return 0.5f * g + ((1.0f / (2.0f * g)) - (1.0f / (8.0f * g)) * (h * h));
}

/*
 * This function perfectly importance samples the phase function from [JenE23]. The details of this are presented in the
 * supplement of the paper.
 *
 * [JenE23] J. Jendersie and E. d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks, 2023.
 *
 * @param diameter Diameter of water droplets in [5,50] in micrometer.
 */
LUMINARY_FUNCTION vec3 jendersie_eon_phase_sample(const vec3 ray, const float diameter, const float2 r_dir, const float r_choice) {
  const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);

  float cos_angle;
  if (r_choice < params.w_d) {
    cos_angle = draine_phase_sample(params.g_d, params.alpha, r_dir.x);
  }
  else {
    cos_angle = henyey_greenstein_phase_sample(params.g_hg, r_dir.x);
  }

  return phase_sample_basis(cos_angle, r_dir.y, ray);
}

LUMINARY_FUNCTION float jendersie_eon_phase_sample_cos_angle(const JendersieEonParams params, const float r_dir, const float r_choice) {
  float cos_angle;
  if (r_choice < params.w_d) {
    cos_angle = draine_phase_sample(params.g_d, params.alpha, r_dir);
  }
  else {
    cos_angle = henyey_greenstein_phase_sample(params.g_hg, r_dir);
  }

  return cos_angle;
}

LUMINARY_FUNCTION float bvh_triangle_intersection(
  const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 origin, const vec3 ray, float2& coords) {
  const vec3 h  = cross_product(ray, edge2);
  const float a = dot_product(edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, edge1);
  const float v = f * dot_product(ray, q);

  coords = make_float2(u, v);

  //  The third check is inverted to catch NaNs since NaNs always return false, the not will turn it into a true
  if (v < 0.0f || u < 0.0f || !(u + v <= 1.0f))
    return FLT_MAX;

  const float t = f * dot_product(edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

LUMINARY_FUNCTION float clampf(const float x, const float a, const float b) {
  return fminf(b, fmaxf(a, x));
}

/*
 * A more numerically stable version of normalize(b-a).
 * Whether this is actually is any better, I don't know, I just pretend.
 */
LUMINARY_FUNCTION vec3 vector_direction_stable(vec3 a, vec3 b) {
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
 * Samples the solid angle of a sphere.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @param origin Point to sample from.
 * @result Normalized direction to the point on the sphere.
 */
LUMINARY_FUNCTION vec3 sample_sphere(const vec3 p, const float r, const vec3 origin, const float2 random, float& area) {
  float r1 = random.x;
  float r2 = random.y;

  vec3 dir      = sub_vector(p, origin);
  const float d = get_length(dir);

  if (d < r) {
    area = 4.0f * PI;
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
LUMINARY_FUNCTION float sample_sphere_solid_angle(const vec3 p, const float r, const vec3 origin) {
  vec3 dir      = sub_vector(p, origin);
  const float d = get_length(dir);

  if (d < r)
    return 2.0f * PI;

  const float a = asinf(r / d);

  return 2.0f * PI * a * a;
}

// Computes tangent space for use with normal mapping without precomputation
// http://www.thetenthplanet.de/archives/1180
LUMINARY_FUNCTION Mat3x3 cotangent_frame(vec3 normal, vec3 e1, vec3 e2, UV t1, UV t2) {
  e1 = normalize_vector(e1);
  e2 = normalize_vector(e2);

  const float abs1 = __frsqrt_rn(t1.u * t1.u + t1.v * t1.v);
  if (!is_non_finite(abs1)) {
    t1.u *= abs1;
    t1.v *= abs1;
  }
  const float abs2 = __frsqrt_rn(t2.u * t2.u + t2.v * t2.v);
  if (!is_non_finite(abs2)) {
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

LUMINARY_FUNCTION RGBF rgb_to_hsv(const RGBF rgb) {
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
LUMINARY_FUNCTION RGBF hsv_to_rgb(RGBF hsv) {
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

LUMINARY_FUNCTION vec3 direction_project(const vec3 a, const vec3 b) {
  return scale_vector(b, dot_product(a, b));
}

LUMINARY_FUNCTION vec3 normal_adaptation_apply(const vec3 V, vec3 shading_normal, const vec3 geometry_normal) {
  // TODO: Fine tune this, this is so far only so that we actually have something to work with

  // Make sure that shading and geometry normal are on the same side
  if (dot_product(shading_normal, geometry_normal) < 0.0f) {
    shading_normal = scale_vector(shading_normal, -1.0f);
  }

  if (dot_product(V, shading_normal) < 0.0f) {
#if 0
    // Rotate the geometry normal to lie on the same plane as V and the shading normal and use that one
    const vec3 cross               = cross_product(V, shading_normal);
    const vec3 proj_geometry_cross = direction_project(geometry_normal, cross);
    return normalize_vector(sub_vector(geometry_normal, proj_geometry_cross));
#else
    // Rotate the shading normal so that V and shading normal are orthogonal and then rotate just a little bit more
    const vec3 proj_shading_V = direction_project(shading_normal, V);
    return normalize_vector(sub_vector(shading_normal, scale_vector(proj_shading_V, 1.1f)));
#endif
  }

  return shading_normal;
}

////////////////////////////////////////////////////////////////////
// Packing
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION float bfloat_unpack(const BFloat16 val) {
  const uint32_t data = val;

  return __uint_as_float(data << 16);
}

LUMINARY_FUNCTION BFloat16 bfloat_pack(const float val) {
  return __float_as_uint(val) >> 16;
}

LUMINARY_FUNCTION float unsigned_bfloat_unpack(const UnsignedBFloat16 val) {
  const uint32_t data = val;

  return __uint_as_float(data << 15);
}

LUMINARY_FUNCTION UnsignedBFloat16 unsigned_bfloat_pack(const float val) {
  return (__float_as_uint(val) >> 15) & 0xFFFF;
}

LUMINARY_FUNCTION RGBF record_unpack(const PackedRecord packed) {
  // 21 bits each
  const uint32_t red   = packed.x & 0x1FFFFF;
  const uint32_t green = (packed.x >> 21) | ((packed.y & 0x3FF) << 11);
  const uint32_t blue  = packed.y >> 10;

  RGBF record;
  record.r = __uint_as_float(red << 11);
  record.g = __uint_as_float(green << 11);
  record.b = __uint_as_float(blue << 11);

  return record;
}

LUMINARY_FUNCTION PackedRecord record_pack(const RGBF record) {
  const uint32_t red   = __float_as_uint(record.r) >> 11;
  const uint32_t green = __float_as_uint(record.g) >> 11;
  const uint32_t blue  = __float_as_uint(record.b) >> 11;

  PackedRecord packed;
  packed.x = red | (green << 21);
  packed.y = (green >> 11) | (blue << 10);

  return packed;
}

LUMINARY_FUNCTION vec3 ray_unpack(const PackedRayDirection packed) {
  float x = packed.x * (1.0f / 0xFFFFFFFF);
  float y = packed.y * (1.0f / 0xFFFFFFFF);

  x = (x * 2.0f) - 1.0f;
  y = (y * 2.0f) - 1.0f;

  vec3 ray      = get_vector(x, y, 1.0f - fabsf(x) - fabsf(y));
  const float t = __saturatef(-ray.z);

  ray.x += (ray.x >= 0.0f) ? -t : t;
  ray.y += (ray.y >= 0.0f) ? -t : t;

  return normalize_vector(ray);
}

LUMINARY_FUNCTION PackedRayDirection ray_pack(const vec3 ray) {
  float x = ray.x;
  float y = ray.y;
  float z = ray.z;

  const float recip_norm = 1.0f / (fabsf(x) + fabsf(y) + fabsf(z));

  x *= recip_norm;
  y *= recip_norm;
  z *= recip_norm;

  const float t = __saturatef(-z);

  x += (x >= 0.0f) ? t : -t;
  y += (y >= 0.0f) ? t : -t;

  x = clampf(x, -1.0f, 1.0f);
  y = clampf(y, -1.0f, 1.0f);

  x = (x + 1.0f) * 0.5f;
  y = (y + 1.0f) * 0.5f;

  PackedRayDirection packed;
  packed.x = (uint32_t) (x * 0xFFFFFFFF + 0.5f);
  packed.y = (uint32_t) (y * 0xFFFFFFFF + 0.5f);

  return packed;
}

LUMINARY_FUNCTION vec3 position_unpack(const PackedPosition packed) {
  const float x = __uint_as_float((packed.x & 0x1FFFFF) << 11);
  const float z = __uint_as_float((packed.y & 0x1FFFFF) << 11);

  const float y = __uint_as_float(((packed.x & 0xFFE00000) >> 11) | (packed.y & 0xFFE00000));

  return get_vector(x, y, z);
}

LUMINARY_FUNCTION PackedPosition position_pack(const vec3 pos) {
  const uint32_t x = __float_as_uint(pos.x) >> 11;
  const uint32_t y = __float_as_uint(pos.y) >> 10;
  const uint32_t z = __float_as_uint(pos.z) >> 11;

  PackedPosition packed;
  packed.x = x | ((y & 0x0007FF) << 21);
  packed.y = z | ((y & 0x3FF800) << 10);

  return packed;
}

LUMINARY_FUNCTION Quaternion16 quaternion_pack(const Quaternion q) {
  // We use the inverse of the quaternion so that it matches with our Quaternion to Matrix conversion for the OptiX BVH.
  Quaternion16 dst;
  dst.x = (uint16_t) (((1.0f - q.x) * 0x7FFF) + 0.5f);
  dst.y = (uint16_t) (((1.0f - q.y) * 0x7FFF) + 0.5f);
  dst.z = (uint16_t) (((1.0f - q.z) * 0x7FFF) + 0.5f);
  dst.w = (uint16_t) (((1.0f + q.w) * 0x7FFF) + 0.5f);

  return dst;
}

LUMINARY_FUNCTION float normed_float_unpack(const uint16_t data) {
  return (data * (1.0f / 0xFFFF));
}

LUMINARY_FUNCTION float unsigned_float_unpack(const uint16_t data) {
  return __uint_as_float(data << 15);
}

LUMINARY_FUNCTION UV uv_unpack(const PackedUV data) {
  UV uv;

  uv.u = __uint_as_float(data & 0xFFFF0000);
  uv.v = __uint_as_float(data << 16);

  return uv;
}

// Octahedron decoding, for example: https://www.shadertoy.com/view/clXXD8
LUMINARY_FUNCTION vec3 normal_unpack(const PackedNormal data) {
  float x = (data & 0xFFFF) * (1.0f / 0xFFFF);
  float y = (data >> 16) * (1.0f / 0xFFFF);

  x = (x * 2.0f) - 1.0f;
  y = (y * 2.0f) - 1.0f;

  vec3 normal   = get_vector(x, y, 1.0f - fabsf(x) - fabsf(y));
  const float t = __saturatef(-normal.z);

  normal.x += (normal.x >= 0.0f) ? -t : t;
  normal.y += (normal.y >= 0.0f) ? -t : t;

  return normalize_vector(normal);
}

LUMINARY_FUNCTION PackedNormal normal_pack(const vec3 normal) {
  float x = normal.x;
  float y = normal.y;
  float z = normal.z;

  const float recip_norm = 1.0f / (fabsf(x) + fabsf(y) + fabsf(z));

  x *= recip_norm;
  y *= recip_norm;
  z *= recip_norm;

  const float t = fmaxf(fminf(-z, 1.0f), 0.0f);

  x += (x >= 0.0f) ? t : -t;
  y += (y >= 0.0f) ? t : -t;

  x = fmaxf(fminf(x, 1.0f), -1.0f);
  y = fmaxf(fminf(y, 1.0f), -1.0f);

  x = (x + 1.0f) * 0.5f;
  y = (y + 1.0f) * 0.5f;

  const uint32_t x_u16 = (uint32_t) (x * 0xFFFF + 0.5f);
  const uint32_t y_u16 = (uint32_t) (y * 0xFFFF + 0.5f);

  return (y_u16 << 16) | x_u16;
}

#endif /* CU_MATH_H */
