#ifndef CU_LUMINARY_LIGHT_LTC_H
#define CU_LUMINARY_LIGHT_LTC_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

// Code taken from https://github.com/AakashKT/LTC-Anisotropic

struct LTCMatrix {
  mat3 mat;
  float normalization;
  bool invert_winding;
} typedef LTCMatrix;

__device__ mat3 light_ltc_fetch(const float3 u, float& normalization) {
  const float4 data0 = tex3D<float4>(device.ltc_tex[0].handle, u.x, u.y, u.z);
  const float4 data1 = tex3D<float4>(device.ltc_tex[1].handle, u.x, u.y, u.z);
  const float2 data2 = tex3D<float2>(device.ltc_tex[2].handle, u.x, u.y, u.z);

  const vec3 m0 = get_vector(data0.x, data0.y, data0.z);
  const vec3 m1 = get_vector(data0.w, data1.x, data1.y);
  const vec3 m2 = get_vector(data1.z, data1.w, data2.x);

  normalization = data2.y;

  const mat3 mat = mat3_get(m0, m1, m2);

  return mat3_transpose(mat);
}

__device__ mat3 light_ltc_fetch_and_interpolate(const float4 u, float& normalization) {
  const float ws   = u.w * 7.0f;
  const float ws_f = floorf(ws);
  const float ws_c = fminf(ws_f + 1.0f, 7.0f);
  const float w    = ws - ws_f;

#if 1
  const float x = (u.x * 7.0 + 0.5) / 8.0;
#else
  const float x = (u.x * 31.0 + 0.5) / 32.0;
#endif

  const float y  = (u.y * 7.0 + 0.5) / 8.0;
  const float z1 = ((u.z * 7.0 + 8.0 * ws_f + 0.5) / 64.0);
  const float z2 = ((u.z * 7.0 + 8.0 * ws_c + 0.5) / 64.0);

  float norm1, norm2;
  const mat3 m1 = light_ltc_fetch(make_float3(x, y, z1), norm1);
  const mat3 m2 = light_ltc_fetch(make_float3(x, y, z2), norm2);

  normalization = lerp(norm1, norm2, w) * (1.0f / (2.0f * PI));

  return mat3_lerp(m1, m2, w);
}

__device__ LTCMatrix light_ltc_load(const vec3 V, const float roughness_u, const float roughness_v) {
  const float theta_o    = acosf(V.z);
  const bool flip_config = roughness_v > roughness_u;
  const float phi_o_tmp  = atan2f(V.y, V.x);
  const float phi_o_tmp2 = flip_config ? (PI / 2.0f - phi_o_tmp) : phi_o_tmp;
  const float phi_o      = phi_o_tmp2 >= 0.0f ? phi_o_tmp2 : phi_o_tmp2 + 2.0f * PI;
  const float u0         = ((flip_config ? roughness_v : roughness_u) - 1e-3f) / (1.0f - 1e-3);
  const float u1         = flip_config ? roughness_u / roughness_v : roughness_v / roughness_u;
  const float u2         = theta_o / PI * 2.0f;

  mat3 ltc_matrix;
  float ltc_normalization;
  bool ltc_invert_winding;

  if (phi_o < 0.5f * PI) {
    const float u3 = phi_o / (PI * 0.5f);
    const float4 u = make_float4(u3, u2, u1, u0);

    ltc_invert_winding = true;
    ltc_matrix         = light_ltc_fetch_and_interpolate(u, ltc_normalization);
  }
  else if (phi_o >= PI * 0.5f && phi_o < PI) {
    const float u3 = (PI - phi_o) / (PI * 0.5f);
    const float4 u = make_float4(u3, u2, u1, u0);

    ltc_invert_winding = false;
    ltc_matrix         = light_ltc_fetch_and_interpolate(u, ltc_normalization);

    ltc_matrix.col0.x = -ltc_matrix.col0.x;
    ltc_matrix.col1.x = -ltc_matrix.col1.x;
    ltc_matrix.col2.x = -ltc_matrix.col2.x;
  }
  else if (phi_o >= PI && phi_o < 1.5f * PI) {
    const float u3 = (phi_o - PI) / (PI * 0.5f);
    const float4 u = make_float4(u3, u2, u1, u0);

    ltc_invert_winding = true;
    ltc_matrix         = light_ltc_fetch_and_interpolate(u, ltc_normalization);

    ltc_matrix.col0.x = -ltc_matrix.col0.x;
    ltc_matrix.col1.x = -ltc_matrix.col1.x;
    ltc_matrix.col2.x = -ltc_matrix.col2.x;
    ltc_matrix.col0.y = -ltc_matrix.col0.y;
    ltc_matrix.col1.y = -ltc_matrix.col1.y;
    ltc_matrix.col2.y = -ltc_matrix.col2.y;
  }
  else if (phi_o >= 1.5f * PI && phi_o <= 2.0f * PI) {
    const float u3 = (2.0f * PI - phi_o) / (PI * 0.5f);
    const float4 u = make_float4(u3, u2, u1, u0);

    ltc_invert_winding = false;
    ltc_matrix         = light_ltc_fetch_and_interpolate(u, ltc_normalization);

    ltc_matrix.col0.y = -ltc_matrix.col0.y;
    ltc_matrix.col1.y = -ltc_matrix.col1.y;
    ltc_matrix.col2.y = -ltc_matrix.col2.y;
  }
  else {
    printf("This shouldn't happen: %f\n", phi_o);
  }

  if (flip_config) {
    ltc_invert_winding = !ltc_invert_winding;

    const vec3 temp = ltc_matrix.col0;
    ltc_matrix.col0 = ltc_matrix.col1;
    ltc_matrix.col1 = temp;
  }

  ltc_matrix = mat3_inverse(ltc_matrix);

  LTCMatrix result;
  result.mat            = ltc_matrix;
  result.normalization  = ltc_normalization;
  result.invert_winding = ltc_invert_winding;

  return result;
}

__device__ float light_ltc_edge_integral(const vec3 v0, const vec3 v1) {
  const float cos_theta = dot_product(v0, v1);
  const float theta     = acosf(cos_theta);
  const float res       = cross_product(v0, v1).z * ((theta > 0.001f) ? theta / sinf(theta) : 1.0f);

  return res;
}

__device__ vec3 light_ltc_clip_edge(const vec3 a, const vec3 b) {
  return add_vector(scale_vector(a, -b.z), scale_vector(b, a.z));
}

__device__ void light_ltc_clip_triangle(vec3 v[4], uint32_t& n) {
  int config = 0;
  if (v[0].z > 0.0)
    config += 1;
  if (v[1].z > 0.0)
    config += 2;
  if (v[2].z > 0.0)
    config += 4;

  // clip
  n = 0;

  switch (config) {
    case 0:
      // clip all
      break;
    case 1:  // v0 clip v1 v2
      n    = 3;
      v[1] = light_ltc_clip_edge(v[0], v[1]);
      v[2] = light_ltc_clip_edge(v[0], v[2]);
      break;
    case 2:  // v1 clip v0 v2
      n    = 3;
      v[0] = light_ltc_clip_edge(v[1], v[0]);
      v[2] = light_ltc_clip_edge(v[1], v[2]);
      break;
    case 3:  // v0 v1 clip v2
      n    = 4;
      v[3] = light_ltc_clip_edge(v[0], v[2]);
      v[2] = light_ltc_clip_edge(v[1], v[2]);
      break;
    case 4:  // v2 clip v0 v1
      n    = 3;
      v[0] = light_ltc_clip_edge(v[2], v[0]);
      v[1] = light_ltc_clip_edge(v[2], v[1]);
      break;
    case 5:  // v0 v2 clip v1
      n    = 4;
      v[3] = v[2];
      v[2] = light_ltc_clip_edge(v[3], v[1]);
      v[1] = light_ltc_clip_edge(v[0], v[1]);
      break;
    case 6:  // v1 v2 clip v0
      n    = 4;
      v[3] = light_ltc_clip_edge(v[2], v[0]);
      v[0] = light_ltc_clip_edge(v[1], v[0]);
      break;
    case 7:  // v0 v1 v2
      n = 3;
      break;
    default:
      break;
  }

  if (n == 3)
    v[3] = v[0];
}

__device__ float light_ltc_triangle_integral_impl(vec3 v[4]) {
  uint32_t n;
  light_ltc_clip_triangle(v, n);

  if (n == 0)
    return 0.0f;

  v[0] = normalize_vector(v[0]);
  v[1] = normalize_vector(v[1]);
  v[2] = normalize_vector(v[2]);
  v[3] = normalize_vector(v[3]);

  float sum = 0.0f;
  sum += light_ltc_edge_integral(v[0], v[1]);
  sum += light_ltc_edge_integral(v[1], v[2]);
  sum += light_ltc_edge_integral(v[2], v[3]);
  if (n == 4)
    sum += light_ltc_edge_integral(v[3], v[0]);

  return sum;
}

__device__ float light_ltc_triangle_integral(
  const LTCMatrix ltc_matrix, const bool include_diffuse, const vec3 origin, const Quaternion rotation_to_z, const vec3 vertex0,
  const vec3 vertex1, const vec3 vertex2) {
  const vec3 dir0 = sub_vector(vertex0, origin);
  const vec3 dir1 = sub_vector(vertex1, origin);
  const vec3 dir2 = sub_vector(vertex2, origin);

  const vec3 dir0_local = quaternion_apply(rotation_to_z, dir0);
  const vec3 dir1_local = quaternion_apply(rotation_to_z, dir1);
  const vec3 dir2_local = quaternion_apply(rotation_to_z, dir2);

  vec3 v[4];

  float diffuse_intensity = 0.0f;
  if (include_diffuse) {
    v[0] = dir0_local;
    v[1] = dir1_local;
    v[2] = dir2_local;

    diffuse_intensity = light_ltc_triangle_integral_impl(v);
  }

  v[0] = mat3_mul_vec(ltc_matrix.mat, dir0_local);
  v[1] = mat3_mul_vec(ltc_matrix.mat, dir1_local);
  v[2] = mat3_mul_vec(ltc_matrix.mat, dir2_local);

  const float microfacet_intensity = light_ltc_triangle_integral_impl(v);

  return fabsf(diffuse_intensity) + fabsf(microfacet_intensity) * ltc_matrix.normalization;
}

#endif /* CU_LUMINARY_LIGHT_LTC_H */
