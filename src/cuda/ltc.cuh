#ifndef CU_LTC_H
#define CU_LTC_H

#if defined(SHADING_KERNEL) && !defined(VOLUME_KERNEL)

#include "bsdf_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

//
// This implementation is based on https://github.com/ishaanshah/risltc.
//

struct LTCCoefficients {
  Mat3x3 shading_to_cosine_transformation;
  Mat3x3 cosine_to_shading_transformation;
  float albedo;  // This might not be necessary, check what we have in here, this is only used for microfacet and I think it may just be the
                 // F0 term.
  float shading_to_cosine_determinant;
  Mat4x4 world_to_shading_transformation;
} typedef LTCCoefficients;

__device__ LTCCoefficients ltc_get_coefficients(GBufferData data) {
  const float NdotV       = dot_product(data.normal, data.V);
  const float inclination = 2.0f * acosf(__saturatef(NdotV)) * (1.0f / PI);

  const float u = (sqrtf(data.roughness) * 63.0f + 0.5f) / 64.0f;
  const float v = (inclination * 63.0f + 0.5f) / 64.0f;

  const float4 data0 = tex2D<float4>(device.ptrs.ltc_tex[0].tex, u, v);
  const float2 data1 = tex2D<float2>(device.ptrs.ltc_tex[1].tex, u, v);

  const vec3 x_axis = normalize_vector(add_vector(mul_vector(get_vector(-NdotV, -NdotV, -NdotV), data.normal), data.V));
  const vec3 y_axis = cross_product(data.normal, x_axis);

  LTCCoefficients coeffs;

  coeffs.world_to_shading_transformation.f11 = x_axis.x;
  coeffs.world_to_shading_transformation.f21 = y_axis.x;
  coeffs.world_to_shading_transformation.f31 = data.normal.x;
  coeffs.world_to_shading_transformation.f41 = 0.0f;
  coeffs.world_to_shading_transformation.f12 = x_axis.y;
  coeffs.world_to_shading_transformation.f22 = y_axis.y;
  coeffs.world_to_shading_transformation.f32 = data.normal.y;
  coeffs.world_to_shading_transformation.f42 = 0.0f;
  coeffs.world_to_shading_transformation.f13 = x_axis.z;
  coeffs.world_to_shading_transformation.f23 = y_axis.z;
  coeffs.world_to_shading_transformation.f33 = data.normal.z;
  coeffs.world_to_shading_transformation.f43 = 0.0f;
  coeffs.world_to_shading_transformation.f14 = -dot_product(data.position, x_axis);
  coeffs.world_to_shading_transformation.f24 = -dot_product(data.position, y_axis);
  coeffs.world_to_shading_transformation.f34 = -dot_product(data.position, data.normal);
  coeffs.world_to_shading_transformation.f44 = 1.0f;

  coeffs.shading_to_cosine_transformation.f11 = data0.x;
  coeffs.shading_to_cosine_transformation.f21 = 0.0f;
  coeffs.shading_to_cosine_transformation.f31 = data0.w;
  coeffs.shading_to_cosine_transformation.f12 = 0.0f;
  coeffs.shading_to_cosine_transformation.f22 = data0.z;
  coeffs.shading_to_cosine_transformation.f32 = 0.0f;
  coeffs.shading_to_cosine_transformation.f13 = -data0.y;
  coeffs.shading_to_cosine_transformation.f23 = 0.0f;
  coeffs.shading_to_cosine_transformation.f33 = data1.x;

  coeffs.albedo = data1.y;

  const float determinant_2x2 = data0.x * data1.x + data0.y * data0.w;

  coeffs.shading_to_cosine_determinant = data0.z * determinant_2x2;

  const float inv_determinant_2x2 = 1.0f / determinant_2x2;

  coeffs.cosine_to_shading_transformation.f11 = data1.x * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f21 = 0.0f;
  coeffs.cosine_to_shading_transformation.f31 = -data0.w * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f12 = 0.0f;
  coeffs.cosine_to_shading_transformation.f22 = 1.0f / data0.z;
  coeffs.cosine_to_shading_transformation.f32 = 0.0f;
  coeffs.cosine_to_shading_transformation.f13 = data0.y * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f23 = 0.0f;
  coeffs.cosine_to_shading_transformation.f33 = data0.x * inv_determinant_2x2;

  return coeffs;
}

__device__ vec3 ltc_get_edge_point_on_horizon(const vec3 below, const vec3 above) {
  const float t = -below.z / (above.z - below.z);

  return get_vector(below.x + t * (above.x - below.x), below.y + t * (above.y - below.y), 0.0f);
}

__device__ uint32_t ltc_clip_triangle(vec3& a, vec3& b, vec3& c, vec3& d) {
  uint32_t bit_mask = 0;
  bit_mask |= (a.z >= 0.0f) ? 0b001 : 0;
  bit_mask |= (b.z >= 0.0f) ? 0b010 : 0;
  bit_mask |= (c.z >= 0.0f) ? 0b100 : 0;

  switch (bit_mask) {
    case 0b111:
      return 3;
    case 0b110:
      d = ltc_get_edge_point_on_horizon(a, c);
      a = ltc_get_edge_point_on_horizon(a, b);
      return 4;
    case 0b101:
      d = c;
      c = ltc_get_edge_point_on_horizon(b, c);
      b = ltc_get_edge_point_on_horizon(b, a);
      return 4;
    case 0b011:
      d = ltc_get_edge_point_on_horizon(c, a);
      c = ltc_get_edge_point_on_horizon(c, b);
      return 4;
    case 0b100:
      a = ltc_get_edge_point_on_horizon(a, c);
      b = ltc_get_edge_point_on_horizon(b, c);
      return 3;
    case 0b010:
      a = ltc_get_edge_point_on_horizon(a, b);
      c = ltc_get_edge_point_on_horizon(c, b);
      return 3;
    case 0b001:
      b = ltc_get_edge_point_on_horizon(b, a);
      c = ltc_get_edge_point_on_horizon(c, a);
      return 3;
    case 0b000:
    default:
      return 0;
  }
}

__device__ float ltc_integrate_edge(const vec3 v0, const vec3 v1) {
  const vec3 v0_norm = normalize_vector(v0);
  const vec3 v1_norm = normalize_vector(v1);

  const float x = dot_product(v0_norm, v1_norm);
  const float y = fabsf(x);

  const float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
  const float b = 3.4175940f + (4.1616724f + y) * y;
  const float v = a / b;

  const float theta_sintheta = (x > 0.0f) ? v : 0.5f * (1.0f / sqrtf(fmaxf(1.0f - x * x, 1e-7f))) - v;

  return cross_product(v0_norm, v1_norm).z * theta_sintheta;
}

__device__ float ltc_integrate_triangle(uint32_t vertex_count, const vec3 a, const vec3 b, const vec3 c, const vec3 d) {
  float integral = 0.0f;
  integral += ltc_integrate_edge(a, b);
  integral += ltc_integrate_edge(b, c);
  integral += ltc_integrate_edge((vertex_count == 4) ? d : c, a);
  if (vertex_count == 4) {
    integral += ltc_integrate_edge(c, d);
  }

  return fabsf(integral);
}

__device__ float ltc_integrate(GBufferData data, LTCCoefficients coeffs, TriangleLight light) {
  float integral = 0.0f;

  vec3 tri_shading_a = transform_vec4_3_position(coeffs.world_to_shading_transformation, light.vertex);
  vec3 tri_shading_b = transform_vec4_3_position(coeffs.world_to_shading_transformation, add_vector(light.vertex, light.edge1));
  vec3 tri_shading_c = transform_vec4_3_position(coeffs.world_to_shading_transformation, add_vector(light.vertex, light.edge2));
  vec3 tri_shading_d;

  // Diffuse
  const uint32_t shading_vertex_count = ltc_clip_triangle(tri_shading_a, tri_shading_b, tri_shading_c, tri_shading_d);
  if (shading_vertex_count) {
    integral += ltc_integrate_triangle(shading_vertex_count, tri_shading_a, tri_shading_b, tri_shading_c, tri_shading_d)
                * luminance(opaque_color(data.albedo));
  }

  vec3 tri_cosine_a = transform_vec4_3_position(coeffs.world_to_shading_transformation, light.vertex);
  vec3 tri_cosine_b = transform_vec4_3_position(coeffs.world_to_shading_transformation, add_vector(light.vertex, light.edge1));
  vec3 tri_cosine_c = transform_vec4_3_position(coeffs.world_to_shading_transformation, add_vector(light.vertex, light.edge2));
  vec3 tri_cosine_d;

  tri_cosine_a = transform_vec3(coeffs.shading_to_cosine_transformation, tri_cosine_a);
  tri_cosine_b = transform_vec3(coeffs.shading_to_cosine_transformation, tri_cosine_b);
  tri_cosine_c = transform_vec3(coeffs.shading_to_cosine_transformation, tri_cosine_c);

  // Microfacet
  const uint32_t cosine_vertex_count = ltc_clip_triangle(tri_cosine_a, tri_cosine_b, tri_cosine_c, tri_cosine_d);
  if (cosine_vertex_count) {
    integral += ltc_integrate_triangle(cosine_vertex_count, tri_cosine_a, tri_cosine_b, tri_cosine_c, tri_cosine_d) * coeffs.albedo;
  }

  return integral;
}

#endif /* SHADING_KERNEL && !VOLUME_KERNEL */

#endif /* CU_LTC_H */
