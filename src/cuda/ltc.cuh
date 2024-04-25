#ifndef CU_LTC_H
#define CU_LTC_H

#include "bsdf_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

//
// This implementation is based on https://github.com/ishaanshah/risltc.
//

struct LTCCoefficients {
  Mat3x3 shading_to_cosine_transformation;
  Mat3x3 cosine_to_shading_transformation;
  float albedo;
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
  coeffs.shading_to_cosine_transformation.f13 = data0.y;
  coeffs.shading_to_cosine_transformation.f23 = 0.0f;
  coeffs.shading_to_cosine_transformation.f33 = data1.x;

  coeffs.albedo = data1.y;

  const float determinant_2x2 = data0.x * data1.x - data0.y * data0.w;

  coeffs.shading_to_cosine_determinant = data0.z * determinant_2x2;

  const float inv_determinant_2x2 = 1.0f / determinant_2x2;

  coeffs.cosine_to_shading_transformation.f11 = data1.x * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f11 = 0.0f;
  coeffs.cosine_to_shading_transformation.f11 = -data0.w * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f11 = 0.0f;
  coeffs.cosine_to_shading_transformation.f11 = 1.0f / data0.z;
  coeffs.cosine_to_shading_transformation.f11 = 0.0f;
  coeffs.cosine_to_shading_transformation.f11 = -data0.y * inv_determinant_2x2;
  coeffs.cosine_to_shading_transformation.f11 = 0.0f;
  coeffs.cosine_to_shading_transformation.f11 = data0.x * inv_determinant_2x2;

  return coeffs;
}

#endif /* CU_LTC_H */
