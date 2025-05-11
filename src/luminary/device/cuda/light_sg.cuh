#ifndef CU_LUMINARY_LIGHT_SG_H
#define CU_LUMINARY_LIGHT_SG_H

#include "math.cuh"
#include "utils.cuh"

struct LightSGData {
  float diffuse_weight;
  float reflection_weight;
  Quaternion rotation_to_z;
  float proj_roughness_u2;
  float proj_roughness_v2;
  vec3 V_local;
  float JJT00;
  float JJT01;
  float JJT11;
  float det_JJT4;
  vec3 reflection_vec;
} typedef LightSGData;

// Code taken from the SG lighting demo distributed with [Tok24].

#define LIGHT_SG_VARIANCE_THRESHOLD (0x1.0p-31f)
#define LIGHT_SG_INV_SQRTPI (0.56418958354775628694807945156077f)
#define LIGHT_SG_UNCERTAINTY_AGGRESSIVENESS (0.7f)

__device__ float light_sg_integral(const float sharpness) {
  return 4.0f * PI * expm1_over_x(-2.0f * sharpness);
}

// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
__device__ float light_vmf_integral(const float cosine, const float sharpness) {
  // Interpolation factor [Tokuyoshi 2022].
  const float A          = 0.6517328826907056171791055021459f;
  const float B          = 1.3418280033141287699294252888649f;
  const float C          = 7.2216687798956709087860872386955f;
  const float steepness  = sharpness * sqrtf((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
  const float lerpFactor = __saturatef(0.5f + 0.5f * (erff(steepness * clampf(cosine, -1.0f, 1.0f)) / erff(steepness)));

  // Interpolation between upper and lower hemispherical integrals .
  const float e = expf(-sharpness);

  return lerp(e, 1.0f, lerpFactor) / (e + 1.0f);
}

__device__ float light_sg_cosine_integral_upper_hemisphere(const float sharpness) {
  if (sharpness <= 0.5) {
    // Taylor-series approximation for the numerical stability.
    return (((((((-1.0f / 362880.0f) * sharpness + 1.0f / 40320.0f) * sharpness - 1.0f / 5040.0f) * sharpness + 1.0f / 720.0f) * sharpness
              - 1.0f / 120.0f)
               * sharpness
             + 1.0f / 24.0f)
              * sharpness
            - 1.0f / 6.0f)
             * sharpness
           + 0.5f;
  }

  return (expm1f(-sharpness) + sharpness) / (sharpness * sharpness);
}

__device__ float light_sg_cosine_integral_lower_hemisphere(const float sharpness) {
  const float e = expf(-sharpness);

  if (sharpness <= 0.5f) {
    // Taylor-series approximation for the numerical stability.
    return e
           * (((((((((1.0f / 403200.0f) * sharpness - 1.0f / 45360.0f) * sharpness + 1.0f / 5760.0f) * sharpness - 1.0f / 840.0f) * sharpness + 1.0f / 144.0f) * sharpness - 1.0f / 30.0f) * sharpness + 1.0f / 8.0f) * sharpness - 1.0f / 3.0f) * sharpness + 0.5f);
  }

  return e * (-expm1f(-sharpness) - sharpness * e) / (sharpness * sharpness);
}

__device__ float light_sg_cosine_integral(const float z, const float sharpness) {
  // Fitted approximation for t(sharpness).
  const float A  = 2.7360831611272558028247203765204f;
  const float B  = 17.02129778174187535455530451145f;
  const float C  = 4.0100826728510421403939290030394f;
  const float D  = 15.219156263147210594866010069381f;
  const float E  = 76.087896272360737270901154261082f;
  const float t  = sharpness * sqrtf(0.5f * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
  const float tz = t * z;

  const float lerp_factor = __saturatef(
    FLT_EPSILON * 0.5f + 0.5f * (z * erfcf(-tz) + erfcf(t))
    - 0.5f * LIGHT_SG_INV_SQRTPI * expf(-tz * tz) * expm1f(t * t * (z * z - 1.0f)) / t);

  // Interpolation between upper and lower hemispherical integrals.
  const float lowerIntegral = light_sg_cosine_integral_lower_hemisphere(sharpness);
  const float upperIntegral = light_sg_cosine_integral_upper_hemisphere(sharpness);

  return 2.0f * lerp(lowerIntegral, upperIntegral, lerp_factor);
}

__device__ float light_sg_ggx(const vec3 m, const float mat00, const float mat01, const float mat11) {
  const float det = fmaxf(mat00 * mat11 - mat01 * mat01, eps);

  const float m_roughness_x = mat11 * m.x - mat01 * m.y;
  const float m_roughness_y = -mat01 * m.x + mat00 * m.y;

  const float dot = m.x * m_roughness_x + m.y * m_roughness_y;

  const float length2 = dot / det + m.z * m.z;

  return 1.0f / (PI * sqrtf(det) * length2 * length2);
}

// Reflection lobe based the symmetric GGX VNDF.
__device__ float light_sg_ggx_reflection_pdf(const vec3 wi, const vec3 m, const float mat00, const float mat01, const float mat11) {
  const float ggx = light_sg_ggx(m, mat00, mat01, mat11);

  const float wi_roughness_x = mat00 * wi.x + mat01 * wi.y;
  const float wi_roughness_y = mat01 * wi.x + mat11 * wi.y;

  const float dot = wi.x * wi_roughness_x + wi.y * wi_roughness_y;

  return ggx / (4.0f * sqrtf(dot + wi.z * wi.z));
}

__device__ LightSGData light_sg_prepare(const GBufferData data) {
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(data.normal);

  // This is already accounting for future anisotropy support.
  const float roughness_u = data.roughness;
  const float roughness_v = data.roughness;

  // Convert the roughness from slope space to projected space.
  const float roughness_u2      = roughness_u * roughness_u;
  const float roughness_v2      = roughness_v * roughness_v;
  const float proj_roughness_u2 = roughness_u2 / fmaxf(1.0f - roughness_u2, eps);
  const float proj_roughness_v2 = roughness_v2 / fmaxf(1.0f - roughness_v2, eps);

  // Compute the Jacobian J for the transformation between halfvetors and reflection vectors at halfvector = normal.
  const vec3 V_local         = quaternion_apply(rotation_to_z, data.V);
  const float V_local_length = sqrtf(V_local.x * V_local.x + V_local.y * V_local.y);
  const float view_x         = (V_local_length != 0.0f) ? V_local.x / V_local_length : 1.0f;
  const float view_y         = (V_local_length != 0.0f) ? V_local.y / V_local_length : 0.0f;

  const float reflection_jacobian00 = 0.5f * view_x;
  const float reflection_jacobian01 = -0.5f * view_y / V_local.z;
  const float reflection_jacobian10 = 0.5f * view_y;
  const float reflection_jacobian11 = 0.5f * view_x / V_local.z;

  // Compute JJ^T matrix.
  const float JJT00 = reflection_jacobian00 * reflection_jacobian00 + reflection_jacobian01 * reflection_jacobian01;
  const float JJT01 = reflection_jacobian00 * reflection_jacobian10 + reflection_jacobian01 * reflection_jacobian11;
  const float JJT11 = reflection_jacobian10 * reflection_jacobian10 + reflection_jacobian11 * reflection_jacobian11;

  const float det_JJT4 = 1.0f / (4.0f * V_local.z * V_local.z);  // = 4 * determiant(JJ^T).

  // Preprocess for the lobe visibility.
  // Approximate the reflection lobe with an SG whose axis is the perfect specular reflection vector.
  // We use a conservative sharpness to filter the visibility.
  const float roughness_max2       = fmaxf(roughness_u2, roughness_v2);
  const float reflection_sharpness = (1.0f - roughness_max2) / fmaxf(2.0f * roughness_max2, eps);
  const vec3 reflection_vec        = scale_vector(reflect_vector(data.V, data.normal), reflection_sharpness);

  LightSGData sg_data;
  sg_data.diffuse_weight    = (GBUFFER_IS_SUBSTRATE_TRANSLUCENT(data.flags) || (data.flags & G_BUFFER_FLAG_METALLIC)) ? 0.0f : 1.0f;
  sg_data.reflection_weight = 1.0f;
  sg_data.rotation_to_z     = rotation_to_z;
  sg_data.proj_roughness_u2 = proj_roughness_u2;
  sg_data.proj_roughness_v2 = proj_roughness_v2;
  sg_data.V_local           = V_local;
  sg_data.JJT00             = JJT00;
  sg_data.JJT01             = JJT01;
  sg_data.JJT11             = JJT11;
  sg_data.det_JJT4          = det_JJT4;
  sg_data.reflection_vec    = reflection_vec;

  return sg_data;
}

// #define LIGHT_TREE_NO_MICROFACET_VARIANCE_FACTOR

// In our case, all light clusters are considered omnidirectional, i.e. sharpness = 0.
__device__ float light_sg_evaluate(
  const LightSGData data, const vec3 position, const vec3 normal, const vec3 mean, const float variance, const float power,
  float& secondary_value) {
  const vec3 light_vec    = sub_vector(mean, position);
  const float distance_sq = dot_product(light_vec, light_vec);
  const vec3 light_dir    = scale_vector(light_vec, rsqrt(distance_sq));

  // Clamp the variance for the numerical stability.
  const float light_variance = fmaxf(variance, LIGHT_SG_VARIANCE_THRESHOLD * distance_sq);

  float emissive_diffuse = power / light_variance;

#ifdef LIGHT_TREE_NO_MICROFACET_VARIANCE_FACTOR
  const float emissive_microfacet = power;
#else
  const float emissive_microfacet = power / light_variance;
#endif

  // Compute SG sharpness for a light distribution viewed from the shading point.
  const float light_sharpness = distance_sq / light_variance;

  secondary_value = (distance_sq < light_variance) ? 1.0f : 0.0f;

  // Axis of the SG product lobe.
  const vec3 product_vec          = add_vector(data.reflection_vec, scale_vector(light_dir, light_sharpness));
  const float product_sharpness   = get_length(product_vec);
  const vec3 product_dir          = scale_vector(product_vec, 1.0f / product_sharpness);
  const float light_lobe_variance = 1.0f / light_sharpness;

  const float filtered_proj_roughness00 = data.proj_roughness_u2 + 2.0f * light_lobe_variance * data.JJT00;
  const float filtered_proj_roughness01 = 2.0f * light_lobe_variance * data.JJT01;
  const float filtered_proj_roughness11 = data.proj_roughness_v2 + 2.0f * light_lobe_variance * data.JJT11;

  // Compute determinant(filteredProjRoughnessMat) in a numerically stable manner.
  const float det = data.proj_roughness_u2 * data.proj_roughness_v2
                    + 2.0f * light_lobe_variance * (data.proj_roughness_u2 * data.JJT00 + data.proj_roughness_v2 * data.JJT11)
                    + light_lobe_variance * light_lobe_variance * data.det_JJT4;

  // NDF filtering in a numerically stable manner.
  const float tr = filtered_proj_roughness00 + filtered_proj_roughness11;

  const float denom       = 1.0f / (1.0f + tr + det);
  const bool denom_finite = is_non_finite(denom) == false;

  const float filtered_roughness_mat00 = (denom_finite)
                                           ? fminf(filtered_proj_roughness00 + det, FLT_MAX) * denom
                                           : fminf(filtered_proj_roughness00, FLT_MAX) / fminf(filtered_proj_roughness00 + 1.0f, FLT_MAX);
  const float filtered_roughness_mat01 = (denom_finite) ? fminf(filtered_proj_roughness01, FLT_MAX) * denom : 0.0f;
  const float filtered_roughness_mat11 = (denom_finite)
                                           ? fminf(filtered_proj_roughness11 + det, FLT_MAX) * denom
                                           : fminf(filtered_proj_roughness11, FLT_MAX) / fminf(filtered_proj_roughness11 + 1.0f, FLT_MAX);

  // Evaluate the filtered distribution.
  const vec3 H            = add_vector(data.V_local, quaternion_apply(data.rotation_to_z, light_dir));
  const vec3 H_normalized = scale_vector(H, 1.0f / fmaxf(get_length(H), eps));
  const float pdf =
    light_sg_ggx_reflection_pdf(data.V_local, H_normalized, filtered_roughness_mat00, filtered_roughness_mat01, filtered_roughness_mat11);

  // Microfacet reflection importance.
  const float visibility            = light_vmf_integral(dot_product(product_dir, normal), product_sharpness);
  const float reflection_importance = visibility * pdf * light_sg_integral(light_sharpness);

  // Diffuse importance.
  const float cosine             = clampf(dot_product(light_dir, normal), -1.0f, 1.0f);
  const float diffuse_importance = light_sg_cosine_integral(cosine, light_sharpness);

  return emissive_diffuse * data.diffuse_weight * diffuse_importance + emissive_microfacet * data.reflection_weight * reflection_importance;
}

#endif /* CU_LUMINARY_LIGHT_SG_H */
