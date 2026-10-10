#ifndef CU_LUMINARY_LIGHT_BSDF_H
#define CU_LUMINARY_LIGHT_BSDF_H

#include "bsdf.cuh"
#include "bsdf_utils.cuh"
#include "light_common.cuh"
#include "material.cuh"
#include "math.cuh"
#include "utils.cuh"

enum LightBSDFSampleTechnique {
  LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFLECTION,
  LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFRACTION
} typedef LightBSDFSampleTechnique;

LUMINARY_FUNCTION float light_bsdf_get_sampling_roughness(const float roughness) {
  return lerp(roughness, 1.0f, 0.04f);
}

LUMINARY_FUNCTION float light_bsdf_get_russian_roulette_probability(const float roughness) {
  return remap01(roughness, 0.5f, 0.1f);
}

LUMINARY_FUNCTION void light_bsdf_get_technique_probabilities(
  const MaterialParams& params, float& reflection_probability, float& refraction_probability) {
  const uint32_t base_substrate = params.flags & MATERIAL_FLAG_BASE_SUBSTRATE_MASK;
  const float ior               = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(params);
  const bool include_refraction = (base_substrate == MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT) && (ior != 1.0f);

  refraction_probability = (include_refraction) ? 0.5f : 0.0f;
  reflection_probability = 1.0f - refraction_probability;
}

LUMINARY_FUNCTION float light_bsdf_get_directional_pdf(
  const MaterialParams& params, const vec3 V, const vec3 L, const float roughness, const float reflection_probability,
  const float refraction_probability) {
  const BSDFRayContext ctx = bsdf_evaluate_analyze(params, get_vector(0.0f, 0.0f, 1.0f), V, L);

  if (ctx.is_refraction) {
    if (refraction_probability == 0.0f)
      return 0.0f;

    const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(params);
    return refraction_probability
           * bsdf_microfacet_refraction_pdf(V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
  }

  float pdf = reflection_probability * bsdf_microfacet_pdf(V, roughness, ctx.NdotH, ctx.NdotV);

  if (refraction_probability > 0.0f) {
    const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(params);
    const vec3 H    = bsdf_normal_from_pair(L, V, 1.0f);

    bool total_reflection;
    (void) refract_vector(V, H, ior, total_reflection);

    if (total_reflection) {
      pdf += refraction_probability * bsdf_microfacet_refraction_tir_reflection_pdf(roughness, ctx.NdotH, ctx.NdotV);
    }
  }

  return pdf;
}

LUMINARY_FUNCTION LightBSDFSampleResult light_bsdf_get_sample(const MaterialContextGeometry& mat_ctx, const PathID& path_id) {
  // Transformation to +Z-Up
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = normalize_vector(quaternion_apply(rotation_to_z, mat_ctx.V));
  const vec3 face_normal_local   = quaternion_apply(rotation_to_z, normal_unpack(mat_ctx.face_normal));

  float reflection_probability, refraction_probability;
  light_bsdf_get_technique_probabilities(mat_ctx.params, reflection_probability, refraction_probability);

  const float choice_random = random_1D(RANDOM_TARGET_LIGHT_BSDF_CHOICE, path_id);

  LightBSDFSampleTechnique technique = LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFLECTION;

  if (choice_random >= reflection_probability)
    technique = LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFRACTION;

  const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx.params);

  const float randomRR                     = random_1D(RANDOM_TARGET_LIGHT_BSDF_RR, path_id);
  const float russian_roulette_probability = light_bsdf_get_russian_roulette_probability(roughness);

  if (randomRR >= russian_roulette_probability) {
    LightBSDFSampleResult result;
    result.sampling_probability = 0.0f;

    return result;
  }

  const float sampling_roughness = light_bsdf_get_sampling_roughness(roughness);

  vec3 microfacet;
  vec3 ray;
  bool is_refraction;
  switch (technique) {
    case LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFLECTION: {
      microfacet    = bsdf_microfacet_sample(V_local, sampling_roughness, path_id, RANDOM_TARGET_LIGHT_BSDF_DIRECTION);
      ray           = reflect_vector(V_local, microfacet);
      is_refraction = false;
    } break;
    case LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFRACTION: {
      const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx.params);

      bool total_reflection;
      microfacet    = bsdf_microfacet_refraction_sample(V_local, sampling_roughness, path_id, RANDOM_TARGET_LIGHT_BSDF_DIRECTION);
      ray           = refract_vector(V_local, microfacet, ior, total_reflection);
      is_refraction = total_reflection == false;
    } break;
  }

  // TODO: We build two contexts in a row here. Consolidate that.
  const BSDFRayContext ctx = bsdf_sample_context(mat_ctx.params, get_vector(0.0f, 0.0f, 1.0f), V_local, microfacet, ray, is_refraction);
  const float pdf =
    light_bsdf_get_directional_pdf(mat_ctx.params, V_local, ray, sampling_roughness, reflection_probability, refraction_probability);
  const RGBF eval =
    (pdf > 0.0f) ? bsdf_evaluate_core(mat_ctx.params, ctx, BSDF_SAMPLING_GENERAL, ray, face_normal_local, 1.0f / pdf) : splat_color(0.0f);

  LightBSDFSampleResult result;
  result.ray                  = ray;
  result.weight               = eval;
  result.is_refraction        = is_refraction;
  result.sampling_probability = pdf;

  result.weight = scale_color(result.weight, 1.0f / russian_roulette_probability);
  result.sampling_probability *= russian_roulette_probability;

  result.ray = normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), result.ray));

  return result;
}

LUMINARY_FUNCTION float light_bsdf_get_probability(const MaterialContextGeometry& mat_ctx, const vec3 L) {
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = normalize_vector(quaternion_apply(rotation_to_z, mat_ctx.V));
  const vec3 L_local             = normalize_vector(quaternion_apply(rotation_to_z, L));

  float reflection_probability, refraction_probability;
  light_bsdf_get_technique_probabilities(mat_ctx.params, reflection_probability, refraction_probability);

  const float roughness          = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx.params);
  const float sampling_roughness = light_bsdf_get_sampling_roughness(roughness);
  float sampling_probability =
    light_bsdf_get_directional_pdf(mat_ctx.params, V_local, L_local, sampling_roughness, reflection_probability, refraction_probability);

  const float russian_roulette_probability = light_bsdf_get_russian_roulette_probability(roughness);
  sampling_probability *= russian_roulette_probability;

  return sampling_probability;
}

#endif /* CU_LUMINARY_LIGHT_BSDF_H */
