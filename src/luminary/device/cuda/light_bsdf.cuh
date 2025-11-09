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
  return lerp(roughness, 1.0f, 0.05f);
}

LUMINARY_FUNCTION LightBSDFSampleResult light_bsdf_get_sample(const MaterialContextGeometry& mat_ctx, const ushort2 pixel) {
  // Transformation to +Z-Up
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = quaternion_apply(rotation_to_z, mat_ctx.V);
  const vec3 face_normal_local   = quaternion_apply(rotation_to_z, material_get_normal<MATERIAL_GEOMETRY_PARAM_FACE_NORMAL>(mat_ctx));

  // Material Context (+Z-Up)
  // TODO: Fuck this, don't do that.
  MaterialContextGeometry mat_ctx_local = mat_ctx;
  mat_ctx_local.V                       = V_local;
  mat_ctx_local.normal                  = get_vector(0.0f, 0.0f, 1.0f);

  const uint32_t base_substrate = mat_ctx_local.flags & MATERIAL_FLAG_BASE_SUBSTRATE_MASK;

  const bool include_refraction = (base_substrate == MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT);

  uint32_t num_techniques = 0;
  num_techniques += 1;                           // Microfacet reflection
  num_techniques += include_refraction ? 1 : 0;  // Microfacet refraction

  const float refraction_probability = (include_refraction) ? 1.0f / num_techniques : 0.0f;

  const float choice_random = random_1D(RANDOM_TARGET_LIGHT_BSDF_CHOICE, pixel);

  const uint32_t technique_id = (uint32_t) (choice_random * num_techniques);

  LightBSDFSampleTechnique technique = LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFLECTION;

  if (technique_id == 1 && include_refraction)
    technique = LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFRACTION;

  const float roughness          = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);
  const float sampling_roughness = light_bsdf_get_sampling_roughness(roughness);

  LightBSDFSampleResult result;
  switch (technique) {
    case LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFLECTION: {
      // TODO: Move things like bsdf_evaluate_core outside the switch
      const vec3 microfacet    = bsdf_microfacet_sample(V_local, sampling_roughness, pixel, RANDOM_TARGET_LIGHT_BSDF_DIRECTION);
      const vec3 ray           = reflect_vector(mat_ctx_local.V, microfacet);
      const BSDFRayContext ctx = bsdf_sample_context(mat_ctx_local, microfacet, ray, false);
      const float pdf          = bsdf_microfacet_pdf(V_local, sampling_roughness, ctx.NdotH, ctx.NdotV);
      const RGBF eval          = bsdf_evaluate_core(mat_ctx_local, ctx, BSDF_SAMPLING_GENERAL, ray, face_normal_local, 1.0f / pdf);

      result.ray                  = ray;
      result.weight               = eval;
      result.is_refraction        = false;
      result.sampling_probability = (1.0f - refraction_probability) * pdf;
    } break;
    case LIGHT_BSDF_SAMPLE_TECHNIQUE_MICROFACET_REFRACTION: {
      const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx_local);

      bool total_reflection;
      const vec3 microfacet    = bsdf_microfacet_refraction_sample(V_local, sampling_roughness, pixel, RANDOM_TARGET_LIGHT_BSDF_DIRECTION);
      const vec3 ray           = refract_vector(mat_ctx_local.V, microfacet, ior, total_reflection);
      const BSDFRayContext ctx = bsdf_sample_context(mat_ctx_local, microfacet, ray, !total_reflection);
      const float pdf =
        bsdf_microfacet_refraction_pdf(V_local, sampling_roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
      const RGBF eval = bsdf_evaluate_core(mat_ctx_local, ctx, BSDF_SAMPLING_GENERAL, ray, face_normal_local, 1.0f / pdf);

      result.ray                  = ray;
      result.weight               = eval;
      result.is_refraction        = !total_reflection;
      result.sampling_probability = refraction_probability * pdf;
    } break;
    default:
      // TODO: Make unreachable
      break;
  }

  result.ray = normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), result.ray));

  return result;
}

LUMINARY_FUNCTION float light_bsdf_get_probability(const MaterialContextGeometry& mat_ctx, const vec3 L) {
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = normalize_vector(quaternion_apply(rotation_to_z, mat_ctx.V));
  const vec3 L_local             = normalize_vector(quaternion_apply(rotation_to_z, L));

  // TODO: Fuck this, don't do that.
  MaterialContextGeometry mat_ctx_local = mat_ctx;
  mat_ctx_local.V                       = V_local;
  mat_ctx_local.normal                  = get_vector(0.0f, 0.0f, 1.0f);

  const uint32_t base_substrate = mat_ctx.flags & MATERIAL_FLAG_BASE_SUBSTRATE_MASK;

  const bool include_refraction = (base_substrate == MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT);

  uint32_t num_techniques = 0;
  num_techniques += 1;                           // Microfacet reflection
  num_techniques += include_refraction ? 1 : 0;  // Microfacet refraction

  const float refraction_probability = (include_refraction) ? 1.0f / num_techniques : 0.0f;

  const BSDFRayContext ctx = bsdf_evaluate_analyze(mat_ctx_local, L_local);

  const float roughness          = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);
  const float sampling_roughness = light_bsdf_get_sampling_roughness(roughness);

  float sampling_probability;
  if (ctx.is_refraction) {
    const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx_local);
    const float microfacet_refraction_pdf =
      bsdf_microfacet_refraction_pdf(V_local, sampling_roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);

    sampling_probability = refraction_probability * microfacet_refraction_pdf;
  }
  else {
    const float microfacet_reflection_pdf = bsdf_microfacet_pdf(V_local, sampling_roughness, ctx.NdotH, ctx.NdotV);

    sampling_probability = (1.0f - refraction_probability) * microfacet_reflection_pdf;
  }

  return sampling_probability;
}

#endif /* CU_LUMINARY_LIGHT_BSDF_H */
