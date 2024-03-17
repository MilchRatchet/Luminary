#ifndef CU_BSDF_H
#define CU_BSDF_H

#include "bsdf_utils.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ BSDFRayContext bsdf_evaluate_analyze(const GBufferData data, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = dot_product(data.normal, data.V);

  // Refraction rays can leave the surface and reflections could enter the surface, we hack here and simply swap their meaning around based
  // on if they enter or leave, otherwise it is not clear which of the two a ray should be.
  context.is_refraction = (context.NdotL < 0.0f);

  const float ambient_ior = bsdf_refraction_index_ambient(data);
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;

  context.refraction_index = ior_in / ior_out;

  vec3 refraction_vector;
  if (context.is_refraction) {
    context.H         = bsdf_refraction_normal_from_pair(L, data.V, ior_out, ior_in);
    refraction_vector = L;
  }
  else {
    context.H         = normalize_vector(add_vector(data.V, L));
    refraction_vector = refract_vector(scale_vector(data.V, -1.0f), context.H, context.refraction_index);
  }

  context.NdotH = dot_product(data.normal, context.H);
  context.HdotV = dot_product(context.H, data.V);
  context.HdotL = dot_product(context.H, L);

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), context.HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), context.HdotV);

  context.f0_dielectric      = bsdf_fresnel_normal_incidence(ior_in, ior_out);
  context.fresnel_dielectric = bsdf_fresnel(context.H, data.V, refraction_vector, ior_in, ior_out);

  return context;
}

__device__ RGBF bsdf_evaluate_core(
  const GBufferData data, const BSDFRayContext context, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf = 1.0f) {
  return bsdf_multiscattering_evaluate(data, context, sampling_hint, one_over_sampling_pdf);
}

__device__ RGBF bsdf_evaluate(
  const GBufferData data, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_transparent_pass,
  const float one_over_sampling_pdf = 1.0f) {
  const BSDFRayContext context = bsdf_evaluate_analyze(data, L);

  is_transparent_pass = context.is_refraction;

  return bsdf_evaluate_core(data, context, sampling_hint, one_over_sampling_pdf);
}

__device__ BSDFRayContext bsdf_sample_context(const GBufferData data, const vec3 H, const vec3 L, const bool is_refraction) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = fabsf(dot_product(data.normal, data.V));

  context.is_refraction = is_refraction;

  if (is_refraction)
    context.NdotL *= -1.0f;

  const float ambient_ior = bsdf_refraction_index_ambient(data);
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;

  context.refraction_index = ior_in / ior_out;

  context.H = H;

  const vec3 refraction_vector =
    (context.is_refraction) ? L : refract_vector(scale_vector(data.V, -1.0f), context.H, context.refraction_index);

  context.NdotH = fabsf(dot_product(data.normal, context.H));
  context.HdotV = fabsf(dot_product(context.H, data.V));
  context.HdotL = fabsf(dot_product(context.H, L));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), context.HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), context.HdotV);

  context.f0_dielectric      = bsdf_fresnel_normal_incidence(ior_in, ior_out);
  context.fresnel_dielectric = bsdf_fresnel(context.H, data.V, refraction_vector, ior_in, ior_out);

  return context;
}

__device__ vec3 bsdf_sample(const GBufferData data, const ushort2 pixel, BSDFSampleInfo& info) {
  // Transformation to +Z-Up
  const Quaternion rotation_to_z = get_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(data.V, rotation_to_z);

  // G Buffer Data (+Z-Up)
  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  info.is_transparent_pass = false;
  info.is_microfacet_based = false;

  vec3 ray_local;

  const vec3 sampled_microfacet = bsdf_microfacet_sample(data_local, pixel);

  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_TBD1 + 1, pixel) < data.albedo.a) {
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_TBD1, pixel) < data.metallic) {
      ray_local = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);

      const BSDFRayContext context = bsdf_sample_context(data_local, sampled_microfacet, ray_local, false);

      info.weight              = bsdf_conductor(data_local, context, BSDF_SAMPLING_MICROFACET, 1.0f);
      info.is_microfacet_based = true;
    }
    else {
      const vec3 microfacet_ray           = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);
      const BSDFRayContext microfacet_ctx = bsdf_sample_context(data_local, sampled_microfacet, microfacet_ray, false);
      const float microfacet_pdf          = bsdf_microfacet_pdf(data_local, microfacet_ctx.NdotH, microfacet_ctx.NdotV);
      const float microfacet_pdf_diffuse  = bsdf_diffuse_pdf(data_local, microfacet_ctx.NdotL);
      const float microfacet_mis_weight = (data_local.roughness < 0.1f) ? 0.5f : microfacet_pdf / (microfacet_pdf + microfacet_pdf_diffuse);
      const RGBF microfacet_eval        = bsdf_glossy(data_local, microfacet_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

      const vec3 diffuse_ray             = bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_REFLECTION + 1, pixel));
      const vec3 diffuse_microfacet      = normalize_vector(add_vector(data.V, diffuse_ray));
      const BSDFRayContext diffuse_ctx   = bsdf_sample_context(data_local, diffuse_microfacet, diffuse_ray, false);
      const float diffuse_pdf            = bsdf_diffuse_pdf(data_local, diffuse_ctx.NdotL);
      const float diffuse_pdf_microfacet = bsdf_microfacet_pdf(data_local, diffuse_ctx.NdotH, diffuse_ctx.NdotV);
      const float diffuse_mis_weight     = (data_local.roughness < 0.1f) ? 0.5f : diffuse_pdf / (diffuse_pdf + diffuse_pdf_microfacet);
      const RGBF diffuse_eval            = bsdf_glossy(data_local, diffuse_ctx, BSDF_SAMPLING_DIFFUSE, 1.0f);

      const float microfacet_weight = microfacet_mis_weight * luminance(microfacet_eval);
      const float diffuse_weight    = diffuse_mis_weight * luminance(diffuse_eval);

      const float sum_weights = microfacet_weight + diffuse_weight;

      const float microfacet_probability = microfacet_weight / sum_weights;
      const float diffuse_probability    = diffuse_weight / sum_weights;

      RGBF final_weight;
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_CHOICE, pixel) < microfacet_probability) {
        ray_local                = microfacet_ray;
        final_weight             = scale_color(microfacet_eval, 1.0f / microfacet_probability);
        info.is_microfacet_based = true;
      }
      else {
        ray_local                = diffuse_ray;
        final_weight             = scale_color(diffuse_eval, 1.0f / diffuse_probability);
        info.is_microfacet_based = false;
      }

      info.weight = final_weight;
    }
  }
  else {
    const float ambient_ior = bsdf_refraction_index_ambient(data_local);
    const float ior_in      = (data_local.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data_local.refraction_index : ambient_ior;
    const float ior_out     = (data_local.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data_local.refraction_index;

    const vec3 reflection_vector        = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);
    const BSDFRayContext reflection_ctx = bsdf_sample_context(data_local, sampled_microfacet, reflection_vector, false);
    const RGBF reflection_eval          = bsdf_dielectric(data_local, reflection_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

    const vec3 refraction_vector        = refract_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet, ior_in / ior_out);
    const BSDFRayContext refraction_ctx = bsdf_sample_context(data_local, sampled_microfacet, refraction_vector, true);
    const RGBF refraction_eval          = bsdf_dielectric(data_local, refraction_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

    const float reflection_weight = luminance(reflection_eval);
    const float refraction_weight = luminance(refraction_eval);

    const float sum_weights = reflection_weight + refraction_weight;

    const float reflection_probability = reflection_weight / sum_weights;
    const float refraction_probability = refraction_weight / sum_weights;

    RGBF final_weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_TBD1 + 2, pixel) < reflection_probability) {
      ray_local    = reflection_vector;
      final_weight = scale_color(reflection_eval, 0.5f * 1.0f / reflection_probability);
    }
    else {
      ray_local                = refraction_vector;
      final_weight             = scale_color(refraction_eval, 0.5f * 1.0f / refraction_probability);
      info.is_transparent_pass = true;
    }

    info.weight              = final_weight;
    info.is_microfacet_based = true;
  }

  return normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
}

#endif /* CU_BSDF_H */
