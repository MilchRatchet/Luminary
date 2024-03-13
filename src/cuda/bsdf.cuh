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
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;

  vec3 reflection_vector, refraction_vector;

  // For every refraction exists a corresponding reflection vector, we always compute the BSDF based on the reflection.
  if (context.is_refraction) {
    context.H = bsdf_refraction_normal_from_pair(L, data.V, ior_out, ior_in);

    reflection_vector = reflect_vector(scale_vector(data.V, -1.0f), context.H);
    refraction_vector = L;

    context.NdotL = dot_product(data.normal, reflection_vector);
  }
  else {
    context.H = normalize_vector(add_vector(data.V, L));

    reflection_vector = L;
    refraction_vector = refract_ray(scale_vector(data.V, -1.0f), context.H, ior_in / ior_out);
  }

  context.NdotH     = dot_product(data.normal, context.H);
  const float HdotV = __saturatef(dot_product(context.H, data.V));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), HdotV);

  // TODO: Dielectric

  return context;
}

__device__ RGBF bsdf_evaluate_core(
  const GBufferData data, const BSDFRayContext context, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf = 1.0f) {
  return bsdf_multiscattering_evaluate(data, context, sampling_hint, one_over_sampling_pdf);
}

__device__ RGBF
  bsdf_evaluate(const GBufferData data, const vec3 L, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf = 1.0f) {
  const BSDFRayContext context = bsdf_evaluate_analyze(data, L);

  return bsdf_evaluate_core(data, context, sampling_hint, one_over_sampling_pdf);
}

__device__ BSDFRayContext bsdf_sample_context(const GBufferData data, const vec3 H, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = dot_product(data.normal, data.V);

  const float ambient_ior = bsdf_refraction_index_ambient(data);
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;

  context.H = H;

  // vec3 refraction_vector;
  // if (context.is_refraction) {
  //   const vec3 reflection_vector = reflect_vector(scale_vector(data.V, -1.0f), context.H);
  //   refraction_vector            = L;
  //
  //  context.NdotL = dot_product(data.normal, reflection_vector);
  //}
  // else {
  //  refraction_vector = refract_ray(scale_vector(data.V, -1.0f), context.H, ior_in / ior_out);
  //}

  context.NdotH     = dot_product(data.normal, context.H);
  const float HdotV = __saturatef(dot_product(context.H, data.V));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), HdotV);

  // TODO: Dielectric

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
  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_TBD1, pixel) < data.metallic) {
    vec3 sampled_microfacet = get_vector(0.0f, 0.0f, 1.0f);
    ray_local               = bsdf_microfacet_sample(data_local, pixel, sampled_microfacet);

    const BSDFRayContext context = bsdf_sample_context(data_local, sampled_microfacet, ray_local);

    info.weight              = bsdf_conductor(data_local, context, BSDF_SAMPLING_MICROFACET, 1.0f);
    info.is_microfacet_based = true;
  }
  else {
    vec3 sampled_microfacet             = get_vector(0.0f, 0.0f, 1.0f);
    const vec3 microfacet_ray           = bsdf_microfacet_sample(data_local, pixel, sampled_microfacet);
    const float microfacet_pdf          = bsdf_microfacet_pdf_reflection(data_local, microfacet_ray);
    const float microfacet_pdf_diffuse  = bsdf_diffuse_pdf(data_local, microfacet_ray);
    const float microfacet_mis_weight   = (data_local.roughness < 0.1f) ? 0.5f : microfacet_pdf / (microfacet_pdf + microfacet_pdf_diffuse);
    const BSDFRayContext microfacet_ctx = bsdf_sample_context(data_local, sampled_microfacet, microfacet_ray);
    const RGBF microfacet_eval          = bsdf_glossy(data_local, microfacet_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

    const vec3 diffuse_ray             = bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_REFLECTION + 1, pixel));
    const float diffuse_pdf            = bsdf_diffuse_pdf(data_local, diffuse_ray);
    const float diffuse_pdf_microfacet = bsdf_microfacet_pdf(data_local, diffuse_ray);
    const float diffuse_mis_weight     = (data_local.roughness < 0.1f) ? 0.5f : diffuse_pdf / (diffuse_pdf + diffuse_pdf_microfacet);
    const BSDFRayContext diffuse_ctx   = bsdf_sample_context(data_local, get_vector(0.0f, 0.0f, 1.0f), diffuse_ray);
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

  return normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
}

#endif /* CU_BSDF_H */
