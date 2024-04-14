#ifndef CU_BSDF_H
#define CU_BSDF_H

#include "bsdf_utils.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ BSDFRayContext bsdf_evaluate_analyze(const GBufferData data, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = fabsf(dot_product(data.normal, data.V));

  // Refraction rays can leave the surface and reflections could enter the surface, we hack here and simply swap their meaning around based
  // on if they enter or leave, otherwise it is not clear which of the two a ray should be.
  context.is_refraction = (context.NdotL < 0.0f);

  context.NdotL = fabsf(context.NdotL);

  const float ior_in  = fminf(3.0f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(3.0f, fmaxf(1.0f, data.ior_out));

  context.refraction_index = ior_in / ior_out;

  vec3 refraction_vector;
  if (context.is_refraction) {
    context.H = bsdf_refraction_normal_from_pair(L, data.V, ior_out, ior_in);
  }
  else {
    context.H = normalize_vector(add_vector(data.V, L));
  }

  refraction_vector = refract_vector(scale_vector(data.V, -1.0f), context.H, context.refraction_index);

  // Invalidate refraction rays that are not possible
  if (context.is_refraction && dot_product(L, refraction_vector) < 1.0f - 64.0f * eps) {
    context.NdotL = -1.0f;
  }

  context.NdotH = fabsf(dot_product(data.normal, context.H));
  context.HdotV = fabsf(dot_product(context.H, data.V));
  context.HdotL = fabsf(dot_product(context.H, L));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), context.HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), context.HdotV);

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

  context.NdotL = fabsf(context.NdotL);

  const float ior_in  = fminf(3.0f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(3.0f, fmaxf(1.0f, data.ior_out));

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

  context.fresnel_dielectric = bsdf_fresnel(context.H, data.V, refraction_vector, ior_in, ior_out);

  return context;
}

__device__ vec3 bsdf_sample(const GBufferData data, const ushort2 pixel, BSDFSampleInfo& info, float& marginal) {
  if (data.flags & G_BUFFER_VOLUME_HIT) {
    const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE, pixel);
    const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BOUNCE_DIR, pixel);

    const vec3 ray = scale_vector(data.V, -1.0f);

    const vec3 scatter_ray = (VOLUME_HIT_TYPE(data.hit_id) != VOLUME_TYPE_OCEAN)
                               ? jendersie_eon_phase_sample(ray, data.roughness, random_dir, random_choice)
                               : ocean_phase_sampling(ray, random_dir, random_choice);

    const float cos_angle = -dot_product(scatter_ray, data.V);

    marginal = (VOLUME_HIT_TYPE(data.hit_id) != VOLUME_TYPE_OCEAN) ? jendersie_eon_phase_function(cos_angle, data.roughness)
                                                                   : ocean_phase(cos_angle);

    return scatter_ray;
  }

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

  vec3 sampled_microfacet_refraction;
  float choice_probability = 1.0f;
  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_ALPHA, pixel) < data.albedo.a) {
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_METALLIC, pixel) < data.metallic) {
      ray_local = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);

      const BSDFRayContext context = bsdf_sample_context(data_local, sampled_microfacet, ray_local, false);

      info.antagonist_weight   = 0.0f;
      info.weight              = bsdf_conductor(data_local, context, BSDF_SAMPLING_MICROFACET, 1.0f);
      info.sampled_technique   = BSDF_CONDUCTOR;
      info.is_microfacet_based = true;
    }
    else {
      const vec3 microfacet_ray           = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);
      const BSDFRayContext microfacet_ctx = bsdf_sample_context(data_local, sampled_microfacet, microfacet_ray, false);
      const RGBF microfacet_eval          = bsdf_glossy(data_local, microfacet_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

      const vec3 diffuse_ray           = bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_DIFFUSE, pixel));
      const vec3 diffuse_microfacet    = normalize_vector(add_vector(data.V, diffuse_ray));
      const BSDFRayContext diffuse_ctx = bsdf_sample_context(data_local, diffuse_microfacet, diffuse_ray, false);
      const RGBF diffuse_eval          = bsdf_glossy(data_local, diffuse_ctx, BSDF_SAMPLING_DIFFUSE, 1.0f);

      const float microfacet_weight = luminance(microfacet_eval);
      const float diffuse_weight    = luminance(diffuse_eval);

      const float sum_weights = microfacet_weight + diffuse_weight;

      const float microfacet_probability = microfacet_weight / sum_weights;
      const float diffuse_probability    = diffuse_weight / sum_weights;

      RGBF final_weight;
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_GLOSSY, pixel) < microfacet_probability) {
        ray_local                = microfacet_ray;
        final_weight             = scale_color(microfacet_eval, 1.0f / microfacet_probability);
        info.is_microfacet_based = true;
        info.antagonist_weight   = diffuse_weight;
        choice_probability       = microfacet_probability;
      }
      else {
        ray_local                = diffuse_ray;
        final_weight             = scale_color(diffuse_eval, 1.0f / diffuse_probability);
        info.is_microfacet_based = false;
        info.antagonist_weight   = microfacet_weight;
        choice_probability       = diffuse_probability;
      }

      info.sampled_technique = BSDF_GLOSSY;
      info.weight            = final_weight;
    }
  }
  else {
    const vec3 reflection_vector        = reflect_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet);
    const BSDFRayContext reflection_ctx = bsdf_sample_context(data_local, sampled_microfacet, reflection_vector, false);
    const RGBF reflection_eval          = bsdf_dielectric(data_local, reflection_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

    sampled_microfacet_refraction = bsdf_microfacet_refraction_sample(data_local, pixel);
    const vec3 refraction_vector =
      refract_vector(scale_vector(data_local.V, -1.0f), sampled_microfacet_refraction, data.ior_in / data.ior_out);
    const BSDFRayContext refraction_ctx = bsdf_sample_context(data_local, sampled_microfacet_refraction, refraction_vector, true);
    const RGBF refraction_eval          = bsdf_dielectric(data_local, refraction_ctx, BSDF_SAMPLING_MICROFACET_REFRACTION, 1.0f);

    const float reflection_weight = luminance(reflection_eval);
    const float refraction_weight = luminance(refraction_eval);

    const float sum_weights = reflection_weight + refraction_weight;

    const float reflection_probability = reflection_weight / sum_weights;
    const float refraction_probability = refraction_weight / sum_weights;

    RGBF final_weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_DIELECTRIC, pixel) < reflection_probability) {
      ray_local              = reflection_vector;
      final_weight           = scale_color(reflection_eval, 1.0f / reflection_probability);
      info.antagonist_weight = refraction_weight;
      choice_probability     = reflection_probability;
    }
    else {
      ray_local                = refraction_vector;
      final_weight             = scale_color(refraction_eval, 1.0f / refraction_probability);
      info.antagonist_weight   = reflection_weight;
      info.is_transparent_pass = true;
      choice_probability       = refraction_probability;
    }

    info.weight              = final_weight;
    info.sampled_technique   = BSDF_DIELECTRIC;
    info.is_microfacet_based = true;
  }

  float sample_pdf;
  if (info.is_transparent_pass) {
    sample_pdf = bsdf_microfacet_refraction_pdf(data_local, sampled_microfacet_refraction.z, data_local.V.z);
  }
  else if (info.is_microfacet_based) {
    sample_pdf = bsdf_microfacet_pdf(data_local, sampled_microfacet.z, data_local.V.z);
  }
  else {
    sample_pdf = bsdf_diffuse_pdf(data_local, ray_local.z);
  }

  marginal = sample_pdf * choice_probability;

  return normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
}

__device__ float bsdf_sample_marginal(const GBufferData data, const vec3 ray, const BSDFSampleInfo info) {
  if (data.flags & G_BUFFER_VOLUME_HIT) {
    const float cos_angle = -dot_product(ray, data.V);

    return (VOLUME_HIT_TYPE(data.hit_id) != VOLUME_TYPE_OCEAN) ? jendersie_eon_phase_function(cos_angle, data.roughness)
                                                               : ocean_phase(cos_angle);
  }

  // Transformation to +Z-Up
  const Quaternion rotation_to_z = get_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(data.V, rotation_to_z);
  const vec3 ray_local           = rotate_vector_by_quaternion(ray, rotation_to_z);

  if (ray_local.z < 0.0f && !info.is_transparent_pass)
    return 0.0f;

  // G Buffer Data (+Z-Up)
  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  float marginal;
  vec3 H = get_vector(0.0f, 0.0f, 1.0f);
  switch (info.sampled_technique) {
    default:
    case BSDF_CONDUCTOR:
      H        = normalize_vector(add_vector(data_local.V, ray_local));
      marginal = 1.0f;
      break;
    case BSDF_GLOSSY: {
      H                        = normalize_vector(add_vector(data_local.V, ray_local));
      const BSDFRayContext ctx = bsdf_sample_context(data_local, H, ray_local, false);

      const BSDFSamplingHint hint = (info.is_microfacet_based) ? BSDF_SAMPLING_MICROFACET : BSDF_SAMPLING_DIFFUSE;

      const RGBF eval    = bsdf_glossy(data_local, ctx, hint, 1.0f);
      const float weight = luminance(eval);

      marginal = ((weight + info.antagonist_weight) > 0.0f) ? weight / (weight + info.antagonist_weight) : 0.0f;
    } break;
    case BSDF_DIELECTRIC: {
      H = (info.is_transparent_pass) ? bsdf_refraction_normal_from_pair(ray_local, data_local.V, data_local.ior_out, data_local.ior_in)
                                     : normalize_vector(add_vector(data_local.V, ray_local));

      const BSDFRayContext ctx = bsdf_sample_context(data_local, H, ray_local, info.is_transparent_pass);

      const BSDFSamplingHint hint = (info.is_transparent_pass) ? BSDF_SAMPLING_MICROFACET_REFRACTION : BSDF_SAMPLING_MICROFACET;

      const RGBF eval    = bsdf_dielectric(data_local, ctx, hint, 1.0f);
      const float weight = luminance(eval);

      marginal = ((weight + info.antagonist_weight) > 0.0f) ? weight / (weight + info.antagonist_weight) : 0.0f;
    } break;
  }

  float sample_pdf;
  if (info.is_microfacet_based) {
    sample_pdf = bsdf_microfacet_pdf(data_local, H.z, data_local.V.z);
  }
  else {
    sample_pdf = bsdf_diffuse_pdf(data_local, ray_local.z);
  }

  return marginal * sample_pdf;
}

#endif /* CU_BSDF_H */
