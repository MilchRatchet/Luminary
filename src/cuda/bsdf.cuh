#ifndef CU_BSDF_H
#define CU_BSDF_H

#if defined(SHADING_KERNEL)

#include "bsdf_utils.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

__device__ BSDFRayContext bsdf_evaluate_analyze(const GBufferData data, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = fabsf(dot_product(data.normal, data.V));

  context.is_refraction = false;

  if (context.NdotL < 0.0f) {
    context.NdotL *= -1.0f;
    context.is_refraction = true;
  }

  const float ior_in  = fminf(2.999f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(2.999f, fmaxf(1.0f, data.ior_out));

  context.refraction_index = ior_in / ior_out;

  vec3 refraction_vector;
  bool total_reflection;
  if (context.is_refraction) {
    context.H         = bsdf_refraction_normal_from_pair(L, data.V, context.refraction_index);
    refraction_vector = L;
  }
  else {
    context.H         = normalize_vector(add_vector(data.V, L));
    refraction_vector = refract_vector(data.V, context.H, context.refraction_index, total_reflection);
  }

  context.NdotH = dot_product(data.normal, context.H);

  if (dot_product(context.H, data.normal) < 0.0f) {
    context.H     = scale_vector(context.H, -1.0f);
    context.NdotH = -context.NdotH;
  }

  context.HdotV = fabsf(dot_product(context.H, data.V));
  context.HdotL = fabsf(dot_product(context.H, L));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), context.HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), context.HdotV);

  context.fresnel_dielectric = (total_reflection) ? 1.0f : bsdf_fresnel(context.H, data.V, refraction_vector, ior_in, ior_out);

  return context;
}

__device__ RGBF bsdf_evaluate_core(
  const GBufferData data, const BSDFRayContext context, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf = 1.0f) {
  return bsdf_multiscattering_evaluate(data, context, sampling_hint, one_over_sampling_pdf);
}

__device__ RGBF bsdf_evaluate(
  const GBufferData data, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_refraction,
  const float one_over_sampling_pdf = 1.0f) {
#ifdef VOLUME_KERNEL
  return scale_color(volume_phase_evaluate(data, VOLUME_HIT_TYPE(data.hit_id), L), one_over_sampling_pdf);
#else
  const BSDFRayContext context = bsdf_evaluate_analyze(data, L);

  is_refraction = context.is_refraction;

  return bsdf_evaluate_core(data, context, sampling_hint, one_over_sampling_pdf);
#endif
}

__device__ BSDFRayContext bsdf_sample_context(const GBufferData data, const vec3 H, const vec3 L, const bool is_refraction) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = fabsf(dot_product(data.normal, data.V));

  context.is_refraction = is_refraction;

  context.NdotL = fabsf(context.NdotL);

  const float ior_in  = fminf(2.999f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(2.999f, fmaxf(1.0f, data.ior_out));

  context.refraction_index = ior_in / ior_out;

  context.H = H;

  bool total_reflection = false;
  const vec3 refraction_vector =
    (context.is_refraction) ? L : refract_vector(data.V, context.H, context.refraction_index, total_reflection);

  context.NdotH = fabsf(dot_product(data.normal, context.H));
  context.HdotV = fabsf(dot_product(context.H, data.V));
  context.HdotL = fabsf(dot_product(context.H, L));

  context.f0_conductor      = opaque_color(data.albedo);
  context.fresnel_conductor = bsdf_fresnel_schlick(context.f0_conductor, bsdf_shadowed_F90(context.f0_conductor), context.HdotV);

  context.f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  context.fresnel_glossy = bsdf_fresnel_schlick(context.f0_glossy, bsdf_shadowed_F90(context.f0_glossy), context.HdotV);

  context.fresnel_dielectric = (total_reflection) ? 1.0f : bsdf_fresnel(context.H, data.V, refraction_vector, ior_in, ior_out);

  return context;
}

__device__ vec3 bsdf_sample(const GBufferData data, const ushort2 pixel, BSDFSampleInfo& info) {
#ifdef VOLUME_KERNEL
  const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE, pixel);
  const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BOUNCE_DIR, pixel);

  const vec3 ray = scale_vector(data.V, -1.0f);

  const vec3 scatter_ray = (VOLUME_HIT_TYPE(data.hit_id) != VOLUME_TYPE_OCEAN)
                             ? jendersie_eon_phase_sample(ray, data.roughness, random_dir, random_choice)
                             : ocean_phase_sampling(ray, random_dir, random_choice);

  info.weight = get_color(1.0f, 1.0f, 1.0f);

  return scatter_ray;
#else
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
  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_ALPHA, pixel) < data.albedo.a) {
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_METALLIC, pixel) < data.metallic) {
      ray_local = reflect_vector(data_local.V, sampled_microfacet);

      const BSDFRayContext context = bsdf_sample_context(data_local, sampled_microfacet, ray_local, false);

      info.weight              = bsdf_conductor(data_local, context, BSDF_SAMPLING_MICROFACET, 1.0f);
      info.is_microfacet_based = true;
    }
    else {
      // Glossy BSDF model
      // We sample two directions, one from the microfacet, one from a cosine distribution
      // We then use RIS with balance heuristic MIS weights to choose a direction.

      // Microfacet evaluation is not numerically stable for very low roughness. We clamp the evaluation here.
      // Note that the microfacet was sampled with the actual roughness. This means we still get perfect mirrors.
      data_local.roughness = fmaxf(data_local.roughness, 0.05f);

      // Microfacet based sample
      const vec3 microfacet_ray           = reflect_vector(data_local.V, sampled_microfacet);
      const BSDFRayContext microfacet_ctx = bsdf_sample_context(data_local, sampled_microfacet, microfacet_ray, false);
      const RGBF microfacet_eval          = bsdf_glossy(data_local, microfacet_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);
      const float microfacet_pdf          = bsdf_microfacet_pdf(data_local, microfacet_ctx.NdotH, microfacet_ctx.NdotV);
      const float microfacet_diffuse_pdf  = bsdf_diffuse_pdf(data_local, microfacet_ctx.NdotL);
      const float microfacet_mis          = microfacet_pdf / (microfacet_pdf + microfacet_diffuse_pdf);

      // Diffuse based sample
      const vec3 diffuse_ray             = bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_DIFFUSE, pixel));
      const vec3 diffuse_microfacet      = normalize_vector(add_vector(data_local.V, diffuse_ray));
      const BSDFRayContext diffuse_ctx   = bsdf_sample_context(data_local, diffuse_microfacet, diffuse_ray, false);
      const RGBF diffuse_eval            = bsdf_glossy(data_local, diffuse_ctx, BSDF_SAMPLING_DIFFUSE, 1.0f);
      const float diffuse_pdf            = bsdf_diffuse_pdf(data_local, diffuse_ctx.NdotL);
      const float diffuse_microfacet_pdf = bsdf_microfacet_pdf(data_local, diffuse_ctx.NdotH, diffuse_ctx.NdotV);
      const float diffuse_mis            = diffuse_pdf / (diffuse_pdf + diffuse_microfacet_pdf);

      const float microfacet_weight = luminance(microfacet_eval) * microfacet_mis;
      const float diffuse_weight    = luminance(diffuse_eval) * diffuse_mis;

      const float sum_weights = microfacet_weight + diffuse_weight;

      const float microfacet_probability = microfacet_weight / sum_weights;

      // For RIS we need to evaluate f / |f| here. This is unstable for low roughness and microfacet BRDFs.
      // Hence we use a little trick, f / p can be evaluated in a stable manner when p is the microfacet PDF,
      // and thus we evaluate f / |f| = (f / p) / |f / p|.
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_GLOSSY, pixel) < microfacet_probability) {
        ray_local                = microfacet_ray;
        info.weight              = scale_color(microfacet_eval, sum_weights / luminance(microfacet_eval));
        info.is_microfacet_based = true;
      }
      else {
        ray_local                = diffuse_ray;
        info.weight              = scale_color(diffuse_eval, sum_weights / luminance(diffuse_eval));
        info.is_microfacet_based = false;
      }
    }
  }
  else if (ior_compress(data_local.ior_in) == ior_compress(data_local.ior_out)) {
    // Fast path for transparent surfaces without refraction/reflection
    ray_local                = scale_vector(data_local.V, -1.0f);
    info.weight              = (data_local.flags & G_BUFFER_COLORED_DIELECTRIC) ? opaque_color(data.albedo) : get_color(1.0f, 1.0f, 1.0f);
    info.is_transparent_pass = true;
    info.is_microfacet_based = true;
  }
  else {
    const vec3 reflection_vector        = reflect_vector(data_local.V, sampled_microfacet);
    const BSDFRayContext reflection_ctx = bsdf_sample_context(data_local, sampled_microfacet, reflection_vector, false);
    const RGBF reflection_eval          = bsdf_dielectric(data_local, reflection_ctx, BSDF_SAMPLING_MICROFACET, 1.0f);

    bool total_reflection;

    sampled_microfacet_refraction = bsdf_microfacet_refraction_sample(data_local, pixel);
    const vec3 refraction_vector =
      refract_vector(data_local.V, sampled_microfacet_refraction, data.ior_in / data.ior_out, total_reflection);
    const BSDFRayContext refraction_ctx = bsdf_sample_context(data_local, sampled_microfacet_refraction, refraction_vector, true);
    const RGBF refraction_eval          = bsdf_dielectric(data_local, refraction_ctx, BSDF_SAMPLING_MICROFACET_REFRACTION, 1.0f);

    const float reflection_weight = luminance(reflection_eval);
    const float refraction_weight = (total_reflection) ? 0.0f : luminance(refraction_eval);

    const float sum_weights = reflection_weight + refraction_weight;

    const float reflection_probability = reflection_weight / sum_weights;
    const float refraction_probability = refraction_weight / sum_weights;

    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_DIELECTRIC, pixel) < reflection_probability) {
      ray_local   = reflection_vector;
      info.weight = scale_color(reflection_eval, 1.0f / reflection_probability);
    }
    else {
      ray_local                = refraction_vector;
      info.weight              = scale_color(refraction_eval, 1.0f / refraction_probability);
      info.is_transparent_pass = true;
    }

    info.is_microfacet_based = true;
  }

  return normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
#endif
}

__device__ vec3 bsdf_sample_microfacet_reflection(const GBufferData data, const ushort2 pixel) {
  const vec3 sampled_microfacet = bsdf_microfacet_sample(data, pixel, QUASI_RANDOM_TARGET_LIGHT_BSDF_RAY);

  return reflect_vector(data.V, sampled_microfacet);
}

__device__ vec3 bsdf_sample_microfacet_refraction(const GBufferData data, const ushort2 pixel) {
  const vec3 sampled_microfacet = bsdf_microfacet_refraction_sample(data, pixel, QUASI_RANDOM_TARGET_LIGHT_BSDF_RAY);

  bool total_reflection;
  return refract_vector(data.V, sampled_microfacet, data.ior_in / data.ior_out, total_reflection);
}

__device__ vec3 bsdf_sample_diffuse(const GBufferData data, const ushort2 pixel) {
  return bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LIGHT_BSDF_RAY, pixel));
}

__device__ void bsdf_sample_for_light_probabilities(
  const GBufferData data, float& reflection_prob, float& refraction_prob, float& diffuse_prob) {
  const float microfacet_reflection_weight = 1.0f;
  const float microfacet_refraction_weight = 1.0f - data.albedo.a;
  const float diffuse_weight               = (1.0f - data.metallic) * data.albedo.a;

  const float sum_weights = microfacet_reflection_weight + microfacet_refraction_weight + diffuse_weight;

  reflection_prob = microfacet_reflection_weight / sum_weights;
  refraction_prob = microfacet_refraction_weight / sum_weights;
  diffuse_prob    = diffuse_weight / sum_weights;
}

__device__ vec3 bsdf_sample_for_light(const GBufferData data, const ushort2 pixel, bool& is_refraction, bool& is_valid) {
#ifdef VOLUME_KERNEL
  const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_BSDF_METHOD, pixel);
  const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LIGHT_BSDF_RAY, pixel);

  const vec3 ray = scale_vector(data.V, -1.0f);

  const vec3 scatter_ray = (VOLUME_HIT_TYPE(data.hit_id) != VOLUME_TYPE_OCEAN)
                             ? jendersie_eon_phase_sample(ray, data.roughness, random_dir, random_choice)
                             : ocean_phase_sampling(ray, random_dir, random_choice);

  is_refraction = true;
  is_valid      = true;

  return scatter_ray;
#else   // VOLUME_KERNEL
  // TODO: It is important that pass through rays are not allowed! Otherwise we run into double counting issues.

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(data.V, rotation_to_z);

  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  float reflection_probability, refraction_probability, diffuse_probability;
  bsdf_sample_for_light_probabilities(data, reflection_probability, refraction_probability, diffuse_probability);

  const float random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_BSDF_METHOD, pixel);

  vec3 ray_local;
  if (random < reflection_probability) {
    ray_local     = bsdf_sample_microfacet_reflection(data_local, pixel);
    is_refraction = false;
  }
  else if (random < reflection_probability + refraction_probability) {
    ray_local     = bsdf_sample_microfacet_refraction(data_local, pixel);
    is_refraction = true;
  }
  else {
    ray_local     = bsdf_sample_diffuse(data_local, pixel);
    is_refraction = false;
  }

  is_valid = (is_refraction) ? ray_local.z < 0.0f : ray_local.z > 0.0f;

  return normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
#endif  // VOLUME_KERNEL
}

__device__ float bsdf_sample_for_light_pdf(const GBufferData data, const vec3 L) {
#ifdef VOLUME_KERNEL
  const float cos_angle = -dot_product(data.V, L);

  const VolumeType volume_hit_type = VOLUME_HIT_TYPE(data.hit_id);

  float pdf;
  if (volume_hit_type == VOLUME_TYPE_OCEAN) {
    pdf = ocean_phase(cos_angle);
  }
  else {
    const float diameter = (volume_hit_type == VOLUME_TYPE_FOG) ? device.scene.fog.droplet_diameter : device.scene.particles.phase_diameter;
    const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);
    pdf                             = jendersie_eon_phase_function(cos_angle, params);
  }

  return pdf;
#else   // VOLUME_KERNEL
  float reflection_probability, refraction_probability, diffuse_probability;
  bsdf_sample_for_light_probabilities(data, reflection_probability, refraction_probability, diffuse_probability);

  const BSDFRayContext context = bsdf_evaluate_analyze(data, L);

  if (context.is_refraction) {
    const float microfacet_refraction_pdf = bsdf_microfacet_refraction_pdf(
      data, context.NdotH, context.NdotV, context.NdotL, context.HdotV, context.HdotL, context.refraction_index);

    return refraction_probability * microfacet_refraction_pdf;
  }
  else {
    const float microfacet_reflection_pdf = bsdf_microfacet_pdf(data, context.NdotH, context.NdotV);
    const float diffuse_pdf               = bsdf_diffuse_pdf(data, context.NdotL);

    return reflection_probability * microfacet_reflection_pdf + diffuse_probability * diffuse_pdf;
  }
#endif  // VOLUME_KERNEL
}

#endif /* SHADING_KERNEL */

#endif /* CU_BSDF_H */
