#ifndef CU_BSDF_H
#define CU_BSDF_H

#include "bsdf_utils.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

__device__ BSDFRayContext bsdf_evaluate_analyze(const GBufferData data, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = __saturatef(dot_product(data.normal, data.V));

  context.is_refraction = (context.NdotL < 0.0f);
  context.NdotL         = (context.is_refraction) ? -context.NdotL : context.NdotL;

  const float ior_in  = fminf(2.999f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(2.999f, fmaxf(1.0f, data.ior_out));

  context.refraction_index = ior_in / ior_out;

  vec3 refraction_vector;
  bool total_reflection;
  if (context.is_refraction) {
    total_reflection  = false;  // TODO: Correctly check if this refraction is possible.
    context.H         = bsdf_normal_from_pair(L, data.V, context.refraction_index);
    refraction_vector = L;
  }
  else {
    context.H         = bsdf_normal_from_pair(L, data.V, 1.0f);
    refraction_vector = refract_vector(data.V, context.H, context.refraction_index, total_reflection);
  }

  context.NdotH = dot_product(data.normal, context.H);

  if (context.NdotH < 0.0f) {
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
#ifdef PHASE_KERNEL
  return scale_color(volume_phase_evaluate(data, VOLUME_HIT_TYPE(data.instance_id), L), one_over_sampling_pdf);
#else
  const BSDFRayContext context = bsdf_evaluate_analyze(data, L);

  is_refraction = context.is_refraction;

  return bsdf_evaluate_core(data, context, sampling_hint, one_over_sampling_pdf);
#endif
}

__device__ BSDFRayContext bsdf_sample_context(const GBufferData data, const vec3 H, const vec3 L, const bool is_refraction) {
  BSDFRayContext context;

  context.NdotL = dot_product(data.normal, L);
  context.NdotV = __saturatef(dot_product(data.normal, data.V));

  context.is_refraction = is_refraction;

  context.NdotL = (is_refraction) ? -context.NdotL : context.NdotL;

  const float ior_in  = fminf(2.999f, fmaxf(1.0f, data.ior_in));
  const float ior_out = fminf(2.999f, fmaxf(1.0f, data.ior_out));

  context.refraction_index = ior_in / ior_out;

  context.H = H;

  bool total_reflection = false;
  const vec3 refraction_vector =
    (context.is_refraction) ? L : refract_vector(data.V, context.H, context.refraction_index, total_reflection);

  context.NdotH = dot_product(data.normal, context.H);

  if (context.NdotH < 0.0f) {
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

__device__ vec3 bsdf_sample(const GBufferData data, const ushort2 pixel, BSDFSampleInfo& info) {
#ifdef PHASE_KERNEL
  const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_VOLUME_CHOISE, pixel);
  const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_VOLUME, pixel);

  const vec3 ray = scale_vector(data.V, -1.0f);

  const vec3 scatter_ray = (VOLUME_HIT_TYPE(data.instance_id) != VOLUME_TYPE_OCEAN)
                             ? jendersie_eon_phase_sample(ray, data.roughness, random_dir, random_choice)
                             : ocean_phase_sampling(ray, random_dir, random_choice);

  info.weight = get_color(1.0f, 1.0f, 1.0f);

  return scatter_ray;
#else

  if (data.albedo.a < 1.0f) {
    const float transparency_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_OPACITY, pixel);

    if (transparency_random > data.albedo.a) {
      info.weight = (data.flags & G_BUFFER_FLAG_COLORED_TRANSPARENCY) ? opaque_color(data.albedo) : get_color(1.0f, 1.0f, 1.0f);
      info.is_microfacet_based = true;
      info.is_transparent_pass = true;

      return scale_vector(data.V, -1.0f);
    }
  }

  // Transformation to +Z-Up
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = quaternion_apply(rotation_to_z, data.V);

  // G Buffer Data (+Z-Up)
  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  info.is_transparent_pass = false;
  info.is_microfacet_based = false;

  vec3 ray_local;

  const uint32_t base_substrate = data_local.flags & G_BUFFER_FLAG_BASE_SUBSTRATE_MASK;

  if (ior_compress(data_local.ior_in) == ior_compress(data_local.ior_out) && (base_substrate == G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT)) {
    // Fast path for transparent surfaces without refraction/reflection
    ray_local                = scale_vector(data_local.V, -1.0f);
    info.weight              = opaque_color(data.albedo);
    info.is_transparent_pass = true;
    info.is_microfacet_based = true;
  }
  else {
    const bool include_diffuse =
      (base_substrate == G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE) && ((data_local.flags & G_BUFFER_FLAG_METALLIC) == 0);
    const bool include_refraction = (base_substrate == G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT);

    // Microfacet evaluation is not numerically stable for very low roughness. We clamp the evaluation here.
    data_local.roughness = fmaxf(data_local.roughness, BSDF_ROUGHNESS_CLAMP);

    float sum_weights  = 0.0f;
    RGBF selected_eval = get_color(0.0f, 0.0f, 0.0f);

    // Microfacet based sample
    if (true) {
      const vec3 microfacet    = bsdf_microfacet_sample(data_local, pixel);
      const vec3 ray           = reflect_vector(data_local.V, microfacet);
      const BSDFRayContext ctx = bsdf_sample_context(data_local, microfacet, ray, false);
      const RGBF eval          = bsdf_evaluate_core(data_local, ctx, BSDF_SAMPLING_MICROFACET);
      const float pdf          = bsdf_microfacet_pdf(data_local, ctx.NdotH, ctx.NdotV);
      const float diffuse_pdf  = (include_diffuse) ? bsdf_diffuse_pdf(data_local, ctx.NdotL) : 0.0f;
      const float refraction_pdf =
        (include_refraction)
          ? bsdf_microfacet_refraction_pdf(data_local, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index)
          : 0.0f;

      const float sum_pdf    = pdf + diffuse_pdf + refraction_pdf;
      const float mis_weight = (sum_pdf > 0.0f) ? pdf / sum_pdf : 0.0f;

      const float weight = color_importance(eval) * mis_weight;

      UTILS_CHECK_NANS(pixel, weight);

      ray_local                = ray;
      sum_weights              = weight;
      selected_eval            = eval;
      info.is_transparent_pass = false;
      info.is_microfacet_based = true;
    }

    // Diffuse based sample
    if (include_diffuse) {
      const vec3 ray             = bsdf_diffuse_sample(quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_DIFFUSE, pixel));
      const vec3 microfacet      = normalize_vector(add_vector(data_local.V, ray));
      const BSDFRayContext ctx   = bsdf_sample_context(data_local, microfacet, ray, false);
      const RGBF eval            = bsdf_evaluate_core(data_local, ctx, BSDF_SAMPLING_DIFFUSE);
      const float pdf            = bsdf_diffuse_pdf(data_local, ctx.NdotL);
      const float microfacet_pdf = bsdf_microfacet_pdf(data_local, ctx.NdotH, ctx.NdotV);
      const float refraction_pdf =
        (include_refraction)
          ? bsdf_microfacet_refraction_pdf(data_local, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index)
          : 0.0f;

      const float sum_pdf    = pdf + microfacet_pdf + refraction_pdf;
      const float mis_weight = (sum_pdf > 0.0f) ? pdf / sum_pdf : 0.0f;

      const float weight = color_importance(eval) * mis_weight;

      UTILS_CHECK_NANS(pixel, weight);

      sum_weights += weight;
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_RIS_DIFFUSE, pixel) * sum_weights < weight) {
        ray_local                = ray;
        selected_eval            = eval;
        info.is_transparent_pass = false;
        info.is_microfacet_based = false;
      }
    }

    // Microfacet refraction based sample
    if (include_refraction) {
      bool total_reflection;
      const vec3 microfacet    = bsdf_microfacet_refraction_sample(data_local, pixel);
      const vec3 ray           = refract_vector(data_local.V, microfacet, data.ior_in / data.ior_out, total_reflection);
      const BSDFRayContext ctx = bsdf_sample_context(data_local, microfacet, ray, !total_reflection);
      const RGBF eval          = bsdf_evaluate_core(data_local, ctx, BSDF_SAMPLING_MICROFACET_REFRACTION);

      float mis_weight = 1.0f;

      // If it is a reflection, then the direction could have been sampled from a microfacet reflection,
      // hence we need to compute its MIS weight.
      if (total_reflection) {
        const float pdf =
          bsdf_microfacet_refraction_pdf(data_local, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index);
        const float reflection_pdf = bsdf_microfacet_pdf(data_local, ctx.NdotH, ctx.NdotV);
        const float diffuse_pdf    = (include_diffuse) ? bsdf_diffuse_pdf(data_local, ctx.NdotL) : 0.0f;

        const float sum_pdf = pdf + reflection_pdf + diffuse_pdf;
        mis_weight          = (sum_pdf > 0.0f) ? pdf / sum_pdf : 0.0f;
      }

      const float weight = color_importance(eval) * mis_weight;

      UTILS_CHECK_NANS(pixel, weight);

      sum_weights += weight;
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_RIS_REFRACTION, pixel) * sum_weights < weight) {
        ray_local                = ray;
        selected_eval            = eval;
        info.is_transparent_pass = !total_reflection;
        info.is_microfacet_based = true;
      }
    }

    // For RIS we need to evaluate f / |f| here. This is unstable for low roughness and microfacet BRDFs.
    // Hence we use a little trick, f / p can be evaluated in a stable manner when p is the microfacet PDF,
    // and thus we evaluate f / |f| = (f / p) / |f / p|.
    info.weight =
      (sum_weights > 0.0f) ? scale_color(selected_eval, sum_weights / color_importance(selected_eval)) : get_color(0.0f, 0.0f, 0.0f);
  }

  UTILS_CHECK_NANS(pixel, info.weight);

  return normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), ray_local));
#endif
}

__device__ vec3 bsdf_sample_microfacet_reflection(const GBufferData data, const ushort2 pixel, const QuasiRandomTarget random_target_ray) {
  const vec3 sampled_microfacet = bsdf_microfacet_sample(data, pixel, random_target_ray);

  return reflect_vector(data.V, sampled_microfacet);
}

__device__ vec3 bsdf_sample_microfacet_refraction(const GBufferData data, const ushort2 pixel, const QuasiRandomTarget random_target_ray) {
  const vec3 sampled_microfacet = bsdf_microfacet_refraction_sample(data, pixel, random_target_ray);

  bool total_reflection;
  return refract_vector(data.V, sampled_microfacet, data.ior_in / data.ior_out, total_reflection);
}

__device__ vec3 bsdf_sample_diffuse(const GBufferData data, const ushort2 pixel, const QuasiRandomTarget random_target_ray) {
  return bsdf_diffuse_sample(quasirandom_sequence_2D(random_target_ray, pixel));
}

__device__ void bsdf_sample_for_light_probabilities(
  const GBufferData data, float& reflection_prob, float& refraction_prob, float& diffuse_prob) {
  // TODO: Consider creating a context and sampling also proportional to albedo etc.
  // TODO: There is this issue where I importance sample the lights too well but end up picking occluded
  // lights which will give me terrible convergence.
  float microfacet_reflection_weight;
  float microfacet_refraction_weight;
  float diffuse_weight;

  switch (data.flags & G_BUFFER_FLAG_BASE_SUBSTRATE_MASK) {
    case G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE:
      microfacet_reflection_weight = 1.0f;
      microfacet_refraction_weight = 0.0f;
      diffuse_weight               = (data.flags & G_BUFFER_FLAG_METALLIC) ? 0.0f : 1.0f;
      break;
    case G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT:
      microfacet_reflection_weight = 1.0f;
      microfacet_refraction_weight = 1.0f;
      diffuse_weight               = 0.0f;
      break;
  }

  const float sum_weights = microfacet_reflection_weight + microfacet_refraction_weight + diffuse_weight;

  reflection_prob = microfacet_reflection_weight / sum_weights;
  refraction_prob = microfacet_refraction_weight / sum_weights;
  diffuse_prob    = diffuse_weight / sum_weights;
}

__device__ vec3 bsdf_sample_for_light(
  const GBufferData data, const ushort2 pixel, const QuasiRandomTarget random_target, bool& is_refraction, bool& is_valid) {
#ifdef PHASE_KERNEL
  const float2 random_dir   = quasirandom_sequence_2D(random_target, pixel);
  const float random_method = quasirandom_sequence_1D(random_target + 1, pixel);

  const vec3 ray = scale_vector(data.V, -1.0f);

  const vec3 scatter_ray = (VOLUME_HIT_TYPE(data.instance_id) != VOLUME_TYPE_OCEAN)
                             ? jendersie_eon_phase_sample(ray, data.roughness, random_dir, random_method)
                             : ocean_phase_sampling(ray, random_dir, random_method);

  is_refraction = true;
  is_valid      = true;

  return scatter_ray;
#else   // PHASE_KERNEL
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = quaternion_apply(rotation_to_z, data.V);

  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  float reflection_probability, refraction_probability, diffuse_probability;
  bsdf_sample_for_light_probabilities(data, reflection_probability, refraction_probability, diffuse_probability);

  const float random_method = quasirandom_sequence_1D(random_target + 1, pixel);

  vec3 ray_local;
  if (random_method < reflection_probability) {
    ray_local     = bsdf_sample_microfacet_reflection(data_local, pixel, random_target);
    is_refraction = false;
  }
  else if (random_method < reflection_probability + refraction_probability) {
    ray_local     = bsdf_sample_microfacet_refraction(data_local, pixel, random_target);
    is_refraction = true;
  }
  else {
    ray_local     = bsdf_sample_diffuse(data_local, pixel, random_target);
    is_refraction = false;
  }

  is_valid = (is_refraction) ? ray_local.z < 0.0f : ray_local.z > 0.0f;

  return normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), ray_local));
#endif  // !PHASE_KERNEL
}

__device__ float bsdf_sample_for_light_pdf(const GBufferData data, const vec3 L) {
#ifdef PHASE_KERNEL
  const float cos_angle = -dot_product(data.V, L);

  const VolumeType volume_hit_type = VOLUME_HIT_TYPE(data.instance_id);

  float pdf;
  if (volume_hit_type == VOLUME_TYPE_OCEAN) {
    pdf = ocean_phase(cos_angle);
  }
  else {
    const float diameter            = (volume_hit_type == VOLUME_TYPE_FOG) ? device.fog.droplet_diameter : device.particles.phase_diameter;
    const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);
    pdf                             = jendersie_eon_phase_function(cos_angle, params);
  }

  return pdf;
#else   // PHASE_KERNEL
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
#endif  // !PHASE_KERNEL
}

#endif /* CU_BSDF_H */
