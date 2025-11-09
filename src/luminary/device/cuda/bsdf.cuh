#ifndef CU_BSDF_H
#define CU_BSDF_H

#include "bsdf_utils.cuh"
#include "material.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

LUMINARY_FUNCTION BSDFRayContext bsdf_evaluate_analyze(const MaterialContextGeometry mat_ctx, const vec3 L) {
  BSDFRayContext context;

  context.NdotL = dot_product(mat_ctx.normal, L);
  context.NdotV = __saturatef(dot_product(mat_ctx.normal, mat_ctx.V));

  context.is_refraction = (context.NdotL < 0.0f);
  context.NdotL         = (context.is_refraction) ? -context.NdotL : context.NdotL;

  const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx);

  vec3 refraction_vector;
  bool total_reflection;
  vec3 H;
  if (context.is_refraction) {
    total_reflection  = false;  // TODO: Correctly check if this refraction is possible.
    H                 = bsdf_normal_from_pair(L, mat_ctx.V, ior);
    refraction_vector = L;
  }
  else {
    H                 = bsdf_normal_from_pair(L, mat_ctx.V, 1.0f);
    refraction_vector = refract_vector(mat_ctx.V, H, ior, total_reflection);
  }

  context.NdotH = dot_product(mat_ctx.normal, H);

  if (context.NdotH < 0.0f) {
    H             = scale_vector(H, -1.0f);
    context.NdotH = -context.NdotH;
  }

  context.HdotV = fabsf(dot_product(H, mat_ctx.V));
  context.HdotL = fabsf(dot_product(H, L));

  context.fresnel_dielectric = (total_reflection) ? 1.0f : bsdf_fresnel(H, mat_ctx.V, refraction_vector, ior);

  return context;
}

LUMINARY_FUNCTION RGBF bsdf_evaluate_core(
  const MaterialContextGeometry ctx, const BSDFRayContext context, const BSDFSamplingHint sampling_hint, const vec3 L,
  const vec3 face_normal, const float one_over_sampling_pdf = 1.0f) {
  const float FN_dot_L  = dot_product(face_normal, L);
  const float face_flip = context.is_refraction ? -1.0f : 1.0f;

  // Invalidate directions that are valid for the shading normal but not for the face normal
  if (FN_dot_L * face_flip < eps) {
    return splat_color(0.0f);
  }

  return bsdf_multiscattering_evaluate(ctx, context, sampling_hint, one_over_sampling_pdf);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION RGBF bsdf_evaluate(
  const MaterialContext<TYPE> ctx, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_refraction,
  const float one_over_sampling_pdf = 1.0f);

template <>
LUMINARY_FUNCTION RGBF bsdf_evaluate(
  const MaterialContextGeometry ctx, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_refraction,
  const float one_over_sampling_pdf) {
  const BSDFRayContext context = bsdf_evaluate_analyze(ctx, L);

  is_refraction = context.is_refraction;

  const vec3 face_normal = material_get_normal<MATERIAL_GEOMETRY_PARAM_FACE_NORMAL>(ctx);

  return bsdf_evaluate_core(ctx, context, sampling_hint, L, face_normal, one_over_sampling_pdf);
}

template <>
LUMINARY_FUNCTION RGBF bsdf_evaluate(
  const MaterialContextVolume ctx, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_refraction,
  const float one_over_sampling_pdf) {
  is_refraction = false;
  return splat_color(volume_phase_evaluate(ctx, L) * one_over_sampling_pdf);
}

template <>
LUMINARY_FUNCTION RGBF bsdf_evaluate(
  const MaterialContextParticle ctx, const vec3 L, const BSDFSamplingHint sampling_hint, bool& is_refraction,
  const float one_over_sampling_pdf) {
  is_refraction = false;
  return scale_color(device.particles.albedo, volume_phase_evaluate(ctx, L) * one_over_sampling_pdf);
}

LUMINARY_FUNCTION BSDFRayContext
  bsdf_sample_context(const MaterialContextGeometry mat_ctx, const vec3 H, const vec3 L, const bool is_refraction) {
  BSDFRayContext context;

  context.NdotL = dot_product(mat_ctx.normal, L);
  context.NdotV = __saturatef(dot_product(mat_ctx.normal, mat_ctx.V));

  context.is_refraction = is_refraction;

  context.NdotL = (is_refraction) ? -context.NdotL : context.NdotL;

  const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx);

  bool total_reflection        = false;
  const vec3 refraction_vector = (context.is_refraction) ? L : refract_vector(mat_ctx.V, H, ior, total_reflection);

  context.NdotH = dot_product(mat_ctx.normal, H);
  context.HdotV = fabsf(dot_product(H, mat_ctx.V));
  context.HdotL = fabsf(dot_product(H, L));

  float flipH = 1.0f;
  if (context.NdotH < 0.0f) {
    flipH         = -1.0f;
    context.NdotH = -context.NdotH;
  }

  context.fresnel_dielectric = (total_reflection) ? 1.0f : bsdf_fresnel(scale_vector(H, flipH), mat_ctx.V, refraction_vector, ior);

  return context;
}

template <MaterialType TYPE, typename RANDOM_SET>
LUMINARY_FUNCTION BSDFSampleInfo<TYPE> bsdf_sample(const MaterialContext<TYPE> mat_ctx, const ushort2 pixel);

template <typename RANDOM_SET>
LUMINARY_FUNCTION BSDFSampleInfo<MATERIAL_GEOMETRY> bsdf_sample<MATERIAL_GEOMETRY>(
  const MaterialContextGeometry mat_ctx, const ushort2 pixel) {
  const float opacity = material_get_float<MATERIAL_GEOMETRY_PARAM_OPACITY>(mat_ctx);

  if (opacity < 1.0f) {
    const float transparency_random = random_1D(RANDOM_SET::OPACITY, pixel);

    if (transparency_random > opacity) {
      const RGBF albedo = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(mat_ctx);

      BSDFSampleInfo<MATERIAL_GEOMETRY> info;
      info.ray                 = scale_vector(mat_ctx.V, -1.0f);
      info.weight              = (mat_ctx.flags & MATERIAL_FLAG_COLORED_TRANSPARENCY) ? albedo : get_color(1.0f, 1.0f, 1.0f);
      info.is_microfacet_based = false;
      info.is_transparent_pass = true;

      return info;
    }
  }

  // Transformation to +Z-Up
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = quaternion_apply(rotation_to_z, mat_ctx.V);
  const vec3 face_normal_local   = quaternion_apply(rotation_to_z, material_get_normal<MATERIAL_GEOMETRY_PARAM_FACE_NORMAL>(mat_ctx));

  // Material Context (+Z-Up)
  MaterialContextGeometry mat_ctx_local = mat_ctx;

  mat_ctx_local.V      = V_local;
  mat_ctx_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  BSDFSampleInfo<MATERIAL_GEOMETRY> info;

  info.is_transparent_pass = false;
  info.is_microfacet_based = false;

  vec3 ray_local;

  const uint32_t base_substrate = mat_ctx_local.flags & MATERIAL_FLAG_BASE_SUBSTRATE_MASK;

  const bool include_diffuse =
    (base_substrate == MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE) && ((mat_ctx_local.flags & MATERIAL_FLAG_METALLIC) == 0);
  const bool include_refraction = (base_substrate == MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT);

  float sum_weights  = 0.0f;
  RGBF selected_eval = get_color(0.0f, 0.0f, 0.0f);

  float resampling_random = random_1D(RANDOM_SET::RESAMPLING, pixel);

  const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx);

  // Microfacet based sample
  if (true) {
    const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);

    const vec3 microfacet    = bsdf_microfacet_sample(V_local, roughness, pixel, RANDOM_SET::REFLECTION);
    const vec3 ray           = reflect_vector(mat_ctx_local.V, microfacet);
    const BSDFRayContext ctx = bsdf_sample_context(mat_ctx_local, microfacet, ray, false);
    const RGBF eval          = bsdf_evaluate_core(mat_ctx_local, ctx, BSDF_SAMPLING_MICROFACET, ray, face_normal_local);
    const float pdf          = bsdf_microfacet_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV);
    const float diffuse_pdf  = (include_diffuse) ? bsdf_diffuse_pdf(ctx.NdotL) : 0.0f;
    const float refraction_pdf =
      (include_refraction) ? bsdf_microfacet_refraction_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior)
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
    const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);

    const vec3 ray             = bsdf_diffuse_sample(random_2D(RANDOM_SET::DIFFUSE, pixel));
    const vec3 microfacet      = normalize_vector(add_vector(mat_ctx_local.V, ray));
    const BSDFRayContext ctx   = bsdf_sample_context(mat_ctx_local, microfacet, ray, false);
    const RGBF eval            = bsdf_evaluate_core(mat_ctx_local, ctx, BSDF_SAMPLING_DIFFUSE, ray, face_normal_local);
    const float pdf            = bsdf_diffuse_pdf(ctx.NdotL);
    const float microfacet_pdf = bsdf_microfacet_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV);
    const float refraction_pdf =
      (include_refraction) ? bsdf_microfacet_refraction_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior)
                           : 0.0f;

    const float sum_pdf    = pdf + microfacet_pdf + refraction_pdf;
    const float mis_weight = (sum_pdf > 0.0f) ? pdf / sum_pdf : 0.0f;

    const float weight = color_importance(eval) * mis_weight;

    UTILS_CHECK_NANS(pixel, weight);

    sum_weights += weight;

    const float resampling_probability = weight / sum_weights;
    if (resampling_random < resampling_probability) {
      ray_local                = ray;
      selected_eval            = eval;
      info.is_transparent_pass = false;
      info.is_microfacet_based = false;

      resampling_random = random_saturate(resampling_random / resampling_probability);
    }
    else {
      resampling_random = random_saturate((resampling_random - resampling_probability) / (1.0f - resampling_probability));
    }
  }

  // Microfacet refraction based sample
  if (include_refraction) {
    const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);

    bool total_reflection;
    const vec3 microfacet    = bsdf_microfacet_refraction_sample(V_local, roughness, pixel, RANDOM_SET::REFRACTION);
    const vec3 ray           = refract_vector(mat_ctx_local.V, microfacet, ior, total_reflection);
    const BSDFRayContext ctx = bsdf_sample_context(mat_ctx_local, microfacet, ray, !total_reflection);
    const RGBF eval          = bsdf_evaluate_core(mat_ctx_local, ctx, BSDF_SAMPLING_MICROFACET_REFRACTION, ray, face_normal_local);

    float mis_weight = 1.0f;

    // If it is a reflection, then the direction could have been sampled from a microfacet reflection,
    // hence we need to compute its MIS weight.
    if (total_reflection) {
      const float pdf = bsdf_microfacet_refraction_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
      const float reflection_pdf = bsdf_microfacet_pdf(V_local, roughness, ctx.NdotH, ctx.NdotV);
      const float diffuse_pdf    = (include_diffuse) ? bsdf_diffuse_pdf(ctx.NdotL) : 0.0f;

      const float sum_pdf = pdf + reflection_pdf + diffuse_pdf;
      mis_weight          = (sum_pdf > 0.0f) ? pdf / sum_pdf : 0.0f;
    }

    const float weight = color_importance(eval) * mis_weight;

    UTILS_CHECK_NANS(pixel, weight);

    sum_weights += weight;

    const float resampling_probability = weight / sum_weights;
    if (resampling_random < resampling_probability) {
      ray_local                = ray;
      selected_eval            = eval;
      info.is_transparent_pass = !total_reflection;
      info.is_microfacet_based = true;

      resampling_random = random_saturate(resampling_random / resampling_probability);
    }
    else {
      resampling_random = random_saturate((resampling_random - resampling_probability) / (1.0f - resampling_probability));
    }
  }

  // For RIS we need to evaluate f / |f| here. This is unstable for low roughness and microfacet BRDFs.
  // Hence we use a little trick, f / p can be evaluated in a stable manner when p is the microfacet PDF,
  // and thus we evaluate f / |f| = (f / p) / |f / p|.
  info.weight =
    (sum_weights > 0.0f) ? scale_color(selected_eval, sum_weights / color_importance(selected_eval)) : get_color(0.0f, 0.0f, 0.0f);

  UTILS_CHECK_NANS(pixel, info.weight);

  info.ray = normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), ray_local));

  return info;
}

template <typename RANDOM_SET>
LUMINARY_FUNCTION BSDFSampleInfo<MATERIAL_VOLUME> bsdf_sample<MATERIAL_VOLUME>(const MaterialContextVolume ctx, const ushort2 pixel) {
  const float random_choice = random_1D(RANDOM_SET::RESAMPLING, pixel);
  const float2 random_dir   = random_2D(RANDOM_SET::DIFFUSE, pixel);

  const vec3 I = scale_vector(ctx.V, -1.0f);

  BSDFSampleInfo<MATERIAL_VOLUME> info;

  info.ray = (ctx.descriptor.type != VOLUME_TYPE_OCEAN)
               ? jendersie_eon_phase_sample(I, device.fog.droplet_diameter, random_dir, random_choice)
               : ocean_phase_sampling(I, random_dir, random_choice);

  return info;
}

template <typename RANDOM_SET>
LUMINARY_FUNCTION BSDFSampleInfo<MATERIAL_PARTICLE> bsdf_sample<MATERIAL_PARTICLE>(const MaterialContextParticle ctx, const ushort2 pixel) {
  const float random_choice = random_1D(RANDOM_SET::RESAMPLING, pixel);
  const float2 random_dir   = random_2D(RANDOM_SET::DIFFUSE, pixel);

  BSDFSampleInfo<MATERIAL_PARTICLE> info;
  info.ray    = jendersie_eon_phase_sample(scale_vector(ctx.V, -1.0f), device.particles.phase_diameter, random_dir, random_choice);
  info.weight = device.particles.albedo;

  return info;
}

LUMINARY_FUNCTION vec3
  bsdf_sample_microfacet_reflection(const MaterialContextGeometry mat_ctx, const ushort2 pixel, const RandomTarget random_target_ray) {
  const float roughness         = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);
  const vec3 sampled_microfacet = bsdf_microfacet_sample(mat_ctx.V, roughness, pixel, random_target_ray);

  return reflect_vector(mat_ctx.V, sampled_microfacet);
}

LUMINARY_FUNCTION vec3
  bsdf_sample_microfacet_refraction(const MaterialContextGeometry mat_ctx, const ushort2 pixel, const RandomTarget random_target_ray) {
  const float roughness         = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);
  const vec3 sampled_microfacet = bsdf_microfacet_refraction_sample(mat_ctx.V, roughness, pixel, random_target_ray);

  const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx);

  bool total_reflection;
  return refract_vector(mat_ctx.V, sampled_microfacet, ior, total_reflection);
}

LUMINARY_FUNCTION vec3
  bsdf_sample_diffuse(const MaterialContextGeometry mat_ctx, const ushort2 pixel, const RandomTarget random_target_ray) {
  return bsdf_diffuse_sample(random_2D(random_target_ray, pixel));
}

LUMINARY_FUNCTION void bsdf_sample_for_light_probabilities(
  const MaterialContextGeometry mat_ctx, float& reflection_prob, float& refraction_prob, float& diffuse_prob) {
  // TODO: Consider creating a context and sampling also proportional to albedo etc.
  // TODO: There is this issue where I importance sample the lights too well but end up picking occluded
  // lights which will give me terrible convergence.
  float microfacet_reflection_weight;
  float microfacet_refraction_weight;
  float diffuse_weight;

  switch (mat_ctx.flags & MATERIAL_FLAG_BASE_SUBSTRATE_MASK) {
    case MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE:
      microfacet_reflection_weight = 1.0f;
      microfacet_refraction_weight = 0.0f;
      diffuse_weight               = (mat_ctx.flags & MATERIAL_FLAG_METALLIC) ? 0.0f : 1.0f;
      break;
    case MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT:
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

template <MaterialType TYPE>
LUMINARY_FUNCTION vec3 bsdf_sample_for_sun(const MaterialContext<TYPE> mat_ctx, const ushort2 pixel, bool& is_refraction, bool& is_valid);

template <>
LUMINARY_FUNCTION vec3
  bsdf_sample_for_sun<MATERIAL_GEOMETRY>(const MaterialContextGeometry mat_ctx, const ushort2 pixel, bool& is_refraction, bool& is_valid) {
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(mat_ctx.normal);
  const vec3 V_local             = quaternion_apply(rotation_to_z, mat_ctx.V);

  MaterialContextGeometry mat_ctx_local = mat_ctx;

  mat_ctx_local.V      = V_local;
  mat_ctx_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  float reflection_probability, refraction_probability, diffuse_probability;
  bsdf_sample_for_light_probabilities(mat_ctx_local, reflection_probability, refraction_probability, diffuse_probability);

  const float random_method = random_1D(MaterialContextGeometry::RANDOM_DL_SUN::BSDF_METHOD, pixel);

  vec3 ray_local;
  if (random_method < reflection_probability) {
    ray_local     = bsdf_sample_microfacet_reflection(mat_ctx_local, pixel, MaterialContextGeometry::RANDOM_DL_SUN::BSDF);
    is_refraction = false;
  }
  else if (random_method < reflection_probability + refraction_probability) {
    ray_local     = bsdf_sample_microfacet_refraction(mat_ctx_local, pixel, MaterialContextGeometry::RANDOM_DL_SUN::BSDF);
    is_refraction = true;
  }
  else {
    ray_local     = bsdf_sample_diffuse(mat_ctx_local, pixel, MaterialContextGeometry::RANDOM_DL_SUN::BSDF);
    is_refraction = false;
  }

  is_valid = (is_refraction) ? ray_local.z < 0.0f : ray_local.z > 0.0f;

  return normalize_vector(quaternion_apply(quaternion_inverse(rotation_to_z), ray_local));
}

template <>
LUMINARY_FUNCTION vec3
  bsdf_sample_for_sun<MATERIAL_VOLUME>(const MaterialContextVolume mat_ctx, const ushort2 pixel, bool& is_refraction, bool& is_valid) {
  const float2 random_dir   = random_2D(MaterialContextVolume::RANDOM_DL_SUN::BSDF, pixel);
  const float random_method = random_1D(MaterialContextVolume::RANDOM_DL_SUN::BSDF_METHOD, pixel);

  const vec3 I = scale_vector(mat_ctx.V, -1.0f);

  const vec3 scatter_ray = (mat_ctx.descriptor.type != VOLUME_TYPE_OCEAN)
                             ? jendersie_eon_phase_sample(I, device.fog.droplet_diameter, random_dir, random_method)
                             : ocean_phase_sampling(I, random_dir, random_method);

  is_refraction = true;
  is_valid      = true;

  return scatter_ray;
}

template <>
LUMINARY_FUNCTION vec3
  bsdf_sample_for_sun<MATERIAL_PARTICLE>(const MaterialContextParticle mat_ctx, const ushort2 pixel, bool& is_refraction, bool& is_valid) {
  const float2 random_dir   = random_2D(MaterialContextParticle::RANDOM_DL_SUN::BSDF, pixel);
  const float random_method = random_1D(MaterialContextParticle::RANDOM_DL_SUN::BSDF_METHOD, pixel);

  is_refraction = true;
  is_valid      = true;

  return jendersie_eon_phase_sample(scale_vector(mat_ctx.V, -1.0f), device.particles.phase_diameter, random_dir, random_method);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float bsdf_sample_for_sun_pdf(const MaterialContext<TYPE> mat_ctx, const vec3 L);

template <>
LUMINARY_FUNCTION float bsdf_sample_for_sun_pdf<MATERIAL_GEOMETRY>(const MaterialContextGeometry mat_ctx, const vec3 L) {
  float reflection_probability, refraction_probability, diffuse_probability;
  bsdf_sample_for_light_probabilities(mat_ctx, reflection_probability, refraction_probability, diffuse_probability);

  const BSDFRayContext ctx = bsdf_evaluate_analyze(mat_ctx, L);
  const float roughness    = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(mat_ctx);

  if (ctx.is_refraction) {
    const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(mat_ctx);
    const float microfacet_refraction_pdf =
      bsdf_microfacet_refraction_pdf(mat_ctx.V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);

    return refraction_probability * microfacet_refraction_pdf;
  }
  else {
    const float microfacet_reflection_pdf = bsdf_microfacet_pdf(mat_ctx.V, roughness, ctx.NdotH, ctx.NdotV);
    const float diffuse_pdf               = bsdf_diffuse_pdf(ctx.NdotL);

    return reflection_probability * microfacet_reflection_pdf + diffuse_probability * diffuse_pdf;
  }
}

template <>
LUMINARY_FUNCTION float bsdf_sample_for_sun_pdf<MATERIAL_VOLUME>(const MaterialContextVolume mat_ctx, const vec3 L) {
  const float cos_angle = -dot_product(mat_ctx.V, L);

  float pdf;
  if (mat_ctx.descriptor.type == VOLUME_TYPE_OCEAN) {
    pdf = ocean_phase(cos_angle);
  }
  else {
    const JendersieEonParams params = jendersie_eon_phase_parameters(device.fog.droplet_diameter);
    pdf                             = jendersie_eon_phase_function(cos_angle, params);
  }

  return pdf;
}

template <>
LUMINARY_FUNCTION float bsdf_sample_for_sun_pdf<MATERIAL_PARTICLE>(const MaterialContextParticle mat_ctx, const vec3 L) {
  const JendersieEonParams params = jendersie_eon_phase_parameters(device.particles.phase_diameter);

  return jendersie_eon_phase_function(-dot_product(mat_ctx.V, L), params);
}

#endif /* CU_BSDF_H */
