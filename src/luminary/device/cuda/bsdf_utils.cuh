#ifndef CU_BSDF_UTILS_H
#define CU_BSDF_UTILS_H

#include "material.cuh"
#include "math.cuh"
#include "utils.cuh"

struct BSDFRayContext {
  vec3 V;
  float fresnel_dielectric;
  float NdotH;
  float NdotL;
  float NdotV;
  float HdotL;
  float HdotV;
  bool is_refraction;
};

// TODO: Prefetch these for light sampling.
struct BSDFDirectionalAlbedos {
  float conductor;
  float glossy;
  RGBF dielectric;
  RGBF dielectric_inv;
} typedef BSDFDirectionalAlbedos;

enum BSDFMaterial { BSDF_CONDUCTOR = 0, BSDF_GLOSSY = 1, BSDF_DIELECTRIC = 2 } typedef BSDFMaterial;

template <MaterialType TYPE>
struct BSDFSampleInfo {};

template <>
struct BSDFSampleInfo<MATERIAL_GEOMETRY> {
  vec3 ray;
  RGBF weight;
  bool is_transparent_pass;
  bool is_microfacet_based;
};

template <>
struct BSDFSampleInfo<MATERIAL_VOLUME> {
  vec3 ray;
  const RGBF weight                         = {1.0f, 1.0f, 1.0f};
  static constexpr bool is_transparent_pass = false;
  static constexpr bool is_microfacet_based = false;
};

template <>
struct BSDFSampleInfo<MATERIAL_PARTICLE> {
  vec3 ray;
  RGBF weight;
  static constexpr bool is_transparent_pass = false;
  static constexpr bool is_microfacet_based = false;
};

enum BSDFSamplingHint {
  BSDF_SAMPLING_GENERAL               = 0,
  BSDF_SAMPLING_MICROFACET            = 1,
  BSDF_SAMPLING_DIFFUSE               = 2,
  BSDF_SAMPLING_MICROFACET_REFRACTION = 3
};

template <MaterialType TYPE>
LUMINARY_FUNCTION bool bsdf_is_pass_through_ray(const MaterialContext<TYPE> ctx, const BSDFSampleInfo<TYPE> info) {
  return false;
}

template <>
LUMINARY_FUNCTION bool bsdf_is_pass_through_ray<MATERIAL_GEOMETRY>(
  const MaterialContextGeometry ctx, const BSDFSampleInfo<MATERIAL_GEOMETRY> info) {
  const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params);
  return info.is_transparent_pass && ((ior == 1.0f) || (info.is_microfacet_based == false));
}

///////////////////////////////////////////////////
// Fresnel
///////////////////////////////////////////////////

LUMINARY_FUNCTION float bsdf_fresnel(const vec3 normal, const vec3 V, const vec3 refraction, const float ior) {
  const float NdotV = dot_product(V, normal);
  const float NdotT = -dot_product(refraction, normal);

  const float s_pol_term1 = ior * NdotV;
  const float s_pol_term2 = 1.0f * NdotT;

  const float p_pol_term1 = ior * NdotT;
  const float p_pol_term2 = 1.0f * NdotV;

  float reflection_s_pol = (s_pol_term1 - s_pol_term2) / (s_pol_term1 + s_pol_term2);
  float reflection_p_pol = (p_pol_term1 - p_pol_term2) / (p_pol_term1 + p_pol_term2);

  reflection_s_pol *= reflection_s_pol;
  reflection_p_pol *= reflection_p_pol;

  return __saturatef(0.5f * (reflection_s_pol + reflection_p_pol));
}

/*
 * Standard Schlick Fresnel approximation. Unused.
 * @param f0 Specular F0.
 * @param f90 Shadow term.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
LUMINARY_FUNCTION RGBF bsdf_fresnel_schlick(const RGBF f0, const float f90, const float HdotV) {
  const float one_minus_HdotV = 1.0f - fabsf(HdotV);
  const float pow2            = one_minus_HdotV * one_minus_HdotV;

  // powf(1.0f - NdotV, 5.0f)
  const float t = pow2 * pow2 * one_minus_HdotV;

  RGBF result = f0;
  RGBF diff   = sub_color(get_color(f90, f90, f90), f0);
  result      = fma_color(diff, t, result);

  return result;
}

LUMINARY_FUNCTION float bsdf_shadowed_F90(const RGBF specular_f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(specular_f0));
}

///////////////////////////////////////////////////
// Refraction
///////////////////////////////////////////////////

LUMINARY_FUNCTION float bsdf_refraction_index_ambient(const vec3 position, const vec3 ray) {
  if (device.ocean.active && position.y < device.ocean.height)
    return device.ocean.refractive_index;

  return 1.0f;
}

// Get normal vector based on incoming ray and refracted ray: PBRT v3 Chapter 8.4.4
LUMINARY_FUNCTION vec3 bsdf_normal_from_pair(const vec3 L, const vec3 V, const float refraction_index) {
  const vec3 refraction_normal = add_vector(L, scale_vector(V, refraction_index));

  const float length = get_length(refraction_normal);

  return (length > 0.0f) ? scale_vector(refraction_normal, 1.0f / length) : V;
}

///////////////////////////////////////////////////
// Microfacet
///////////////////////////////////////////////////

// S can be either L or V, doesn't matter.
LUMINARY_FUNCTION float bsdf_microfacet_evaluate_smith_G1_GGX(const float roughness4, const float NdotS) {
  const float NdotS2 = fmaxf(0.0001f, NdotS * NdotS);
  return 2.0f / (sqrtf(((roughness4 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(
  const float roughness4, const float NdotL, const float NdotV) {
  const float a = NdotV * sqrtf(roughness4 + NdotL * (NdotL - roughness4 * NdotL));
  const float b = NdotL * sqrtf(roughness4 + NdotV * (NdotV - roughness4 * NdotV));
  return 0.5f / (a + b);
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate_smith_G2_over_G1_height_correlated_GGX(
  const float roughness4, const float NdotL, const float NdotV) {
  const float G1V = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV);
  const float G1L = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotL);
  return G1L / (G1V + G1L - G1V * G1L);
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate_D_GGX(const float NdotH, const float roughness4) {
  const float NdotH2 = fminf(NdotH * NdotH, 1.0f);
  const float a      = 1.0f - NdotH2 + roughness4 * NdotH2;

  return roughness4 / (PI * a * a);
}

/*
 * Samples a normal from the hemisphere with GGX VNDF for reflections. Method found in [EtoT23].
 * Note that we ignore the case of NdotV (==data.V.z) < 0.0f due to how our sampling works,
 * hence we can optimize their code for NdotV >= 0.0f.
 *
 * [EtoT23] K. Eto and Y. Tokuyoshi, "Bounded VNDF Sampling for Smith–GGX Reflections", ACM SIGGRAPH Asia, 2023.
 */
LUMINARY_FUNCTION vec3 bsdf_microfacet_sample_normal(const vec3 V, const float roughness, const float2 random) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const vec3 v = normalize_vector(get_vector(roughness2 * V.x, roughness2 * V.y, V.z));

  const float phi       = 2.0f * PI * random.x;
  const float s         = 1.0f + sqrtf(V.x * V.x + V.y * V.y);
  const float s2        = s * s;
  const float k         = (1.0f - roughness4) * s2 / (s2 + roughness4 * V.z * V.z);
  const float b         = k * v.z;
  const float z         = (1.0f - random.y) * (1.0f + b) - b;
  const float sin_theta = sqrtf(__saturatef(1.0f - z * z));
  const float x         = sin_theta * cosf(phi);
  const float y         = sin_theta * sinf(phi);
  const vec3 sampled    = add_vector(get_vector(x, y, z), v);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

LUMINARY_FUNCTION float bsdf_microfacet_pdf(const vec3 V, const float roughness, const float NdotH, const float NdotV) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  // NdotV == data.V.z

  const float D = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);

  const float len2 = roughness4 * (V.x * V.x + V.y * V.y);
  const float t    = sqrtf(len2 + V.z * V.z);

  const float s  = 1.0f + sqrtf(V.x * V.x + V.y * V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * V.z * V.z);
  return D / (2.0f * (k * NdotV + t));
}

LUMINARY_FUNCTION vec3 bsdf_microfacet_sample(const vec3 V, const float roughness, const PathID& path_id, const uint32_t target) {
  const float2 random = random_2D(target, path_id);
  return bsdf_microfacet_sample_normal(V, roughness, random);
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate(const float roughness, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D  = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2 = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // D * G2 * NdotL / (4.0f * NdotL * NdotV)
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * NdotL;
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate_sampled_microfacet(
  const vec3 V, const float roughness, const float NdotL, const float NdotV) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float G2 = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // NdotV == data.V.z

  const float len2 = roughness4 * (V.x * V.x + V.y * V.y);
  const float t    = sqrtf(len2 + V.z * V.z);

  const float s  = 1.0f + sqrtf(V.x * V.x + V.y * V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * V.z * V.z);

  // G2 contains (4 * NdotL * NdotV) in the denominator
  return 2.0f * (k * NdotV + t) * G2 * NdotL;
}

LUMINARY_FUNCTION float bsdf_microfacet_evaluate_sampled_diffuse(
  const float roughness, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D  = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2 = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * PI;
}

/*
 * Samples a normal from the hemisphere with GGX VNDF for refractions. Method found in [DupB23].
 *
 * [DupB23] J. Dupuy and A. Benyoub, "Sampling Visible GGX Normals with Spherical Caps", 2023. arXiv:2306.05044
 */
LUMINARY_FUNCTION vec3 bsdf_microfacet_refraction_sample_normal(const vec3 V, const float roughness, const float2 random) {
  const float roughness2 = roughness * roughness;
  const vec3 v           = normalize_vector(get_vector(roughness2 * V.x, roughness2 * V.y, V.z));

  const float phi       = 2.0f * PI * random.x;
  const float z         = (1.0f - random.y) * (1.0f + v.z) - v.z;
  const float sin_theta = sqrtf(__saturatef(1.0f - z * z));
  const float x         = sin_theta * cosf(phi);
  const float y         = sin_theta * sinf(phi);
  const vec3 sampled    = add_vector(get_vector(x, y, z), v);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

LUMINARY_FUNCTION float bsdf_microfacet_refraction_pdf(
  const vec3 V, const float roughness, const float NdotH, const float NdotV, const float NdotL, const float HdotV, const float HdotL,
  const float refraction_index) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D  = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G1 = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV);

  float denominator = refraction_index * HdotV + HdotL;
  denominator       = denominator * denominator;

  // See Heitz14.
  return D * G1 * (HdotV / NdotV) * (HdotL / denominator);
}

LUMINARY_FUNCTION vec3
  bsdf_microfacet_refraction_sample(const vec3 V, const float roughness, const PathID& path_id, const uint32_t target) {
  const float2 random = random_2D(target, path_id);
  return bsdf_microfacet_refraction_sample_normal(V, roughness, random);
}

LUMINARY_FUNCTION float bsdf_microfacet_refraction_evaluate(
  const float roughness, const float HdotL, const float HdotV, const float NdotH, const float NdotL, const float NdotV,
  const float refraction_index) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D  = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2 = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  float denominator = refraction_index * HdotV + HdotL;
  denominator       = denominator * denominator;

  // See Walter07.
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return 4.0f * NdotL * HdotV * HdotL * D * G2 / denominator;
}

LUMINARY_FUNCTION float bsdf_microfacet_refraction_evaluate_sampled_microfacet(
  const float roughness, const float HdotL, const float HdotV, const float NdotH, const float NdotL, const float NdotV,
  const float refraction_index) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float G2_over_G1 = bsdf_microfacet_evaluate_smith_G2_over_G1_height_correlated_GGX(roughness4, NdotL, NdotV);

  // See Heitz14.
  return G2_over_G1;
}

///////////////////////////////////////////////////
// Diffuse
///////////////////////////////////////////////////

LUMINARY_FUNCTION vec3 bsdf_diffuse_sample(const float2 random) {
  return sample_ray_sphere(random.x, random.y);
}

LUMINARY_FUNCTION float bsdf_diffuse_pdf(const float NdotL) {
  return __saturatef(NdotL) * (1.0f / PI);
}

LUMINARY_FUNCTION float bsdf_diffuse_evaluate(const float NdotL) {
  return __saturatef(NdotL) * (1.0f / PI);
}

LUMINARY_FUNCTION float bsdf_diffuse_evaluate_sampled_diffuse() {
  return 1.0f;
}

LUMINARY_FUNCTION float bsdf_diffuse_evaluate_sampled_microfacet(
  const vec3 V, const float roughness, const float NdotL, const float NdotH, const float NdotV) {
  const float roughness2 = roughness * roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);

  const float len2 = roughness4 * (V.x * V.x + V.y * V.y);
  const float t    = sqrtf(len2 + V.z * V.z);

  const float s  = 1.0f + sqrtf(V.x * V.x + V.y * V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * V.z * V.z);

  return NdotL * (2.0f * (k * NdotV + t)) / (PI * D);
}

///////////////////////////////////////////////////
// Material
///////////////////////////////////////////////////

LUMINARY_FUNCTION float bsdf_conductor_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float>(device.bsdf_lut_conductor.handle, NdotV, roughness);
}

LUMINARY_FUNCTION RGBF bsdf_conductor(
  const MaterialParams& params, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return splat_color(0.0f);

  if (MATERIAL_IS_SUBSTRATE_OPAQUE(params.flags) == false)
    return splat_color(0.0f);

  if ((params.flags & MATERIAL_FLAG_METALLIC) == 0)
    return splat_color(0.0f);

  float ior = 1.0f;
  if (sampling_hint == BSDF_SAMPLING_MICROFACET_REFRACTION)
    ior = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);

  const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);

  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = bsdf_microfacet_evaluate_sampled_diffuse(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_MICROFACET_REFRACTION:
      ss_term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV)
                / bsdf_microfacet_refraction_pdf(ctx.V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
      break;
  };

  const RGBF albedo = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(params);

  const float directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, roughness);

  const RGBF f0_conductor      = albedo;
  const RGBF fresnel_conductor = bsdf_fresnel_schlick(f0_conductor, bsdf_shadowed_F90(f0_conductor), ctx.HdotV);

  const RGBF ss_term_with_fresnel = scale_color(fresnel_conductor, ss_term);
  const RGBF ms_term_with_fresnel = mul_color(f0_conductor, scale_color(fresnel_conductor, ((1.0f / directional_albedo) - 1.0f) * ss_term));

  return add_color(ss_term_with_fresnel, ms_term_with_fresnel);
}

LUMINARY_FUNCTION float bsdf_glossy_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float>(device.bsdf_lut_glossy.handle, NdotV, roughness);
}

LUMINARY_FUNCTION RGBF bsdf_glossy(
  const MaterialParams& params, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return splat_color(0.0f);

  if (MATERIAL_IS_SUBSTRATE_OPAQUE(params.flags) == false)
    return splat_color(0.0f);

  if ((params.flags & MATERIAL_FLAG_METALLIC) != 0)
    return splat_color(0.0f);

  float ior = 1.0f;
  if (sampling_hint == BSDF_SAMPLING_MICROFACET_REFRACTION)
    ior = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);

  const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);

  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = bsdf_microfacet_evaluate_sampled_diffuse(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_MICROFACET_REFRACTION:
      ss_term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV)
                / bsdf_microfacet_refraction_pdf(ctx.V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
      break;
  };

  float diff_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      diff_term = bsdf_diffuse_evaluate(ctx.NdotL) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_DIFFUSE:
      diff_term = 1.0f;
      break;
    case BSDF_SAMPLING_MICROFACET:
      diff_term = bsdf_diffuse_evaluate_sampled_microfacet(ctx.V, roughness, ctx.NdotL, ctx.NdotH, ctx.NdotV);
      break;
    case BSDF_SAMPLING_MICROFACET_REFRACTION:
      diff_term = bsdf_diffuse_evaluate(ctx.NdotL)
                  / bsdf_microfacet_refraction_pdf(ctx.V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
      break;
  };

  const RGBF albedo = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(params);

  const float conductor_directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, roughness);
  const float glossy_directional_albedo    = bsdf_glossy_directional_albedo(ctx.NdotV, roughness);

  const RGBF f0_glossy      = get_color(0.04f, 0.04f, 0.04f);
  const RGBF fresnel_glossy = bsdf_fresnel_schlick(f0_glossy, bsdf_shadowed_F90(f0_glossy), ctx.HdotV);

  const RGBF ss_term_with_fresnel = scale_color(fresnel_glossy, ss_term / conductor_directional_albedo);
  const RGBF diff_term_with_color = scale_color(albedo, diff_term * (1.0f - glossy_directional_albedo));

  return add_color(ss_term_with_fresnel, diff_term_with_color);
}

LUMINARY_FUNCTION float bsdf_dielectric_directional_albedo(const float NdotV, const float roughness, const float ior) {
  const bool use_inv = (ior > 1.0f);

  const float ior_tex_coord = use_inv ? (ior - 1.0f) * 0.5f : (1.0f / ior - 1.0f) * 0.5f;

  const DeviceTextureHandle handle = use_inv ? device.bsdf_lut_dielectric_inv.handle : device.bsdf_lut_dielectric.handle;

  return tex3D<float>(handle, NdotV, roughness, ior_tex_coord);
}

LUMINARY_FUNCTION RGBF bsdf_dielectric(
  const MaterialParams& params, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return splat_color(0.0f);

  if (MATERIAL_IS_SUBSTRATE_TRANSLUCENT(params.flags) == false)
    return splat_color(0.0f);

  const float ior       = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);
  const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(params);

  float term;
  if (ctx.is_refraction) {
    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        term = bsdf_microfacet_refraction_evaluate(roughness, ctx.HdotL, ctx.HdotV, ctx.NdotH, ctx.NdotL, ctx.NdotV, ior)
               * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        term =
          bsdf_microfacet_refraction_evaluate_sampled_microfacet(roughness, ctx.HdotL, ctx.HdotV, ctx.NdotH, ctx.NdotL, ctx.NdotV, ior);
        break;
      case BSDF_SAMPLING_MICROFACET:
      case BSDF_SAMPLING_DIFFUSE:
        term = 0.0f;
        break;
    };

    term *= (1.0f - ctx.fresnel_dielectric);
  }
  else {
    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_MICROFACET:
        term = bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, ctx.NdotL, ctx.NdotV);
        break;
      case BSDF_SAMPLING_DIFFUSE:
        term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV) / bsdf_diffuse_pdf(ctx.NdotL);
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        term = bsdf_microfacet_evaluate(roughness, ctx.NdotH, ctx.NdotL, ctx.NdotV)
               / bsdf_microfacet_refraction_pdf(ctx.V, roughness, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ior);
        break;
    };

    term *= ctx.fresnel_dielectric;
  }

  const RGBF albedo = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(params);

  const float dielectric_directional_albedo = bsdf_dielectric_directional_albedo(ctx.NdotV, roughness, ior);
  term /= dielectric_directional_albedo;

  if (ior == 1.0f && ctx.is_refraction) {
    // TODO: Energy conservation does not work correctly for dielectric, investigate.
    term = (sampling_hint == BSDF_SAMPLING_MICROFACET_REFRACTION) ? 1.0f : 0.0f;
  }

  return scale_color(albedo, term);
}

///////////////////////////////////////////////////
// Multiscattering
///////////////////////////////////////////////////

LUMINARY_FUNCTION RGBF bsdf_multiscattering_evaluate(
  const MaterialParams& params, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  const float opacity = material_get_float<MATERIAL_GEOMETRY_PARAM_OPACITY>(params);

  if (ctx.is_refraction)
    return scale_color(bsdf_dielectric(params, ctx, sampling_hint, one_over_sampling_pdf), opacity);

  RGBF conductor  = bsdf_conductor(params, ctx, sampling_hint, one_over_sampling_pdf);
  RGBF glossy     = bsdf_glossy(params, ctx, sampling_hint, one_over_sampling_pdf);
  RGBF dielectric = bsdf_dielectric(params, ctx, sampling_hint, one_over_sampling_pdf);

  return scale_color(add_color(add_color(conductor, glossy), dielectric), opacity);
}

#endif /* CU_BSDF_UTILS_H */
