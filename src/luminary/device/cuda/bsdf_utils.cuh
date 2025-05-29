#ifndef CU_BSDF_UTILS_H
#define CU_BSDF_UTILS_H

#include "math.cuh"
#include "splitting.cuh"
#include "utils.cuh"

struct BSDFRayContext {
  vec3 H;
  RGBF f0_conductor;
  RGBF f0_glossy;
  RGBF fresnel_conductor;
  RGBF fresnel_glossy;
  float fresnel_dielectric;
  float NdotH;
  float NdotL;
  float NdotV;
  float refraction_index;
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

struct BSDFSampleInfo {
  RGBF weight;
  bool is_transparent_pass;
  bool is_microfacet_based;
} typedef BSDFSampleInfo;

enum BSDFSamplingHint {
  BSDF_SAMPLING_GENERAL               = 0,
  BSDF_SAMPLING_MICROFACET            = 1,
  BSDF_SAMPLING_DIFFUSE               = 2,
  BSDF_SAMPLING_MICROFACET_REFRACTION = 3
};

#if !defined(PHASE_KERNEL)
__device__ bool bsdf_is_pass_through_ray(const bool is_transparent_pass, const float ior_in, const float ior_out) {
  return is_transparent_pass && (ior_in == ior_out);
}
#else  /* !PHASE_KERNEL */
__device__ bool bsdf_is_pass_through_ray(const bool is_transparent_pass, const float ior_in, const float ior_out) {
  return false;
}
#endif /* PHASE_KERNEL */

///////////////////////////////////////////////////
// Fresnel
///////////////////////////////////////////////////

__device__ float bsdf_fresnel(const vec3 normal, const vec3 V, const vec3 refraction, const float index_in, const float index_out) {
  const float NdotV = dot_product(V, normal);
  const float NdotT = -dot_product(refraction, normal);

  const float s_pol_term1 = index_in * NdotV;
  const float s_pol_term2 = index_out * NdotT;

  const float p_pol_term1 = index_in * NdotT;
  const float p_pol_term2 = index_out * NdotV;

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
__device__ RGBF bsdf_fresnel_schlick(const RGBF f0, const float f90, const float HdotV) {
  const float one_minus_HdotV = 1.0f - fabsf(HdotV);
  const float pow2            = one_minus_HdotV * one_minus_HdotV;

  // powf(1.0f - NdotV, 5.0f)
  const float t = pow2 * pow2 * one_minus_HdotV;

  RGBF result = f0;
  RGBF diff   = sub_color(get_color(f90, f90, f90), f0);
  result      = fma_color(diff, t, result);

  return result;
}

__device__ float bsdf_shadowed_F90(const RGBF specular_f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(specular_f0));
}

///////////////////////////////////////////////////
// Refraction
///////////////////////////////////////////////////

__device__ float bsdf_refraction_index_ambient(const vec3 position, const vec3 ray) {
  if (device.ocean.active && position.y < device.ocean.height)
    return device.ocean.refractive_index;

  return 1.0f;
}

// Get normal vector based on incoming ray and refracted ray: PBRT v3 Chapter 8.4.4
__device__ vec3 bsdf_normal_from_pair(const vec3 L, const vec3 V, const float refraction_index) {
  const vec3 refraction_normal = add_vector(L, scale_vector(V, refraction_index));

  const float length = get_length(refraction_normal);

  return (length > 0.0f) ? scale_vector(refraction_normal, 1.0f / length) : V;
}

///////////////////////////////////////////////////
// Microfacet
///////////////////////////////////////////////////

// S can be either L or V, doesn't matter.
__device__ float bsdf_microfacet_evaluate_smith_G1_GGX(const float roughness4, const float NdotS) {
  const float NdotS2 = fmaxf(0.0001f, NdotS * NdotS);
  return 2.0f / (sqrtf(((roughness4 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__ float bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(const float roughness4, const float NdotL, const float NdotV) {
  const float a = NdotV * sqrtf(roughness4 + NdotL * (NdotL - roughness4 * NdotL));
  const float b = NdotL * sqrtf(roughness4 + NdotV * (NdotV - roughness4 * NdotV));
  return 0.5f / (a + b);
}

__device__ float bsdf_microfacet_evaluate_smith_G2_over_G1_height_correlated_GGX(
  const float roughness4, const float NdotL, const float NdotV) {
  const float G1V = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV);
  const float G1L = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotL);
  return G1L / (G1V + G1L - G1V * G1L);
}

__device__ float bsdf_microfacet_evaluate_D_GGX(const float NdotH, const float roughness4) {
  if (roughness4 < 1e-8f)
    return 0.0f;

  const float a = ((roughness4 - 1.0f) * NdotH * NdotH + 1.0f);
  return roughness4 / (PI * a * a);
}

/*
 * Samples a normal from the hemisphere with GGX VNDF for reflections. Method found in [EtoT23].
 * Note that we ignore the case of NdotV (==data.V.z) < 0.0f due to how our sampling works,
 * hence we can optimize their code for NdotV >= 0.0f.
 *
 * [EtoT23] K. Eto and Y. Tokuyoshi, "Bounded VNDF Sampling for Smith–GGX Reflections", ACM SIGGRAPH Asia, 2023.
 */
__device__ vec3 bsdf_microfacet_sample_normal(const GBufferData data, const float2 random) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;

  const vec3 v = normalize_vector(get_vector(roughness2 * data.V.x, roughness2 * data.V.y, data.V.z));

  const float phi       = 2.0f * PI * random.x;
  const float s         = 1.0f + sqrtf(data.V.x * data.V.x + data.V.y * data.V.y);
  const float s2        = s * s;
  const float k         = (1.0f - roughness4) * s2 / (s2 + roughness4 * data.V.z * data.V.z);
  const float b         = k * v.z;
  const float z         = (1.0f - random.y) * (1.0f + b) - b;
  const float sin_theta = sqrtf(__saturatef(1.0f - z * z));
  const float x         = sin_theta * cosf(phi);
  const float y         = sin_theta * sinf(phi);
  const vec3 sampled    = add_vector(get_vector(x, y, z), v);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

__device__ float bsdf_microfacet_pdf(const GBufferData data, const float NdotH, const float NdotV) {
  // NdotV == data.V.z
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;

  const float D = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);

  const float len2 = roughness4 * (data.V.x * data.V.x + data.V.y * data.V.y);
  const float t    = sqrtf(len2 + data.V.z * data.V.z);

  const float s  = 1.0f + sqrtf(data.V.x * data.V.x + data.V.y * data.V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * data.V.z * data.V.z);
  return D / (2.0f * (k * NdotV + t));
}

__device__ vec3 bsdf_microfacet_sample(
  const GBufferData data, const ushort2 pixel, const uint32_t target = QUASI_RANDOM_TARGET_BSDF_MICROFACET,
  const uint32_t sequence_id = device.state.sample_id, const uint32_t depth = device.state.depth) {
  vec3 H = get_vector(0.0f, 0.0f, 1.0f);
  if (data.roughness > 0.0f) {
    const float2 random = quasirandom_sequence_2D_base_float(target, pixel, sequence_id, depth);
    H                   = bsdf_microfacet_sample_normal(data, random);
  }

  return H;
}

__device__ float bsdf_microfacet_evaluate(const GBufferData data, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // D * G2 * NdotL / (4.0f * NdotL * NdotV)
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * NdotL;
}

__device__ float bsdf_microfacet_evaluate_sampled_microfacet(const GBufferData data, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // NdotV == data.V.z

  const float len2 = roughness4 * (data.V.x * data.V.x + data.V.y * data.V.y);
  const float t    = sqrtf(len2 + data.V.z * data.V.z);

  const float s  = 1.0f + sqrtf(data.V.x * data.V.x + data.V.y * data.V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * data.V.z * data.V.z);

  // G2 contains (4 * NdotL * NdotV) in the denominator
  return 2.0f * (k * NdotV + t) * G2 * NdotL;
}

__device__ float bsdf_microfacet_evaluate_sampled_diffuse(const GBufferData data, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * PI;
}

/*
 * Samples a normal from the hemisphere with GGX VNDF for refractions. Method found in [DupB23].
 *
 * [DupB23] J. Dupuy and A. Benyoub, "Sampling Visible GGX Normals with Spherical Caps", 2023. arXiv:2306.05044
 */
__device__ vec3 bsdf_microfacet_refraction_sample_normal(const GBufferData data, const float2 random) {
  const float roughness2 = data.roughness * data.roughness;

  const vec3 v = normalize_vector(get_vector(roughness2 * data.V.x, roughness2 * data.V.y, data.V.z));

  const float phi       = 2.0f * PI * random.x;
  const float z         = (1.0f - random.y) * (1.0f + v.z) - v.z;
  const float sin_theta = sqrtf(__saturatef(1.0f - z * z));
  const float x         = sin_theta * cosf(phi);
  const float y         = sin_theta * sinf(phi);
  const vec3 sampled    = add_vector(get_vector(x, y, z), v);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

__device__ float bsdf_microfacet_refraction_pdf(
  const GBufferData data, const float NdotH, const float NdotV, const float NdotL, const float HdotV, const float HdotL,
  const float refraction_index) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G1         = bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV);

  float denominator = refraction_index * HdotV + HdotL;
  denominator       = denominator * denominator;

  // See Heitz14.
  return D * G1 * (HdotV / NdotV) * (HdotL / denominator);
}

__device__ vec3 bsdf_microfacet_refraction_sample(
  const GBufferData data, const ushort2 pixel, const uint32_t target = QUASI_RANDOM_TARGET_BSDF_REFRACTION,
  const uint32_t sequence_id = device.state.sample_id, const uint32_t depth = device.state.depth) {
  vec3 H = get_vector(0.0f, 0.0f, 1.0f);
  if (data.roughness > 0.0f) {
    const float2 random = quasirandom_sequence_2D_base_float(target, pixel, sequence_id, depth);
    H                   = bsdf_microfacet_refraction_sample_normal(data, random);
  }

  return H;
}

__device__ float bsdf_microfacet_refraction_evaluate(
  const GBufferData data, const float HdotL, const float HdotV, const float NdotH, const float NdotL, const float NdotV,
  const float refraction_index) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  float denominator = refraction_index * HdotV + HdotL;
  denominator       = denominator * denominator;

  // See Walter07.
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return 4.0f * NdotL * HdotV * HdotL * D * G2 / denominator;
}

__device__ float bsdf_microfacet_refraction_evaluate_sampled_microfacet(
  const GBufferData data, const float HdotL, const float HdotV, const float NdotH, const float NdotL, const float NdotV,
  const float refraction_index) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float G2_over_G1 = bsdf_microfacet_evaluate_smith_G2_over_G1_height_correlated_GGX(roughness4, NdotL, NdotV);

  // See Heitz14.
  return G2_over_G1;
}

///////////////////////////////////////////////////
// Diffuse
///////////////////////////////////////////////////

__device__ vec3 bsdf_diffuse_sample(const float2 random) {
  return sample_ray_sphere(random.x, random.y);
}

__device__ float bsdf_diffuse_pdf(const GBufferData data, const float NdotL) {
  return NdotL * (1.0f / PI);
}

__device__ float bsdf_diffuse_evaluate(const GBufferData data, const float NdotL) {
  return NdotL * (1.0f / PI);
}

__device__ float bsdf_diffuse_evaluate_sampled_diffuse() {
  return 1.0f;
}

__device__ float bsdf_diffuse_evaluate_sampled_microfacet(const GBufferData data, const float NdotL, const float NdotH, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4);

  const float len2 = roughness4 * (data.V.x * data.V.x + data.V.y * data.V.y);
  const float t    = sqrtf(len2 + data.V.z * data.V.z);

  const float s  = 1.0f + sqrtf(data.V.x * data.V.x + data.V.y * data.V.y);
  const float s2 = s * s;
  const float k  = (1.0f - roughness4) * s2 / (s2 + roughness4 * data.V.z * data.V.z);

  return NdotL * (2.0f * (k * NdotV + t)) / (PI * D);
}

///////////////////////////////////////////////////
// Material
///////////////////////////////////////////////////

__device__ float bsdf_conductor_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float>(device.bsdf_lut_conductor.handle, NdotV, roughness);
}

__device__ RGBF bsdf_conductor(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf,
  const uint8_t layer_mask) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return splat_color(0.0f);

  if ((data.flags & G_BUFFER_FLAG_BASE_SUBSTRATE_MASK) != G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE)
    return splat_color(0.0f);

  if ((data.flags & G_BUFFER_FLAG_METALLIC) == 0)
    return splat_color(0.0f);

  if ((layer_mask & (1 << MATERIAL_LAYER_TYPE_MICROFACET_REFLECTION)) == 0)
    return splat_color(0.0f);

  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = bsdf_microfacet_evaluate_sampled_diffuse(data, ctx.NdotH, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_MICROFACET_REFRACTION:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV)
                / bsdf_microfacet_refraction_pdf(data, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index);
      break;
  };

  const float directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, data.roughness);

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel_conductor, ss_term);
  const RGBF ms_term_with_fresnel =
    mul_color(ctx.f0_conductor, scale_color(ctx.fresnel_conductor, ((1.0f / directional_albedo) - 1.0f) * ss_term));

  return add_color(ss_term_with_fresnel, ms_term_with_fresnel);
}

__device__ float bsdf_glossy_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float>(device.bsdf_lut_glossy.handle, NdotV, roughness);
}

__device__ RGBF bsdf_glossy(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf,
  const uint8_t layer_mask) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  if ((data.flags & G_BUFFER_FLAG_BASE_SUBSTRATE_MASK) != G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE)
    return get_color(0.0f, 0.0f, 0.0f);

  if ((data.flags & G_BUFFER_FLAG_METALLIC) != 0)
    return get_color(0.0f, 0.0f, 0.0f);

  float ss_term = 0.0f;
  if ((layer_mask & (1 << MATERIAL_LAYER_TYPE_MICROFACET_REFLECTION)) != 0) {
    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_MICROFACET:
        ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotV);
        break;
      case BSDF_SAMPLING_DIFFUSE:
        ss_term = bsdf_microfacet_evaluate_sampled_diffuse(data, ctx.NdotH, ctx.NdotL, ctx.NdotV);
        break;
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV)
                  / bsdf_microfacet_refraction_pdf(data, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index);
        break;
    };
  }

  float diff_term = 0.0f;
  if ((layer_mask & (1 << MATERIAL_LAYER_TYPE_DIFFUSE)) != 0) {
    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        diff_term = bsdf_diffuse_evaluate(data, ctx.NdotL) * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_DIFFUSE:
        diff_term = 1.0f;
        break;
      case BSDF_SAMPLING_MICROFACET:
        diff_term = bsdf_diffuse_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotH, ctx.NdotV);
        break;
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        diff_term = bsdf_diffuse_evaluate(data, ctx.NdotL)
                    / bsdf_microfacet_refraction_pdf(data, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index);
        break;
    };
  }

  const float conductor_directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, data.roughness);
  const float glossy_directional_albedo    = bsdf_glossy_directional_albedo(ctx.NdotV, data.roughness);

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel_glossy, ss_term / conductor_directional_albedo);
  const RGBF diff_term_with_color = scale_color(opaque_color(data.albedo), diff_term * (1.0f - glossy_directional_albedo));

  return add_color(ss_term_with_fresnel, diff_term_with_color);
}

__device__ float bsdf_dielectric_directional_albedo(const float NdotV, const float roughness, const float ior) {
  const bool use_inv = (ior > 1.0f);

  const float ior_tex_coord = use_inv ? (ior - 1.0f) * 0.5f : (1.0f / ior - 1.0f) * 0.5f;

  const DeviceTextureHandle handle = use_inv ? device.bsdf_lut_dielectric_inv.handle : device.bsdf_lut_dielectric.handle;

  return tex3D<float>(handle, NdotV, roughness, ior_tex_coord);
}

__device__ RGBF bsdf_dielectric(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf,
  const uint8_t layer_mask) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return splat_color(0.0f);

  if ((data.flags & G_BUFFER_FLAG_BASE_SUBSTRATE_MASK) != G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT)
    return splat_color(0.0f);

  float term;
  if (ctx.is_refraction) {
    if ((layer_mask & (1 << MATERIAL_LAYER_TYPE_MICROFACET_REFRACTION)) == 0)
      return splat_color(0.0f);

    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        term = bsdf_microfacet_refraction_evaluate(data, ctx.HdotL, ctx.HdotV, ctx.NdotH, ctx.NdotL, ctx.NdotV, ctx.refraction_index)
               * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        term = bsdf_microfacet_refraction_evaluate_sampled_microfacet(
          data, ctx.HdotL, ctx.HdotV, ctx.NdotH, ctx.NdotL, ctx.NdotV, ctx.refraction_index);
        break;
      case BSDF_SAMPLING_MICROFACET:
      case BSDF_SAMPLING_DIFFUSE:
        term = 0.0f;
        break;
    };

    term *= (1.0f - ctx.fresnel_dielectric);
  }
  else {
    if ((layer_mask & (1 << MATERIAL_LAYER_TYPE_MICROFACET_REFLECTION)) == 0)
      return splat_color(0.0f);

    switch (sampling_hint) {
      case BSDF_SAMPLING_GENERAL:
        term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
        break;
      case BSDF_SAMPLING_MICROFACET:
        term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotV);
        break;
      case BSDF_SAMPLING_DIFFUSE:
        term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) / bsdf_diffuse_pdf(data, ctx.NdotL);
      case BSDF_SAMPLING_MICROFACET_REFRACTION:
        term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV)
               / bsdf_microfacet_refraction_pdf(data, ctx.NdotH, ctx.NdotV, ctx.NdotL, ctx.HdotV, ctx.HdotL, ctx.refraction_index);
        break;
    };

    term *= ctx.fresnel_dielectric;
  }

  const float dielectric_directional_albedo = bsdf_dielectric_directional_albedo(ctx.NdotV, data.roughness, ctx.refraction_index);
  term /= dielectric_directional_albedo;

  if (ctx.refraction_index == 1.0f && ctx.is_refraction) {
    // TODO: Energy conservation does not work correctly for dielectric, investigate.
    term = (sampling_hint == BSDF_SAMPLING_MICROFACET_REFRACTION) ? 1.0f : 0.0f;
  }

  return scale_color(opaque_color(data.albedo), term);
}

///////////////////////////////////////////////////
// Multiscattering
///////////////////////////////////////////////////

__device__ RGBF bsdf_multiscattering_evaluate(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf,
  const uint8_t layer_mask) {
  if (ctx.is_refraction)
    return scale_color(bsdf_dielectric(data, ctx, sampling_hint, one_over_sampling_pdf, layer_mask), data.albedo.a);

  RGBF conductor  = bsdf_conductor(data, ctx, sampling_hint, one_over_sampling_pdf, layer_mask);
  RGBF glossy     = bsdf_glossy(data, ctx, sampling_hint, one_over_sampling_pdf, layer_mask);
  RGBF dielectric = bsdf_dielectric(data, ctx, sampling_hint, one_over_sampling_pdf, layer_mask);

  return scale_color(add_color(add_color(conductor, glossy), dielectric), data.albedo.a);
}

#endif /* CU_BSDF_UTILS_H */
