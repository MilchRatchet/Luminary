#ifndef CU_BSDF_UTILS_H
#define CU_BSDF_UTILS_H

#include "math.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

struct BSDFRayContext {
  vec3 H;
  RGBF f0_conductor;
  RGBF f0_glossy;
  RGBF f0_dielectric;
  RGBF fresnel_conductor;
  RGBF fresnel_glossy;
  RGBF fresnel_dielectric;
  float NdotH;
  float NdotL;
  float NdotV;
  bool is_refraction;
};

struct BSDFDirectionalAlbedos {
  float conductor;
  float glossy;
  RGBF dielectric;
  RGBF dielectric_inv;
} typedef BSDFDirectionalAlbedos;

struct BSDFSampleInfo {
  RGBF weight;
  bool is_transparent_pass;
  bool is_microfacet_based;
} typedef BSDFSampleInfo;

enum BSDFLUT { BSDF_LUT_SS = 0, BSDF_LUT_SPECULAR = 1, BSDF_LUT_DIELEC = 2, BSDF_LUT_DIELEC_INV = 3 } typedef BSDFLUT;

enum BSDFSamplingHint { BSDF_SAMPLING_GENERAL = 0, BSDF_SAMPLING_MICROFACET = 1, BSDF_SAMPLING_DIFFUSE = 2 };

__device__ RGBF bsdf_diffuse_color(const GBufferData data) {
  return scale_color(opaque_color(data.albedo), 1.0f - data.metallic);
}

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

__device__ RGBF bsdf_fresnel_normal_incidence(const GBufferData data, const float ior_in, const float ior_out) {
  const RGBF standard_f0 = get_color(0.04f, 0.04f, 0.04f);
  const RGBF diff        = sub_color(opaque_color(data.albedo), standard_f0);

  const RGBF specular_f0 = fma_color(diff, data.metallic, standard_f0);

  const float fresnel_f0 =
    bsdf_fresnel(get_vector(0.0f, 0.0f, 1.0f), get_vector(0.0f, 0.0f, 1.0f), get_vector(0.0f, 0.0f, -1.0f), ior_in, ior_out);

  return fma_color(
    sub_color(specular_f0, get_color(fresnel_f0, fresnel_f0, fresnel_f0)), data.albedo.a, get_color(fresnel_f0, fresnel_f0, fresnel_f0));
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

/*
 * Fresnel approximation as found in the paper by Fdez-Aguera
 * @param f0 Specular F0.
 * @param roughness Material roughness.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
__device__ RGBF bsdf_fresnel_roughness(const RGBF f0, const float roughness, const float HdotV) {
  const float one_minus_HdotV = 1.0f - HdotV;
  const float pow2            = one_minus_HdotV * one_minus_HdotV;

  // powf(1.0f - NdotV, 5.0f)
  const float t = pow2 * pow2 * one_minus_HdotV;

  const float s = 1.0f - roughness;
  const RGBF Fr = sub_color(max_color(get_color(s, s, s), f0), f0);

  return fma_color(Fr, t, f0);
}

__device__ float bsdf_shadowed_F90(const RGBF specular_f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(specular_f0));
}

__device__ RGBF bsdf_fresnel_composite(
  const GBufferData data, const vec3 refraction, const float ior_in, const float ior_out, const RGBF f0, const float HdotV) {
  float fresnel_ior = bsdf_fresnel(data.normal, data.V, refraction, ior_in, ior_out);

  RGBF fresnel_approx = bsdf_fresnel_schlick(f0, bsdf_shadowed_F90(f0), HdotV);  // bsdf_fresnel_roughness(f0, data.roughness, HdotV);
  fresnel_approx      = scale_color(fresnel_approx, data.albedo.a);
  fresnel_approx      = add_color(fresnel_approx, scale_color(get_color(fresnel_ior, fresnel_ior, fresnel_ior), 1.0f - data.albedo.a));

  return fresnel_approx;
}

///////////////////////////////////////////////////
// Refraction
///////////////////////////////////////////////////

__device__ float bsdf_refraction_index_ambient(const GBufferData data) {
  if (device.scene.toy.active && toy_is_inside(data.position))
    return device.scene.toy.refractive_index;

  if (device.scene.ocean.active && data.position.y < device.scene.ocean.height)
    return device.scene.ocean.refractive_index;

  return 1.0f;
}

__device__ float bsdf_refraction_index(const GBufferData data) {
  const float ambient_index_of_refraction = bsdf_refraction_index_ambient(data);

  const float refraction_index = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index / ambient_index_of_refraction
                                                                              : ambient_index_of_refraction / data.refraction_index;

  return refraction_index;
}

// Get normal vector based on incoming ray and refracted ray: https://physics.stackexchange.com/a/762982
__device__ vec3 bsdf_refraction_normal_from_pair(const vec3 L, const vec3 V, const float ior_L, const float ior_V) {
  return normalize_vector(add_vector(scale_vector(V, ior_V), scale_vector(L, ior_L)));
}

///////////////////////////////////////////////////
// Microfacet
///////////////////////////////////////////////////

// S can be either L or V, doesn't matter.
__device__ float bsdf_microfacet_evaluate_smith_G1_GGX(const float roughness4, const float NdotS) {
  const float NdotS2 = __saturatef(fmaxf(0.0001f, NdotS * NdotS));
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

__device__ float bsdf_microfacet_evaluate(const GBufferData data, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(roughness4, NdotH);
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // D * G2 * NdotL / (4.0f * NdotL * NdotV)
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * NdotL;
}

__device__ float bsdf_microfacet_evaluate_sampled_microfacet(const GBufferData data, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float G2_over_G1 = bsdf_microfacet_evaluate_smith_G2_over_G1_height_correlated_GGX(roughness4, NdotL, NdotV);

  // G2 / G1
  return G2_over_G1;
}

__device__ float bsdf_microfacet_evaluate_sampled_diffuse(const GBufferData data, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  const float D          = bsdf_microfacet_evaluate_D_GGX(roughness4, NdotH);
  const float G2         = bsdf_microfacet_evaluate_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  // D * G2 * NdotL * PI / (4.0f * NdotL * NdotV * NdotL)
  // G2 contains (4 * NdotL * NdotV) in the denominator
  return D * G2 * PI;
}

/*
 * Samples a normal from the hemisphere with GGX VNDF. Method found in [DupB23].
 *
 * [DupB23] J. Dupuy and A. Benyoub, "Sampling Visible GGX Normals with Spherical Caps", 2023. arXiv:2306.05044
 */
__device__ vec3 bsdf_microfacet_sample_normal_GGX(const vec3 v, const float2 random) {
  const float phi       = 2.0f * PI * random.x;
  const float z         = (1.0f - random.y) * (1.0f + v.z) - v.z;
  const float sin_theta = sqrtf(__saturatef(1.0f - z * z));
  const float x         = sin_theta * cosf(phi);
  const float y         = sin_theta * sinf(phi);
  const vec3 result     = add_vector(get_vector(x, y, z), v);

  return result;
}

__device__ vec3 bsdf_microfacet_sample_normal(const GBufferData data, const float2 random) {
  const float roughness2 = data.roughness * data.roughness;

  vec3 v_hemi = normalize_vector(get_vector(roughness2 * data.V.x, roughness2 * data.V.y, data.V.z));

  vec3 sampled = bsdf_microfacet_sample_normal_GGX(v_hemi, random);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

__device__ float bsdf_microfacet_pdf(const GBufferData data, const vec3 H) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;

  const float NdotH = __saturatef(dot_product(data.normal, H));
  const float NdotV = __saturatef(dot_product(data.normal, data.V));

  return bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4) * bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV) / (4.0f * NdotV);
}

__device__ vec3 bsdf_microfacet_sample(
  const GBufferData data, const ushort2 pixel, vec3& H, const uint32_t sequence_id = device.temporal_frames,
  const uint32_t depth = device.depth) {
  H = get_vector(0.0f, 0.0f, 1.0f);
  if (data.roughness > 0.0f) {
    const float2 random = quasirandom_sequence_2D_base(QUASI_RANDOM_TARGET_BSDF_REFLECTION, pixel, sequence_id, depth);
    H                   = bsdf_microfacet_sample_normal(data, random);
  }

  return reflect_vector(scale_vector(data.V, -1.0f), H);
}

__device__ float bsdf_microfacet_pdf_reflection(const GBufferData data, const vec3 L) {
  const vec3 H = normalize_vector(add_vector(data.V, L));

  return bsdf_microfacet_pdf(data, H);
}

__device__ float bsdf_microfacet_pdf_refraction(const GBufferData data, const vec3 L) {
  const float ambient_ior = bsdf_refraction_index_ambient(data);
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;

  const vec3 H = bsdf_refraction_normal_from_pair(L, data.V, ior_out, ior_in);

  return bsdf_microfacet_pdf(data, H);
}

///////////////////////////////////////////////////
// Diffuse
///////////////////////////////////////////////////

__device__ vec3 bsdf_diffuse_sample(const float2 random) {
  return sample_ray_sphere(random.x, random.y);
}

__device__ float bsdf_diffuse_evaluate(const GBufferData data, const BSDFRayContext ctx) {
  return ctx.NdotL * (1.0f / PI);
}

__device__ float bsdf_diffuse_pdf(const GBufferData data, const vec3 L) {
  const float NdotL = dot_product(data.normal, L);
  return NdotL * (1.0f / PI);
}

///////////////////////////////////////////////////
// Material
///////////////////////////////////////////////////

__device__ float bsdf_conductor_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float4>(device.ptrs.bsdf_energy_lut[BSDF_LUT_SS].tex, NdotV, roughness).x;
}

__device__ RGBF bsdf_conductor(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = 0.0f;
      break;
  };

  const float directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, data.roughness);

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel_conductor, ss_term);
  const RGBF ms_term_with_fresnel =
    mul_color(ctx.f0_conductor, scale_color(ctx.fresnel_conductor, ((1.0f / directional_albedo) - 1.0f) * ss_term));

  return add_color(ss_term_with_fresnel, ms_term_with_fresnel);
}

__device__ float bsdf_glossy_directional_albedo(const float NdotV, const float roughness) {
  return tex2D<float4>(device.ptrs.bsdf_energy_lut[BSDF_LUT_SPECULAR].tex, NdotV, roughness).x;
}

__device__ RGBF
  bsdf_glossy(const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.NdotL <= 0.0f || ctx.NdotV <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = 0.0f;
      break;
  };

  float diff_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      diff_term = bsdf_diffuse_evaluate(data, ctx) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET: {
      diff_term = 0.0f;
    } break;
    case BSDF_SAMPLING_DIFFUSE:
      diff_term = 1.0f;
      break;
  };

  const float conductor_directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, data.roughness);
  const float glossy_directional_albedo    = bsdf_glossy_directional_albedo(ctx.NdotV, data.roughness);

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel_glossy, ss_term / conductor_directional_albedo);
  const RGBF diff_term_with_color = scale_color(opaque_color(data.albedo), diff_term * (1.0f - glossy_directional_albedo));

  return add_color(ss_term_with_fresnel, diff_term_with_color);
}

///////////////////////////////////////////////////
// Multiscattering
///////////////////////////////////////////////////

__device__ RGBF bsdf_multiscattering_evaluate(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.is_refraction)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF conductor = scale_color(bsdf_conductor(data, ctx, sampling_hint, one_over_sampling_pdf), data.metallic);
  RGBF glossy    = scale_color(bsdf_glossy(data, ctx, sampling_hint, one_over_sampling_pdf), 1.0f - data.metallic);

  return add_color(conductor, glossy);
}

#endif /* CU_BSDF_UTILS_H */
