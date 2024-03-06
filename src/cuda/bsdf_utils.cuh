#ifndef CU_BSDF_UTILS_H
#define CU_BSDF_UTILS_H

#include "math.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

struct BSDFRayContext {
  vec3 H;
  RGBF f0;
  RGBF fresnel;
  RGBF diffuse;
  RGBF refraction;
  float NdotH;
  float NdotL;
  float NdotV;
  bool is_refraction;
};

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
  const float one_minus_HdotV = 1.0f - HdotV;
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

__device__ float bsdf_microfacet_evaluate_sampled_microfacet(
  const GBufferData data, const float NdotH, const float NdotL, const float NdotV) {
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

__device__ vec3 bsdf_microfacet_sample_normal(const GBufferData data, const vec3 V_local, const float2 random) {
  const float roughness2 = data.roughness * data.roughness;

  vec3 v_hemi = normalize_vector(get_vector(roughness2 * V_local.x, roughness2 * V_local.y, V_local.z));

  vec3 sampled = bsdf_microfacet_sample_normal_GGX(v_hemi, random);

  return normalize_vector(get_vector(sampled.x * roughness2, sampled.y * roughness2, sampled.z));
}

__device__ vec3 bsdf_microfacet_sample(const GBufferData data, const vec3 V_local, const ushort2 pixel, vec3& H, float& sampling_weight) {
  float weight                 = 0.0f;
  const uint32_t total_samples = 10;
  H                            = get_vector(0.0f, 0.0f, 1.0f);
  vec3 chosen_ray;

  for (uint32_t i = 0; i < total_samples; i++) {
    vec3 microfacet = get_vector(0.0f, 0.0f, 1.0f);
    if (data.roughness > 0.0f) {
      const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_REFLECTION + i, pixel);
      microfacet          = bsdf_microfacet_sample_normal(data, V_local, random);
    }

    vec3 sampled_ray = reflect_vector(scale_vector(V_local, -1.0f), microfacet);

    if (sampled_ray.z > 0.0f) {
      weight += 1.0f;

      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_CHOICE + i, pixel) * weight < 1.0f) {
        chosen_ray = sampled_ray;
        H          = microfacet;
      }
    }
  }

  sampling_weight = weight / total_samples;

  return chosen_ray;
}

__device__ float bsdf_microfacet_pdf(const GBufferData data, const vec3 H) {
  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;

  const float NdotH = __saturatef(dot_product(data.normal, H));
  const float NdotV = __saturatef(dot_product(data.normal, data.V));

  return bsdf_microfacet_evaluate_D_GGX(NdotH, roughness4) * bsdf_microfacet_evaluate_smith_G1_GGX(roughness4, NdotV) / (4.0f * NdotV);
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

__constant__ float bsdf_conductor_coeffs[20] = {1.0247217f, -10.984229f, 10.918318f,  46.93353f,  -54.779343f, 21.742077f, -30.368898f,
                                                31.919222f, -8.013965f,  -6.2407165f, 1.0f,       -10.218104f, 10.955399f, 44.08196f,
                                                -55.33452f, 21.437538f,  -23.744568f, 33.265057f, -7.9268975f, -5.930959f};

__device__ float bsdf_conductor_directional_albedo(const float NdotV, const float roughness) {
  const float u = NdotV;
  const float r = roughness;

  const float u2  = u * u;
  const float r2  = r * r;
  const float ur  = u * r;
  const float r3  = r2 * r;
  const float u3  = u2 * u;
  const float r2u = r2 * u;
  const float u2r = u2 * r;

  const float num = bsdf_conductor_coeffs[0] + bsdf_conductor_coeffs[1] * r + bsdf_conductor_coeffs[2] * u + bsdf_conductor_coeffs[3] * r2
                    + bsdf_conductor_coeffs[4] * ur + bsdf_conductor_coeffs[5] * u2 + bsdf_conductor_coeffs[6] * r3
                    + bsdf_conductor_coeffs[7] * r2u + bsdf_conductor_coeffs[8] * u2r + bsdf_conductor_coeffs[9] * u3;

  const float denom = bsdf_conductor_coeffs[10] + bsdf_conductor_coeffs[11] * r + bsdf_conductor_coeffs[12] * u
                      + bsdf_conductor_coeffs[13] * r2 + bsdf_conductor_coeffs[14] * ur + bsdf_conductor_coeffs[15] * u2
                      + bsdf_conductor_coeffs[16] * r3 + bsdf_conductor_coeffs[17] * r2u + bsdf_conductor_coeffs[18] * u2r
                      + bsdf_conductor_coeffs[19] * u3;

  return __saturatef(num / denom);
}

__device__ RGBF bsdf_conductor(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotH, ctx.NdotL, ctx.NdotV);
      break;
    case BSDF_SAMPLING_DIFFUSE:
      ss_term = bsdf_microfacet_evaluate_sampled_diffuse(data, ctx.NdotH, ctx.NdotL, ctx.NdotV);
      break;
  };

  const float conductor_directional_albedo = bsdf_conductor_directional_albedo(ctx.NdotV, data.roughness);

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel, ss_term);
  const RGBF ms_term_with_fresnel = mul_color(ctx.f0, scale_color(ctx.fresnel, ((1.0f / conductor_directional_albedo) - 1.0f) * ss_term));

  return add_color(ss_term_with_fresnel, ms_term_with_fresnel);
}

__constant__ float bsdf_glossy_coeffs[40] = {
  0.04301317f, 132.98329f, -0.9273584f, -0.61434704f, -262.23462f, -137.75214f, -234.72151f, 5.125822f,  -0.37465897f, 9.284745f,
  129.71187f,  171.82188f, 400.04813f,  206.99231f,   1.0847985f,  428.02484f,  -2.2108653f, -6.056363f, 0.95864034f,  -11.775469f,
  1.0f,        139.43494f, -24.177433f, -3.7300687f,  -253.77824f, 6.717145f,   98.03935f,   153.19194f, -184.53282f,  230.02286f,
  113.9376f,   66.64211f,  108.315094f, 23.577564f,   120.04127f,  102.90899f,  17.030241f,  25.947954f, 75.77901f,    49.348934f};

__device__ float bsdf_glossy_directional_albedo(const float NdotV, const float roughness, const float f0) {
  const float u = NdotV;
  const float r = roughness;
  const float f = f0;

  const float u2  = u * u;
  const float r2  = r * r;
  const float f2  = f * f;
  const float ur  = u * r;
  const float uf  = u * f;
  const float rf  = r * f;
  const float r3  = r2 * r;
  const float u3  = u2 * u;
  const float f3  = f2 * f;
  const float urf = ur * f;
  const float r2f = r2 * f;
  const float u2f = u2 * f;
  const float f2r = f2 * r;
  const float f2u = f2 * u;
  const float r2u = r2 * u;
  const float u2r = u2 * r;

  const float num = bsdf_glossy_coeffs[0] + bsdf_glossy_coeffs[1] * f + bsdf_glossy_coeffs[2] * r + bsdf_glossy_coeffs[3] * u
                    + bsdf_glossy_coeffs[4] * f2 + bsdf_glossy_coeffs[5] * rf + bsdf_glossy_coeffs[6] * uf + bsdf_glossy_coeffs[7] * r2
                    + bsdf_glossy_coeffs[8] * ur + bsdf_glossy_coeffs[9] * u2 + bsdf_glossy_coeffs[10] * f3 + bsdf_glossy_coeffs[11] * f2r
                    + bsdf_glossy_coeffs[12] * f2u + bsdf_glossy_coeffs[13] * r2f + bsdf_glossy_coeffs[14] * urf
                    + bsdf_glossy_coeffs[15] * u2f + bsdf_glossy_coeffs[16] * r3 + bsdf_glossy_coeffs[17] * r2u
                    + bsdf_glossy_coeffs[18] * u2r * bsdf_glossy_coeffs[19] * u3;

  const float denom =
    bsdf_glossy_coeffs[20] + bsdf_glossy_coeffs[21] * f + bsdf_glossy_coeffs[22] * r + bsdf_glossy_coeffs[23] * u
    + bsdf_glossy_coeffs[24] * f2 + bsdf_glossy_coeffs[25] * rf + bsdf_glossy_coeffs[26] * uf + bsdf_glossy_coeffs[27] * r2
    + bsdf_glossy_coeffs[28] * ur + bsdf_glossy_coeffs[29] * u2 + bsdf_glossy_coeffs[30] * f3 + bsdf_glossy_coeffs[31] * f2r
    + bsdf_glossy_coeffs[32] * f2u + bsdf_glossy_coeffs[33] * r2f + bsdf_glossy_coeffs[34] * urf + bsdf_glossy_coeffs[35] * u2f
    + bsdf_glossy_coeffs[36] * r3 + bsdf_glossy_coeffs[37] * r2u + bsdf_glossy_coeffs[38] * u2r * bsdf_glossy_coeffs[39] * u3;

  return num / denom;
}

__device__ RGBF
  bsdf_glossy(const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  float ss_term;
  switch (sampling_hint) {
    case BSDF_SAMPLING_GENERAL:
      ss_term = bsdf_microfacet_evaluate(data, ctx.NdotH, ctx.NdotL, ctx.NdotV) * one_over_sampling_pdf;
      break;
    case BSDF_SAMPLING_MICROFACET:
      ss_term = bsdf_microfacet_evaluate_sampled_microfacet(data, ctx.NdotH, ctx.NdotL, ctx.NdotV);
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
  const float glossy_directional_albedo    = bsdf_glossy_directional_albedo(ctx.NdotV, data.roughness, luminance(ctx.f0));

  const RGBF ss_term_with_fresnel = scale_color(ctx.fresnel, ss_term / conductor_directional_albedo);
  const RGBF diff_term_with_color = scale_color(ctx.diffuse, diff_term /** (1.0f - glossy_directional_albedo)*/);

  return add_color(ss_term_with_fresnel, diff_term_with_color);
}

///////////////////////////////////////////////////
// Multiscattering
///////////////////////////////////////////////////

__device__ RGBF bsdf_multiscattering_evaluate(
  const GBufferData data, const BSDFRayContext ctx, const BSDFSamplingHint sampling_hint, const float one_over_sampling_pdf) {
  if (ctx.is_refraction)
    return get_color(0.0f, 0.0f, 0.0f);

  if (data.metallic == 0.0f)
    return bsdf_glossy(data, ctx, sampling_hint, one_over_sampling_pdf);

  return bsdf_conductor(data, ctx, sampling_hint, one_over_sampling_pdf);
}

#endif /* CU_BSDF_UTILS_H */
