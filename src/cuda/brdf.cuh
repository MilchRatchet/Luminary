#ifndef CU_BRDF_H
#define CU_BRDF_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "sky_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

struct BRDFInstance {
  RGBAhalf albedo;
  RGBAhalf diffuse;
  RGBAhalf specular_f0;
  RGBAhalf fresnel;
  vec3 normal;
  float roughness;
  float metallic;
  RGBF term;
  vec3 V;
  vec3 L;
} typedef BRDFInstance;

/*
 * This BRDF implementation is based on the BRDF Crashcourse by Jakub Boksansky and
 * "A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting" by Fdez-Aguera.
 *
 * Essentially instead of using separate BRDFs for diffuse and specular components, the BRDF by Fdez-Aguera
 * provides both diffuse and specular component based on the microfacet model which achieves perfect energy conservation.
 * However, only metallic values of 0 and 1 provide energy conservation as all values in between have energy loss.
 */

/*
 * There are two fresnel approximations. One is the standard Schlick approximation. The other is some
 * approximation found in the paper by Fdez-Aguera.
 */
__device__ float brdf_shadowed_F90(const RGBAhalf specular_f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(RGBAhalf_to_RGBF(specular_f0)));
}

__device__ RGBAhalf brdf_albedo_as_specular_f0(const RGBAhalf albedo, const float metallic) {
  const RGBAhalf specular_f0 = get_RGBAhalf(0.04f, 0.04f, 0.04f, 0.00f);
  const RGBAhalf diff        = sub_RGBAhalf(sqrt_RGBAhalf(albedo), specular_f0);

  return fma_RGBAhalf(diff, metallic, specular_f0);
}

__device__ RGBAhalf brdf_albedo_as_diffuse(const RGBAhalf albedo, const float metallic) {
  return scale_RGBAhalf(albedo, 1.0f - metallic);
}

/*
 * Standard Schlick Fresnel approximation.
 * @param f0 Specular F0.
 * @param f90 Shadow term.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
__device__ RGBAhalf brdf_fresnel_schlick(const RGBAhalf f0, const float f90, const float NdotV) {
  const float t = powf(1.0f - NdotV, 5.0f);

  RGBAhalf result = f0;
  RGBAhalf diff   = sub_RGBAhalf(get_RGBAhalf(f90, f90, f90, 0.0f), f0);
  result          = fma_RGBAhalf(diff, t, result);

  return result;
}

/*
 * Fresnel approximation as found in the paper by Fdez-Aguera
 * @param f0 Specular F0.
 * @param roughness Material roughness.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
__device__ RGBAhalf brdf_fresnel_roughness(const RGBAhalf f0, const float roughness, const float NdotV) {
  const float t     = powf(1.0f - NdotV, 5.0f);
  const __half s    = 1.0f - roughness;
  const RGBAhalf Fr = sub_RGBAhalf(max_RGBAhalf(get_RGBAhalf(s, s, s, 0.0f), f0), f0);

  return fma_RGBAhalf(Fr, t, f0);
}

__device__ float brdf_smith_G1_GGX(const float roughness4, const float NdotS2) {
  return 2.0f / (sqrtf(((roughness4 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__ float brdf_smith_G2_over_G1_height_correlated(const float roughness4, const float NdotL, const float NdotV) {
  const float G1V = brdf_smith_G1_GGX(roughness4, NdotV * NdotV);
  const float G1L = brdf_smith_G1_GGX(roughness4, NdotL * NdotL);
  return G1L / (G1V + G1L - G1V * G1L);
}

__device__ float brdf_smith_G2_height_correlated_GGX(const float roughness4, const float NdotL, const float NdotV) {
  const float a = NdotV * sqrtf(roughness4 + NdotL * (NdotL - roughness4 * NdotL));
  const float b = NdotL * sqrtf(roughness4 + NdotV * (NdotV - roughness4 * NdotV));
  return 0.5f / (a + b);
}

__device__ vec3 brdf_sample_microfacet_GGX_hemisphere(const vec3 v, const float r1, const float r2) {
  const float tmp = v.x * v.x + v.y * v.y;

  vec3 w1 = (tmp > 0.0f) ? scale_vector(get_vector(-v.y, v.x, 0.0f), rsqrtf(tmp)) : get_vector(1.0f, 0.0f, 0.0f);
  vec3 w2 = cross_product(v, w1);

  float phi = 2.0f * PI * r1;
  float r   = sqrtf(r2);

  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);

  float s  = (1.0f + v.z) * 0.5f;
  t2       = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
  float ti = sqrtf(fmaxf(1.0f - t1 * t1 - t2 * t2, 0.0f));

  vec3 result = scale_vector(v, ti);
  result      = add_vector(result, scale_vector(w1, t1));
  result      = add_vector(result, scale_vector(w2, t2));

  return result;
}

/*
 * Computes a vector based on GGX distribution.
 * @param v Opposite of ray direction.
 * @param alpha Squared roughness.
 * @param random1 Uniform random number in [0,1).
 * @param random2 Uniform random number in [0,1).
 * @result Vector randomly sampled according to GGX distribution.
 */
__device__ vec3 brdf_sample_microfacet_GGX(const vec3 v, const float alpha, const float random1, const float random2) {
  vec3 v_hemi = normalize_vector(get_vector(alpha * v.x, alpha * v.y, v.z));

  vec3 sampled = brdf_sample_microfacet_GGX_hemisphere(v_hemi, random1, random2);

  return normalize_vector(get_vector(sampled.x * alpha, sampled.y * alpha, sampled.z));
}

__device__ vec3 brdf_sample_microfacet(const vec3 V_local, const float roughness2, const float alpha, const float beta) {
  return brdf_sample_microfacet_GGX(V_local, roughness2, alpha, beta);
}

/*
 * Multiscattering microfacet model by Fdez-Aguera.
 */
__device__ RGBAhalf brdf_microfacet_multiscattering(
  const float NdotV, const RGBAhalf fresnel, const RGBF specular_f0, const RGBAhalf diffuse, const float brdf_term) {
  const RGBF FssEss = scale_color(RGBAhalf_to_RGBF(fresnel), brdf_term);

  const float Ems = (1.0f - brdf_term);

  const RGBF F_avg =
    add_color(specular_f0, get_color((1.0f - specular_f0.r) / 21.0f, (1.0f - specular_f0.g) / 21.0f, (1.0f - specular_f0.b) / 21.0f));

  const RGBF FmsEms = get_color(F_avg.r / (1.0f - F_avg.r * Ems), F_avg.g / (1.0f - F_avg.g * Ems), F_avg.b / (1.0f - F_avg.b * Ems));

  const RGBAhalf SSMS = RGBF_to_RGBAhalf(add_color(FssEss, scale_color(mul_color(FssEss, FmsEms), Ems)));

  const RGBAhalf Edss = sub_RGBAhalf(get_RGBAhalf(1.0f, 1.0f, 1.0f, 0.0f), SSMS);

  const RGBAhalf Kd = mul_RGBAhalf(diffuse, Edss);

  return add_RGBAhalf(Kd, SSMS);
}

__device__ BRDFInstance brdf_sample_ray_microfacet(BRDFInstance brdf, const vec3 V_local, float alpha, float beta) {
  const float roughness2 = brdf.roughness * brdf.roughness;
  vec3 H_local           = get_vector(0.0f, 0.0f, 1.0f);

  if (roughness2 > 0.0f) {
    H_local = brdf_sample_microfacet(V_local, roughness2, alpha, beta);
  }

  const vec3 L_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

  const float HdotL = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, L_local)));
  const float NdotL = fmaxf(0.00001f, fminf(1.0f, L_local.z));
  const float NdotV = fmaxf(0.00001f, fminf(1.0f, V_local.z));

  switch (device.scene.material.fresnel) {
    case SCHLICK:
      brdf.fresnel = brdf_fresnel_schlick(brdf.specular_f0, brdf_shadowed_F90(brdf.specular_f0), HdotL);
      break;
    case FDEZ_AGUERA:
    default:
      brdf.fresnel = brdf_fresnel_roughness(brdf.specular_f0, brdf.roughness, HdotL);
      break;
  }

  const float brdf_term = brdf_smith_G2_over_G1_height_correlated(roughness2 * roughness2, NdotL, NdotV);

  const RGBAhalf F = brdf_microfacet_multiscattering(NdotV, brdf.fresnel, RGBAhalf_to_RGBF(brdf.specular_f0), brdf.diffuse, brdf_term);

  brdf.term = mul_color(brdf.term, RGBAhalf_to_RGBF(F));

  brdf.L = L_local;

  return brdf;
}

__device__ vec3 brdf_sample_ray_hemisphere(const float alpha, const float beta) {
  const float a = sqrtf(alpha);
  const float b = 2.0f * PI * beta;

  // How this works:
  // Standard way is a = acosf(alpha) and then return (sinf(a) * cosf(b), sinf(a) * sinf(b), cosf(a)).
  // What we can do instead is sample a point uniformly on the disk on which the hemisphere lies. (i.e. z = 0.0f).
  // Then since we want a normalized direction the ray must satisfy 1.0f = sqrtf(x * x + y * y + z * z).
  // Further, since we only want the upper hemisphere it must be that z >= 0.0f.
  // Using these constraints, we get that z = sqrtf(1.0f - x * x + y * y).
  // Then we can use that x * x + y * y = a * a * (cosf(b) * cosf(b) + sinf(b) * sinf(b)) = a * a.
  // Hence, we have that z = sqrtf(1.0f - a * a) = sqrtf(1.0f - sqrtf(alpha) * sqrtf(alpha)) = sqrtf(1.0f - alpha).
  return get_vector(a * cosf(b), a * sinf(b), sqrtf(1.0f - alpha));
}

__device__ vec3 brdf_sample_ray_diffuse(const float alpha, const float beta) {
  return brdf_sample_ray_hemisphere(alpha, beta);
}

__device__ float brdf_spec_probability(const float metallic) {
  return lerp(0.5f, 1.0f, metallic);
}

__device__ float brdf_evaluate_microfacet_GGX(const float roughness4, const float NdotH) {
  const float a = ((roughness4 - 1.0f) * NdotH * NdotH + 1.0f);
  return roughness4 / (PI * a * a);
}

__device__ RGBAhalf brdf_evaluate_microfacet(BRDFInstance brdf, const float NdotH, const float NdotL, const float NdotV) {
  const float roughness4 = brdf.roughness * brdf.roughness * brdf.roughness * brdf.roughness;
  const float D          = brdf_evaluate_microfacet_GGX(roughness4, NdotH);
  const float G2         = brdf_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  return brdf_microfacet_multiscattering(NdotV, brdf.fresnel, RGBAhalf_to_RGBF(brdf.specular_f0), brdf.diffuse, D * G2 * NdotL);
}

__device__ BRDFInstance brdf_get_instance(RGBAhalf albedo, const vec3 V, const vec3 normal, const float roughness, const float metallic) {
  // An albedo of all 1 produces incorrect results
  albedo = min_RGBAhalf(albedo, get_RGBAhalf(1.0f - eps, 1.0f - eps, 1.0f - eps, 1.0f));

  BRDFInstance brdf;
  brdf.albedo      = albedo;
  brdf.diffuse     = brdf_albedo_as_diffuse(albedo, metallic);
  brdf.specular_f0 = brdf_albedo_as_specular_f0(albedo, metallic);
  brdf.V           = V;
  brdf.roughness   = roughness;
  brdf.metallic    = metallic;
  brdf.normal      = normal;
  brdf.term        = get_color(1.0f, 1.0f, 1.0f);

  return brdf;
}

__device__ BRDFInstance brdf_get_instance_scattering() {
  BRDFInstance brdf;
  brdf.albedo      = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
  brdf.diffuse     = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
  brdf.specular_f0 = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
  brdf.V           = get_vector(0.0f, 0.0f, 0.0f);
  brdf.roughness   = 0.0f;
  brdf.metallic    = 0.0f;
  brdf.normal      = get_vector(0.0f, 0.0f, 0.0f);
  brdf.term        = get_color(1.0f, 1.0f, 1.0f);

  return brdf;
}

/*
 * This computes the BRDF weight of a light sample.
 * Writes term of the BRDFInstance.
 */
__device__ BRDFInstance brdf_evaluate(BRDFInstance brdf) {
  float NdotL = dot_product(brdf.normal, brdf.L);
  float NdotV = dot_product(brdf.normal, brdf.V);

  if (NdotL <= 0.0f || NdotV <= 0.0f) {
    brdf.term = get_color(0.0f, 0.0f, 0.0f);
    return brdf;
  }

  const vec3 H = normalize_vector(add_vector(brdf.V, brdf.L));

  const float NdotH = __saturatef(dot_product(brdf.normal, H));
  const float HdotV = __saturatef(dot_product(H, brdf.V));

  switch (device.scene.material.fresnel) {
    case SCHLICK:
      brdf.fresnel = brdf_fresnel_schlick(brdf.specular_f0, brdf_shadowed_F90(brdf.specular_f0), HdotV);
      break;
    case FDEZ_AGUERA:
    default:
      brdf.fresnel = brdf_fresnel_roughness(brdf.specular_f0, brdf.roughness, HdotV);
      break;
  }

  RGBAhalf specular = brdf_evaluate_microfacet(brdf, NdotH, NdotL, NdotV);

  brdf.term = mul_color(brdf.term, RGBAhalf_to_RGBF(specular));

  return brdf;
}

__device__ BRDFInstance brdf_apply_sample(BRDFInstance brdf, LightSample light, vec3 pos) {
  BRDFInstance result = brdf;

  switch (light.presampled_id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN: {
      vec3 sky_pos = world_to_sky_transform(pos);
      result.L     = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, light.seed);
      result.term  = scale_color(result.term, 0.5f * ONE_OVER_PI * sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, sky_pos));
    } break;
    case LIGHT_ID_TOY:
      result.L    = toy_sample_ray(pos, light.seed);
      result.term = scale_color(result.term, 0.5f * ONE_OVER_PI * toy_get_solid_angle(pos));
      break;
    default: {
      const TriangleLight triangle = load_triangle_light(device.restir.presampled_triangle_lights, light.presampled_id);
      result.L                     = sample_triangle(triangle, pos, light.seed);
      result.term                  = scale_color(result.term, 0.5f * ONE_OVER_PI * sample_triangle_solid_angle(triangle, pos));
    } break;
  }

  result.term = scale_color(result.term, light.weight);

  return brdf_evaluate(result);
}

__device__ BRDFInstance brdf_apply_sample_scattering(BRDFInstance brdf, LightSample light, vec3 pos, float anisotropy) {
  BRDFInstance result = brdf_get_instance_scattering();

  switch (light.presampled_id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN: {
      vec3 sky_pos = world_to_sky_transform(pos);
      result.L     = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, light.seed);
      result.term  = scale_color(result.term, 0.25f * ONE_OVER_PI * sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, sky_pos));
    } break;
    case LIGHT_ID_TOY:
      result.L    = toy_sample_ray(pos, light.seed);
      result.term = scale_color(result.term, 0.25f * ONE_OVER_PI * toy_get_solid_angle(pos));
      break;
    default: {
      const TriangleLight triangle = load_triangle_light(device.restir.presampled_triangle_lights, light.presampled_id);
      result.L                     = sample_triangle(triangle, pos, light.seed);
      result.term                  = scale_color(result.term, 0.25f * ONE_OVER_PI * sample_triangle_solid_angle(triangle, pos));
    } break;
  }

  result.term = scale_color(result.term, light.weight);

  const float cos_angle = dot_product(scale_vector(brdf.V, -1.0f), result.L);
  const float phase     = henvey_greenstein(cos_angle, anisotropy);

  return result;
}

/*
 * Samples a ray based on the BRDFs and multiplies record with sampling weight.
 * Writes L and term of the BRDFInstance.
 */
__device__ BRDFInstance brdf_sample_ray(BRDFInstance brdf, const ushort2 index) {
  const float specular_prob = brdf_spec_probability(brdf.metallic);
  const int use_specular    = white_noise() < specular_prob;
  const float alpha         = white_noise();
  const float beta          = white_noise();

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(brdf.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(brdf.V, rotation_to_z);

  if (use_specular) {
    brdf = brdf_sample_ray_microfacet(brdf, V_local, alpha, beta);

    brdf.L = normalize_vector(rotate_vector_by_quaternion(brdf.L, inverse_quaternion(rotation_to_z)));
  }
  else {
    const vec3 L_local = brdf_sample_ray_diffuse(alpha, beta);
    brdf.L             = normalize_vector(rotate_vector_by_quaternion(L_local, inverse_quaternion(rotation_to_z)));
    brdf               = brdf_evaluate(brdf);
  }

  return brdf;
}

/*
 * This is not correct but it looks right.
 *
 * Essentially I handle refraction like specular reflection. I sample a microfacet and
 * simply refract along it instead of reflecting. Weighting is done exactly like in reflection.
 * This is most likely completely non physical but since refraction plays such a small role at the moment,
 * it probably doesnt matter too much.
 */
__device__ BRDFInstance brdf_sample_ray_refraction(BRDFInstance brdf, const float index, const float r1, const float r2) {
  const float roughness2 = brdf.roughness * brdf.roughness;

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(brdf.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(brdf.V, rotation_to_z);

  vec3 H_local = get_vector(0.0f, 0.0f, 1.0f);

  if (roughness2 > 0.0f) {
    H_local = brdf_sample_microfacet(V_local, roughness2, r1, r2);
  }

  vec3 L_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

  const float HdotL = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, L_local)));
  const float HdotV = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, V_local)));
  const float NdotL = fmaxf(0.00001f, fminf(1.0f, L_local.z));
  const float NdotV = fmaxf(0.00001f, fminf(1.0f, V_local.z));

  switch (device.scene.material.fresnel) {
    case SCHLICK:
      brdf.fresnel = brdf_fresnel_schlick(brdf.specular_f0, brdf_shadowed_F90(brdf.specular_f0), HdotL);
      break;
    case FDEZ_AGUERA:
    default:
      brdf.fresnel = brdf_fresnel_roughness(brdf.specular_f0, brdf.roughness, HdotL);
      break;
  }

  const float brdf_term = brdf_smith_G2_over_G1_height_correlated(roughness2 * roughness2, NdotL, NdotV);

  const RGBAhalf F = brdf_microfacet_multiscattering(NdotV, brdf.fresnel, RGBAhalf_to_RGBF(brdf.specular_f0), brdf.diffuse, brdf_term);

  brdf.term = mul_color(brdf.term, RGBAhalf_to_RGBF(F));

  const float b = 1.0f - index * index * (1.0f - HdotV * HdotV);

  const vec3 ray_local = scale_vector(V_local, -1.0f);

  if (b < 0.0f) {
    L_local = normalize_vector(reflect_vector(ray_local, scale_vector(H_local, -1.0f)));
  }
  else {
    L_local = normalize_vector(add_vector(scale_vector(ray_local, index), scale_vector(H_local, index * HdotV - sqrtf(b))));
  }

  brdf.L = normalize_vector(rotate_vector_by_quaternion(L_local, inverse_quaternion(rotation_to_z)));

  return brdf;
}

#endif /* CU_BRDF_H */
