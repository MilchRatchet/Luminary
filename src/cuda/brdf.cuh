#ifndef CU_BRDF_H
#define CU_BRDF_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

struct BRDFInstance {
  RGBAhalf albedo;
  RGBAhalf diffuse;
  RGBAhalf specular_f0;
  RGBAhalf fresnel;
  vec3 normal;
  float roughness;
  float metallic;
  RGBAhalf term;
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

/*
 * Computes a vector based on GGX distribution.
 * @param v Opposite of ray direction.
 * @param alpha Squared roughness.
 * @param random1 Uniform random number in [0,1).
 * @param random2 Uniform random number in [0,2 * PI).
 * @result Vector randomly sampled according to GGX distribution.
 */
__device__ vec3 brdf_sample_microfacet_GGX(const vec3 v, const float alpha, const float random1, const float random2) {
  if (alpha == 0.0f)
    return get_vector(0.0f, 0.0f, 1.0f);

  vec3 v_hemi;

  v_hemi.x = alpha * v.x;
  v_hemi.y = alpha * v.y;
  v_hemi.z = v.z;

  const float length_squared = v_hemi.x * v_hemi.x + v_hemi.y * v_hemi.y;
  vec3 T1;

  if (length_squared == 0.0f) {
    T1.x = 1.0f;
    T1.y = 0.0f;
    T1.z = 0.0f;
  }
  else {
    const float length = rsqrtf(length_squared);
    T1.x               = -v_hemi.y * length;
    T1.y               = v_hemi.x * length;
    T1.z               = 0.0f;
  }

  const vec3 T2 = cross_product(v_hemi, T1);

  const float r   = sqrtf(random1);
  const float phi = random2;
  const float t1  = r * cosf(phi);
  const float s   = 0.5f * (1.0f + v_hemi.z);
  const float t2  = lerp(sqrtf(1.0f - t1 * t1), r * sinf(phi), s);

  vec3 normal_hemi;

  const float scalar = sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
  normal_hemi.x      = alpha * (t1 * T1.x + t2 * T2.x + scalar * v_hemi.x);
  normal_hemi.y      = alpha * (t1 * T1.y + t2 * T2.y + scalar * v_hemi.y);
  normal_hemi.z      = fmaxf(0.0f, t1 * T1.z + t2 * T2.z + scalar * v_hemi.z);

  return normalize_vector(normal_hemi);
}

__device__ vec3 brdf_sample_light_ray(const LightSample light, const vec3 origin) {
  switch (light.id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN:
      return sample_sphere(device.sun_pos, SKY_SUN_RADIUS, world_to_sky_transform(origin));
    case LIGHT_ID_TOY:
      return sample_sphere(device.scene.toy.position, device.scene.toy.scale, origin);
    default:
      const TriangleLight triangle = load_triangle_light(light.id);
      return sample_triangle(triangle, origin);
  }
}

__device__ float brdf_light_sample_target_weight(const LightSample light) {
  float weight;
  switch (light.id) {
    case LIGHT_ID_SUN:
      weight = 2e+04f * device.scene.sky.sun_strength;
      break;
    case LIGHT_ID_TOY:
      weight = device.scene.toy.material.b;
      break;
    case LIGHT_ID_NONE:
      weight = 0.0f;
      break;
    default: {
      weight = device.scene.material.default_material.b;
    } break;
  }

  return weight * light.solid_angle;
}

__device__ float brdf_light_sample_solid_angle(const LightSample light, const vec3 pos) {
  switch (light.id) {
    case LIGHT_ID_SUN:
      return 0.5f * ONE_OVER_PI * sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, world_to_sky_transform(pos));
    case LIGHT_ID_TOY:
      return 0.5f * ONE_OVER_PI * sample_sphere_solid_angle(device.scene.toy.position, device.scene.toy.scale, pos);
    case LIGHT_ID_NONE:
      return 0.0f;
    default:
      const TriangleLight triangle = load_triangle_light(light.id);
      return 0.5f * ONE_OVER_PI * sample_triangle_solid_angle(triangle, pos);
  }
}

__device__ float brdf_light_sample_shading_weight(const LightSample light) {
  return (1.0f / brdf_light_sample_target_weight(light)) * (light.weight / light.M);
}

__device__ LightSample brdf_light_sample_update(LightSample x, const LightSample y, const float update_weight, const float r) {
  x.weight += update_weight;
  x.M += y.M;
  if (r < (update_weight / x.weight)) {
    x.id          = y.id;
    x.solid_angle = y.solid_angle;
  }

  return x;
}

__device__ LightSample sample_light(const vec3 position, uint32_t& ran_offset) {
  const vec3 sky_pos = world_to_sky_transform(position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);
  uint32_t light_count  = 0;
  light_count += sun_visible;
  light_count += toy_visible;
  light_count += (device.scene.material.lights_active) ? device.scene.triangle_lights_length : 0;

  LightSample selected;
  selected.id          = LIGHT_ID_NONE;
  selected.M           = 0;
  selected.solid_angle = 0.0f;
  selected.weight      = 0.0f;

  if (!light_count)
    return selected;

  const float ran = white_noise_offset(ran_offset++);

  if (device.iteration_type != TYPE_CAMERA && sun_visible && ran < 0.5f) {
    selected.id          = LIGHT_ID_SUN;
    selected.M           = 1;
    selected.solid_angle = brdf_light_sample_solid_angle(selected, position);
    selected.weight      = 2.0f * brdf_light_sample_target_weight(selected);

    return selected;
  }

  const int reservoir_sampling_size = min(light_count, device.reservoir_size);

  const float light_count_float = ((float) light_count) - 1.0f + 0.9999999f;

  for (int i = 0; i < reservoir_sampling_size; i++) {
    const float r1       = white_noise_precise_offset(ran_offset++);
    uint32_t light_index = (uint32_t) (r1 * light_count_float);

    light_index += !sun_visible;
    light_index += (!toy_visible && light_index) ? 1 : 0;

    if (light_index >= device.scene.triangle_lights_length + 2)
      continue;

    LightSample light;
    light.id = LIGHT_ID_NONE;
    light.M  = 1;

    switch (light_index) {
      case 0:
        light.id = LIGHT_ID_SUN;
        break;
      case 1:
        light.id = LIGHT_ID_TOY;
        break;
      default: {
        const TriangleLight triangle = load_triangle_light(light_index - 2);
        light.id                     = light_index - 2;
      } break;
    }

    if (light.id == selected.id)
      continue;

    light.solid_angle = brdf_light_sample_solid_angle(light, position);
    light.weight      = brdf_light_sample_target_weight(light) * light_count_float;

    const float r = white_noise_offset(ran_offset++);

    selected = brdf_light_sample_update(selected, light, light.weight, r);
  }

  return selected;
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

  brdf.term = mul_RGBAhalf(brdf.term, F);

  brdf.L = L_local;

  return brdf;
}

__device__ vec3 brdf_sample_ray_hemisphere(const float alpha, const float beta) {
  const float a = sqrtf(alpha);
  return get_vector(a * cosf(beta), a * sinf(beta), sqrtf(1.0f - alpha));
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

__device__ BRDFInstance
  brdf_get_instance(const RGBAhalf albedo, const vec3 V, const vec3 normal, const float roughness, const float metallic) {
  BRDFInstance brdf;
  brdf.albedo      = albedo;
  brdf.diffuse     = brdf_albedo_as_diffuse(albedo, metallic);
  brdf.specular_f0 = brdf_albedo_as_specular_f0(albedo, metallic);
  brdf.V           = V;
  brdf.roughness   = roughness;
  brdf.metallic    = metallic;
  brdf.normal      = normal;
  brdf.term        = get_RGBAhalf(1.0f, 1.0f, 1.0f, 0.0f);

  return brdf;
}

/*
 * This computes the BRDF weight of a light sample.
 * Writes term of the BRDFInstance.
 */
__device__ BRDFInstance brdf_evaluate(BRDFInstance brdf) {
  const vec3 H = normalize_vector(add_vector(brdf.V, brdf.L));

  float NdotL = dot_product(brdf.normal, brdf.L);
  float NdotV = dot_product(brdf.normal, brdf.V);

  if (NdotL <= 0.0f || NdotV <= 0.0f) {
    brdf.term = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
    return brdf;
  }

  NdotL = fminf(fmaxf(0.0001f, NdotL), 1.0f);
  NdotV = fminf(fmaxf(0.0001f, NdotV), 1.0f);

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

  brdf.term = mul_RGBAhalf(brdf.term, specular);

  return brdf;
}

/*
 * Samples a ray based on the BRDFs and multiplies record with sampling weight.
 * Writes L and term of the BRDFInstance.
 */
__device__ BRDFInstance brdf_sample_ray(BRDFInstance brdf, const ushort2 index) {
  const float specular_prob = brdf_spec_probability(brdf.metallic);
  const int use_specular    = white_noise() < specular_prob;
  const float alpha         = white_noise();
  const float beta          = 2.0f * PI * white_noise();

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
  const float alpha = r1;
  const float beta  = r2;

  const float roughness2 = brdf.roughness * brdf.roughness;

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(brdf.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(brdf.V, rotation_to_z);

  vec3 H_local = get_vector(0.0f, 0.0f, 1.0f);

  if (roughness2 > 0.0f) {
    H_local = brdf_sample_microfacet(V_local, roughness2, alpha, beta);
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

  brdf.term = mul_RGBAhalf(brdf.term, F);

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
