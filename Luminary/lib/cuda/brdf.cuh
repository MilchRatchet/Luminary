#ifndef CU_BRDF_H
#define CU_BRDF_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

/*
 * This BRDF implementation is based on the BRDF Crashcourse by Jakub Boksansky.
 * Multiscattering compensation is based on the blog "Image Based Lighting with Multiple Scattering" by Bruno Opsenica and the paper
 * "A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting" by Fdez-Aguera.
 */

/*
 * There are two diffuse BRDFs implemented.
 *  - Lambertian
 *  - Frostbite-Disney
 *
 * The frostbite-disney BRDF does not seem to work very well in combination with the specular BRDF.
 * The image tends to become a lot darker than it should.
 * The lambertian BRDF works a lot better, however, even that one has some energy loss when combined with the specular BRDF.
 *
 * The specular BRDF is a microfacet GGX distributed BRDF. Since it only takes single scattering into consideration
 * there is a multiscattering compensation, however, while this improves energy conservation, there is still some
 * energy lost.
 */

/*
 * There are two fresnel approximations. One is the standard Schlick approximation. The other is some
 * approximation found in the paper by Fdez-Aguera.
 */
__device__ float brdf_shadowed_F90(const RGBF f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(f0));
}

__device__ RGBF brdf_albedo_as_specular_f0(const RGBF albedo, const float metallic) {
  RGBF specular_f0;
  specular_f0.r = lerp(0.04f, albedo.r, metallic);
  specular_f0.g = lerp(0.04f, albedo.g, metallic);
  specular_f0.b = lerp(0.04f, albedo.b, metallic);
  return specular_f0;
}

__device__ RGBF brdf_albedo_as_diffuse(const RGBF albedo, const float metallic) {
  return scale_color(albedo, 1.0f - metallic);
}

/*
 * Standard Schlick Fresnel approximation.
 * @param f0 Specular F0.
 * @param f90 Shadow term.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
__device__ RGBF brdf_fresnel_schlick(const RGBF f0, const float f90, const float NdotV) {
  RGBF result;

  const float t = powf(1.0f - NdotV, 5.0f);

  result.r = lerp(f0.r, f90, t);
  result.g = lerp(f0.g, f90, t);
  result.b = lerp(f0.b, f90, t);

  return result;
}

/*
 * Fresnel approximation as found in the paper by Fdez-Aguera
 * @param f0 Specular F0.
 * @param roughness Material roughness.
 * @param NdotV Cosine Angle.
 * @result Fresnel approximation.
 */
__device__ RGBF brdf_fresnel_roughness(const RGBF f0, const float roughness, const float NdotV) {
  const float t = powf(1.0f - NdotV, 5.0f);

  const RGBF Fr = sub_color(max_color(get_color(1.0f - roughness, 1.0f - roughness, 1.0f - roughness), f0), f0);
  return add_color(f0, scale_color(Fr, t));
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
    const float length = 1.0f / sqrtf(length_squared);
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

__device__ vec3 brdf_sample_light_ray(const RestirSample light, const vec3 origin) {
  switch (light.id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN:
      return sample_sphere(device_sun, SKY_SUN_RADIUS, world_to_sky_transform(origin));
    case LIGHT_ID_TOY:
      return sample_sphere(device_scene.toy.position, device_scene.toy.scale, origin);
    default:
      const TraversalTriangle triangle = device_scene.traversal_triangles[light.id];
      return sample_triangle(triangle, origin);
  }
}

__device__ RestirSample brdf_finalize_restir_sample(RestirSample light, const vec3 origin, const vec3 normal) {
  light.weight = 1.0f / light.weight;
  switch (light.id) {
    case LIGHT_ID_SUN:
      light.weight *= sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, world_to_sky_transform(origin), normal);
      break;
    case LIGHT_ID_TOY:
      light.weight *= sample_sphere_solid_angle(device_scene.toy.position, device_scene.toy.scale, origin, normal);
      break;
    case LIGHT_ID_NONE:
      light.weight = 0.0f;
      break;
    default:
      const TraversalTriangle triangle = device_scene.traversal_triangles[light.id];
      light.weight *= sample_triangle_solid_angle(triangle, origin, normal);
      break;
  }

  return light;
}

__device__ RestirSample sample_light(const vec3 position, const vec3 normal) {
  const vec3 sky_pos = world_to_sky_transform(position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device_sun, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device_scene.toy.active && device_scene.toy.emissive);
  uint32_t light_count  = 0;
  light_count += sun_visible;
  light_count += toy_visible;
  light_count += (device_scene.material.lights_active) ? device_scene.lights_ids_length - 2 : 0;

  float weight_sum = 0.0f;
  RestirSample selected;
  selected.id     = LIGHT_ID_NONE;
  selected.weight = 0.0f;

  if (!light_count)
    return selected;

  const int reservoir_sampling_size = min(light_count, device_reservoir_size);

  const float light_count_float = ((float) light_count) - 1.0f + 0.9999999f;

  for (int i = 0; i < reservoir_sampling_size; i++) {
    const float r1       = white_noise();
    uint32_t light_index = (uint32_t) (r1 * light_count_float);

    light_index += !sun_visible;
    light_index += (!toy_visible && light_index) ? 1 : 0;

    if (light_index >= device_scene.lights_ids_length)
      continue;

    const uint32_t light_id = __ldg(device_scene.lights_ids + light_index);

    float weight;
    float solid_angle;
    switch (light_id) {
      case LIGHT_ID_SUN:
        solid_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, sky_pos, normal);
        weight      = device_scene.sky.sun_strength;
        break;
      case LIGHT_ID_TOY:
        solid_angle = sample_sphere_solid_angle(device_scene.toy.position, device_scene.toy.scale, position, normal);
        weight      = device_scene.toy.material.b;
        break;
      default: {
        const TraversalTriangle triangle = device_scene.traversal_triangles[light_id];

        solid_angle = sample_triangle_solid_angle(triangle, position, normal);
        weight      = device_scene.material.default_material.b;
      } break;
    }

    RestirSample light;
    light.id     = light_id;
    light.weight = weight * solid_angle;

    weight_sum += light.weight;

    const float r2 = white_noise();

    if (r2 < light.weight / weight_sum) {
      selected = light;
    }
  }

  selected.weight = (selected.weight * reservoir_sampling_size) / (light_count * weight_sum);

  return selected;
}

__device__ float brdf_diffuse_term_frostbyte_disney(const float roughness, const float HdotV, const float NdotL, const float NdotV) {
  const float energyBias   = 0.5f * roughness;
  const float energyFactor = lerp(1.0f, 1.0f / 1.51f, roughness);

  const float FD90MinusOne = energyBias + 2.0f * HdotV * HdotV * roughness - 1.0f;

  const float FDL = 1.0f + (FD90MinusOne * pow(1.0f - NdotL, 5.0f));
  const float FDV = 1.0f + (FD90MinusOne * pow(1.0f - NdotV, 5.0f));

  return FDL * FDV * energyFactor;
}

__device__ float brdf_diffuse_term(const float roughness, const float HdotV, const float NdotL, const float NdotV) {
  switch (device_scene.material.diffuse) {
    case FROSTBITEDISNEY:
      return brdf_diffuse_term_frostbyte_disney(roughness, HdotV, NdotL, NdotV);
    case LAMBERTIAN:
    default:
      return 1.0f;
  }
}

__device__ vec3 brdf_sample_ray_hemisphere(float alpha, const float beta, const vec3 normal) {
  return normalize_vector(sample_ray_from_angles_and_vector(sqrtf(alpha), beta, normal));
}

__device__ vec3 brdf_sample_microfacet(const vec3 V_local, const float roughness2, const float alpha, const float beta) {
  return brdf_sample_microfacet_GGX(V_local, roughness2, alpha, beta);
}

/*
 * Multiscattering compensation for microfacet model similar to in the paper by Fdez-Aguera.
 * I scale the multiscattering term with the roughness because there cannot be any multiscattering in perfectly smooth surfaces.
 * In fact the issue came up in light sampling. The multiscattering assumes Ess + Ems = 1 for perfect reflectors.
 * However, this is obviously only true for physically feasible in-out directions. However light sampling generates
 * infeasible directions for which there should really be Ess + Ems = 0.
 */
__device__ RGBF brdf_microfacet_multiscattering(
  const float roughness2, const float NdotL, const float NdotV, const RGBF fresnel, const RGBF specular_f0, const float brdf_term) {
  const RGBF FssEss = scale_color(fresnel, brdf_term);

  const float Ems = (1.0f - brdf_term);
  const RGBF F_avg =
    add_color(specular_f0, get_color((1.0f - specular_f0.r) / 21.0f, (1.0f - specular_f0.g) / 21.0f, (1.0f - specular_f0.b) / 21.0f));
  const RGBF FmsEms = get_color(
    Ems * FssEss.r * F_avg.r / (1.0f - F_avg.r * Ems), Ems * FssEss.g * F_avg.g / (1.0f - F_avg.g * Ems),
    Ems * FssEss.b * F_avg.b / (1.0f - F_avg.b * Ems));
  return add_color(FssEss, scale_color(FmsEms, roughness2));
}

__device__ vec3
  brdf_sample_ray_microfacet(RGBF& record, const vec3 V_local, const float roughness2, RGBF specular_f0, float alpha, float beta) {
  vec3 H_local = get_vector(0.0f, 0.0f, 1.0f);

  if (roughness2 > 0.0f) {
    H_local = brdf_sample_microfacet(V_local, roughness2, alpha, beta);
  }

  const vec3 L_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

  const float HdotL = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, L_local)));
  const float NdotL = fmaxf(0.00001f, fminf(1.0f, L_local.z));
  const float NdotV = fmaxf(0.00001f, fminf(1.0f, V_local.z));

  RGBF fresnel;
  switch (device_scene.material.fresnel) {
    case SCHLICK:
      fresnel = brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), HdotL);
      break;
    case FDEZ_AGUERA:
    default:
      fresnel = brdf_fresnel_roughness(specular_f0, sqrtf(roughness2), HdotL);
      break;
  }

  const float brdf_term = brdf_smith_G2_over_G1_height_correlated(roughness2 * roughness2, NdotL, NdotV);

  record = mul_color(record, brdf_microfacet_multiscattering(roughness2, NdotL, NdotV, fresnel, specular_f0, brdf_term));

  return L_local;
}

__device__ vec3
  brdf_sample_ray_specular(RGBF& record, const vec3 V_local, const float roughness2, RGBF specular_f0, float alpha, float beta) {
  return brdf_sample_ray_microfacet(record, V_local, roughness2, specular_f0, alpha, beta);
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

/*
 * Samples a ray based on the BRDFs and multiplies record with sampling weight.
 * @result Returns true if the sample is valid. (atm, it always returns true)
 */
__device__ bool brdf_sample_ray(
  vec3& ray, RGBF& record, const ushort2 index, const uint32_t state, const RGBF albedo, const vec3 V, const vec3 normal,
  const vec3 face_normal, const float roughness, const float metallic) {
  const float specular_prob = brdf_spec_probability(metallic);
  const int use_specular    = blue_noise(index.x, index.y, state, 10) < specular_prob;
  const float alpha         = blue_noise(index.x, index.y, state, 2);
  const float beta          = 2.0f * PI * blue_noise(index.x, index.y, state, 3);

  const RGBF specular_f0 = brdf_albedo_as_specular_f0(albedo, metallic);

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);
  const vec3 V_local             = rotate_vector_by_quaternion(V, rotation_to_z);

  vec3 L_local;

  if (use_specular) {
    L_local = brdf_sample_ray_specular(record, V_local, roughness * roughness, specular_f0, alpha, beta);
    record  = scale_color(record, 1.0f / specular_prob);
  }
  else {
    L_local = brdf_sample_ray_diffuse(alpha, beta);

    const vec3 H_specular = brdf_sample_microfacet(V_local, roughness * roughness, alpha, beta);
    const float VdotH     = fmaxf(0.00001f, fminf(1.0f, dot_product(V_local, H_specular)));
    const RGBF fresnel    = brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), VdotH);

    const RGBF diffuse_reflectance = brdf_albedo_as_diffuse(albedo, metallic);

    record = mul_color(
      record, scale_color(
                mul_color(diffuse_reflectance, get_color(1.0f - fresnel.r, 1.0f - fresnel.g, 1.0f - fresnel.b)),
                brdf_diffuse_term(roughness, VdotH, L_local.z, V_local.z) / (1.0f - specular_prob)));
  }

  ray = normalize_vector(rotate_vector_by_quaternion(L_local, inverse_quaternion(rotation_to_z)));

  return true;
}

__device__ float brdf_evaluate_microfacet_GGX(const float roughness4, const float NdotH) {
  const float a = ((roughness4 - 1.0f) * NdotH * NdotH + 1.0f);
  return roughness4 / (PI * a * a);
}

__device__ RGBF brdf_evaluate_microfacet(
  const float roughness, const float NdotH, const float NdotL, const float NdotV, const RGBF fresnel, const RGBF specular_f0) {
  const float roughness4 = fmaxf(0.00001f, roughness * roughness * roughness * roughness);
  const float D          = brdf_evaluate_microfacet_GGX(roughness4, NdotH);
  const float G2         = brdf_smith_G2_height_correlated_GGX(roughness4, NdotL, NdotV);

  return brdf_microfacet_multiscattering(roughness * roughness, NdotL, NdotV, fresnel, specular_f0, D * G2);
}

// This is supposed to be an interface in the case we want to allow for different BRDFs in the future
__device__ RGBF brdf_evaluate_specular(
  const float roughness, const float NdotH, const float NdotL, const float NdotV, const RGBF fresnel, const RGBF specular_f0) {
  return brdf_evaluate_microfacet(roughness, NdotH, NdotL, NdotV, fresnel, specular_f0);
}

__device__ RGBF brdf_evaluate_lambertian(const RGBF albedo, const float NdotL) {
  return scale_color(albedo, ONE_OVER_PI * NdotL);
}

__device__ RGBF
  brdf_evaluate_frostbyte_disney(const RGBF albedo, const float NdotL, const float NdotV, const float HdotV, const float roughness) {
  return scale_color(albedo, brdf_diffuse_term_frostbyte_disney(roughness, HdotV, NdotL, NdotV) * ONE_OVER_PI * NdotL);
}

__device__ RGBF brdf_evaluate_diffuse(const RGBF albedo, const float NdotL, const float NdotV, const float HdotV, const float roughness) {
  switch (device_scene.material.diffuse) {
    case FROSTBITEDISNEY:
      return brdf_evaluate_frostbyte_disney(albedo, NdotL, NdotV, HdotV, roughness);
    case LAMBERTIAN:
    default:
      return brdf_evaluate_lambertian(albedo, NdotL);
  }
}

/*
 * This computes the BRDF weight of a light sample.
 * @result Multiplicative weight.
 */
__device__ RGBF
  brdf_evaluate(const RGBF albedo, const vec3 V, const vec3 ray, const vec3 normal, const float roughness, const float metallic) {
  const vec3 H = normalize_vector(add_vector(V, ray));

  float NdotL = dot_product(normal, ray);
  float NdotV = dot_product(normal, V);

  if (NdotL <= 0.0f || NdotV <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  NdotL = fminf(fmaxf(0.00001f, NdotL), 1.0f);
  NdotV = fminf(fmaxf(0.00001f, NdotV), 1.0f);

  const float NdotH = __saturatef(dot_product(normal, H));
  const float HdotV = __saturatef(dot_product(H, V));

  const RGBF specular_f0    = brdf_albedo_as_specular_f0(albedo, metallic);
  const RGBF diffuse_albedo = brdf_albedo_as_diffuse(albedo, metallic);

  RGBF fresnel;
  switch (device_scene.material.fresnel) {
    case SCHLICK:
      fresnel = brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), HdotV);
      break;
    case FDEZ_AGUERA:
    default:
      fresnel = brdf_fresnel_roughness(specular_f0, roughness, HdotV);
      break;
  }

  const RGBF specular = brdf_evaluate_specular(roughness, NdotH, NdotL, NdotV, fresnel, specular_f0);
  const RGBF diffuse  = brdf_evaluate_diffuse(diffuse_albedo, NdotL, NdotV, HdotV, roughness);

  return add_color(specular, mul_color(diffuse, get_color(1.0f - fresnel.r, 1.0f - fresnel.g, 1.0f - fresnel.b)));
}

/*
 * This is not correct but it looks right.
 *
 * Essentially I handle refraction like specular reflection. I sample a microfacet and
 * simply refract along it instead of reflecting. Weighting is done exactly like in reflection.
 * The albedo is simply taken as the specular f0, I don't know I feel like that makes sense
 * from an artistic standpoint. This is most likely completely non physical but since
 * refraction plays such a small role at the moment, it probably doesnt matter too much.
 */
__device__ vec3 brdf_sample_ray_refraction(
  RGBF& record, const RGBF albedo, const vec3 normal, const vec3 ray, const float roughness, const float index, const float r1,
  const float r2) {
  const float alpha = r1;
  const float beta  = r2;

  const float roughness2 = roughness * roughness;

  const RGBF specular_f0 = albedo;

  const vec3 V = scale_vector(ray, -1.0f);

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);
  const vec3 V_local             = rotate_vector_by_quaternion(V, rotation_to_z);

  vec3 H_local = get_vector(0.0f, 0.0f, 1.0f);

  if (roughness2 > 0.0f) {
    H_local = brdf_sample_microfacet(V_local, roughness2, alpha, beta);
  }

  vec3 L_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

  const float HdotL = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, L_local)));
  const float HdotV = fmaxf(0.00001f, fminf(1.0f, dot_product(H_local, V_local)));
  const float NdotL = fmaxf(0.00001f, fminf(1.0f, L_local.z));
  const float NdotV = fmaxf(0.00001f, fminf(1.0f, V_local.z));

  RGBF fresnel;
  switch (device_scene.material.fresnel) {
    case SCHLICK:
      fresnel = brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), HdotL);
      break;
    case FDEZ_AGUERA:
    default:
      fresnel = brdf_fresnel_roughness(specular_f0, sqrtf(roughness2), HdotL);
      break;
  }

  const float brdf_term = brdf_smith_G2_over_G1_height_correlated(roughness2 * roughness2, NdotL, NdotV);

  record = mul_color(record, brdf_microfacet_multiscattering(roughness2, NdotL, NdotV, fresnel, specular_f0, brdf_term));

  const float b = 1.0f - index * index * (1.0f - HdotV * HdotV);

  const vec3 ray_local = scale_vector(V_local, -1.0f);

  if (b < 0.0f) {
    L_local = normalize_vector(reflect_vector(ray_local, scale_vector(H_local, -1.0f)));
  }
  else {
    L_local = normalize_vector(add_vector(scale_vector(ray_local, index), scale_vector(H_local, index * HdotV - sqrtf(b))));
  }

  return normalize_vector(rotate_vector_by_quaternion(L_local, inverse_quaternion(rotation_to_z)));
}

#endif /* CU_BRDF_H */
