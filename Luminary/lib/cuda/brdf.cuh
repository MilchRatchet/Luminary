#ifndef CU_BRDF_H
#define CU_BRDF_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

struct LightSample {
  vec3 dir;
  float angle;
  uint32_t id;
  float weight;
} typedef LightSample;

__device__ float shadowed_F90(const RGBF f0) {
  const float t = 1.0f / 0.04f;
  return fminf(1.0f, t * luminance(f0));
}

__device__ RGBF Fresnel_Schlick(const RGBF f0, const float f90, const float NdotS) {
  RGBF result;

  const float t = powf(1.0f - NdotS, 5.0f);

  result.r = lerp(f0.r, f90, t);
  result.g = lerp(f0.g, f90, t);
  result.b = lerp(f0.b, f90, t);

  return result;
}

__device__ float Smith_G1_GGX(const float alpha2, const float NdotS2) {
  return 2.0f / (sqrtf(((alpha2 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__ float Smith_G2_over_G1_height_correlated(const float alpha2, const float NdotL, const float NdotV) {
  const float G1V = Smith_G1_GGX(alpha2, NdotV * NdotV);
  const float G1L = Smith_G1_GGX(alpha2, NdotL * NdotL);
  return G1L / (G1V + G1L - G1V * G1L);
}

__device__ vec3 sample_GGX_VNDF(const vec3 v, const float alpha, const float random1, const float random2) {
  if (alpha < eps * eps)
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

__device__ vec3 specular_BRDF(
  RGBF& record, const vec3 normal, const vec3 V, const RGBAF albedo, const float roughness, const float metallic, const float beta,
  const float gamma, const float spec_prob) {
  const float alpha              = roughness * roughness;
  const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);

  float weight = 1.0f;

  const vec3 V_local = rotate_vector_by_quaternion(V, rotation_to_z);

  vec3 H_local = sample_GGX_VNDF(V_local, alpha, beta, gamma);

  const vec3 ray_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

  const float HdotR = __saturatef(dot_product(H_local, ray_local));
  const float NdotR = fmaxf(eps, fminf(1.0f, ray_local.z));
  const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));

  vec3 ray = normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));

  RGBF specular_f0;
  specular_f0.r = lerp(0.04f, albedo.r, metallic);
  specular_f0.g = lerp(0.04f, albedo.g, metallic);
  specular_f0.b = lerp(0.04f, albedo.b, metallic);

  const RGBF F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotR);

  const float milchs_energy_recovery = lerp(1.0f, 1.51f + 1.51f * NdotV, roughness);

  weight *= milchs_energy_recovery * Smith_G2_over_G1_height_correlated(alpha * alpha, NdotR, NdotV) / spec_prob;

  record.r *= F.r * weight;
  record.g *= F.g * weight;
  record.b *= F.b * weight;

  return ray;
}

__device__ vec3 diffuse_BRDF(
  RGBF& record, const vec3 normal, const vec3 V, const RGBAF albedo, const float roughness, const float metallic, const float beta,
  const float gamma, const float spec_prob) {
  const float alpha = sqrtf(beta);

  vec3 ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

  vec3 H;
  H.x = V.x + ray.x;
  H.y = V.y + ray.y;
  H.z = V.z + ray.z;
  H   = normalize_vector(H);

  const float half_angle   = __saturatef(dot_product(H, ray));
  const float energyFactor = lerp(1.0f, 1.0f / 1.51f, roughness);

  const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

  const float angle          = __saturatef(dot_product(normal, ray));
  const float previous_angle = __saturatef(dot_product(V, normal));

  const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
  const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

  float weight = FDL * FDV * energyFactor * (1.0f - metallic) / (1.0f - spec_prob);

  record.r *= albedo.r * weight;
  record.g *= albedo.g * weight;
  record.b *= albedo.b * weight;

  return ray;
}

__device__ vec3 light_BRDF(
  RGBF& record, const vec3 normal, const vec3 V, const LightSample light, const RGBAF albedo, const float roughness, const float metallic,
  const float beta, const float gamma) {
  const float alpha = sqrtf(beta);

  vec3 ray = normalize_vector(sample_ray_from_angles_and_vector(alpha * light.angle, gamma, light.dir));

  float weight = light.weight;

  vec3 H;
  H.x = V.x + ray.x;
  H.y = V.y + ray.y;
  H.z = V.z + ray.z;
  H   = normalize_vector(H);

  const float half_angle   = __saturatef(dot_product(H, ray));
  const float energyFactor = lerp(1.0f, 1.0f / 1.51f, roughness);

  const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

  const float angle          = __saturatef(dot_product(normal, ray));
  const float previous_angle = __saturatef(dot_product(V, normal));

  const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
  const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

  weight *= FDL * FDV * energyFactor * (1.0f - metallic);

  record.r *= albedo.r * weight;
  record.g *= albedo.g * weight;
  record.b *= albedo.b * weight;

  return ray;
}

__device__ vec3 refraction_BRDF(
  RGBF& record, const vec3 normal, const vec3 ray, const float roughness, const float index, const float r1, const float r2) {
  const float alpha = roughness * roughness;

  vec3 H;

  const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);
  const vec3 V                   = scale_vector(ray, -1.0f);
  const vec3 V_local             = rotate_vector_by_quaternion(V, rotation_to_z);

  const vec3 H_local = sample_GGX_VNDF(V_local, alpha, r1, r2);
  H                  = rotate_vector_by_quaternion(H_local, inverse_quaternion(rotation_to_z));

  const float HdotV = fmaxf(eps, fminf(1.0f, dot_product(H_local, V_local)));
  const float NdotH = fmaxf(eps, fminf(1.0f, H_local.z));
  const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));

  RGBF specular_f0 = get_color(1.0f, 1.0f, 1.0f);

  const RGBF F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotV);

  const float weight = Smith_G2_over_G1_height_correlated(alpha * alpha, NdotH, NdotV);

  record.r *= F.r * weight;
  record.g *= F.g * weight;
  record.b *= F.b * weight;

  const float a = -dot_product(ray, H);
  const float b = 1.0f - index * index * (1.0f - a * a);

  if (b < 0.0f) {
    return normalize_vector(reflect_vector(ray, scale_vector(H, -1.0f)));
  }

  return normalize_vector(add_vector(scale_vector(ray, index), scale_vector(H, index * a - sqrtf(b))));
}

__device__ LightSample sample_light(const vec3 position, const vec3 normal) {
  const int sun_visible = (device_sun.y >= -0.1f);
  const int toy_visible = (device_scene.toy.active && device_scene.toy.emissive);
  uint32_t light_count  = 0;
  light_count += sun_visible;
  light_count += toy_visible;
  light_count += (device_lights_active) ? device_scene.lights_length - 2 : 0;

  float weight_sum = 0.0f;
  LightSample selected;
  selected.weight = -1.0f;

  if (!light_count)
    return selected;

  const int reservoir_sampling_size = min(light_count, 8);

  for (int i = 0; i < reservoir_sampling_size; i++) {
    const float r1       = white_noise();
    uint32_t light_index = (uint32_t) (r1 * light_count);

    light_index += !sun_visible;
    light_index += (toy_visible || light_index < TOY_LIGHT) ? 0 : 1;

    const float4 light_data = __ldg((float4*) (device_scene.lights + light_index));

    LightSample light;
    light.dir.x = light_data.x;
    light.dir.y = light_data.y;
    light.dir.z = light_data.z;
    light.dir   = sub_vector(light.dir, position);

    const float d = get_length(light.dir) + eps;

    light.dir    = normalize_vector(light.dir);
    light.angle  = __saturatef(atanf(light_data.w / d));
    light.id     = light_index;
    light.weight = light.angle * (dot_product(light.dir, normal) >= 0.0f);

    switch (light_index) {
      case 0:
        light.weight *= device_scene.sky.sun_strength;
        break;
      case 1:
        light.weight *= device_scene.toy.material.b;
        break;
      default:
        light.weight *= device_default_material.b;
        break;
    }

    weight_sum += light.weight;

    const float r2 = white_noise();

    if (r2 < light.weight / weight_sum) {
      selected = light;
    }
  }

  selected.weight = light_count / reservoir_sampling_size * selected.angle * weight_sum / selected.weight;

  return selected;
}

#endif /* CU_BRDF_H */
