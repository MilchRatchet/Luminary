#ifndef CU_SHADING_KERNEL_H
#define CU_SHADING_KERNEL_H

#if defined(SHADING_KERNEL) && defined(OPTIX_KERNEL)

#include "bsdf.cuh"
#include "caustics.cuh"
#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define SKIP_IOR_CHECK (0xFFFFFFFF)
#define MAX_COMPRESSABLE_COLOR (1.99999988079071044921875f)

__device__ void optix_compress_color(RGBF color, unsigned int& data0, unsigned int& data1) {
  uint32_t bits_r = (__float_as_uint(fminf(color.r + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_g = (__float_as_uint(fminf(color.g + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_b = (__float_as_uint(fminf(color.b + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;

  data0 = bits_r | (bits_g << 21);
  data1 = (bits_g >> 11) | (bits_b << 10);
}

__device__ RGBF optix_decompress_color(unsigned int data0, unsigned int data1) {
  uint32_t bits_r = data0 & 0x1FFFFF;
  uint32_t bits_g = (data0 >> 21) & 0x7FF | ((data1 & 0x3FF) << 11);
  uint32_t bits_b = (data1 >> 10) & 0x1FFFFF;

  RGBF color;
  color.r = __uint_as_float(0x3F800000u | (bits_r << 2)) - 1.0f;
  color.g = __uint_as_float(0x3F800000u | (bits_g << 2)) - 1.0f;
  color.b = __uint_as_float(0x3F800000u | (bits_b << 2)) - 1.0f;

  return color;
}

__device__ bool optix_evaluate_ior_culling(const uint32_t ior_data, const ushort2 index) {
  const int8_t ior_stack_pop_max = (int8_t) (ior_data >> 24);
  if (ior_stack_pop_max > 0) {
    const uint32_t ior_stack = device.ptrs.ior_stack[get_pixel_id(index.x, index.y)];

    const uint32_t ray_ior = (ior_data & 0xFF);

    if (ior_stack_pop_max >= 3) {
      if (ray_ior != (ior_stack >> 24))
        return true;
    }

    if (ior_stack_pop_max >= 2) {
      if (ray_ior != (ior_stack >> 16))
        return true;
    }

    if (ior_stack_pop_max >= 1) {
      if (ray_ior != (ior_stack >> 8))
        return true;
    }
  }

  return false;
}

__device__ bool optix_toy_shadowing(
  const vec3 position, const vec3 dir, const float dist, unsigned int& compressed_ior, RGBF& light_color) {
  if (device.scene.toy.active) {
    const float toy_dist = get_toy_distance(position, dir);

    if (toy_dist < dist) {
      if (device.scene.material.enable_ior_shadowing && ior_compress(device.scene.toy.refractive_index) != compressed_ior)
        return false;

      RGBF toy_transparency = scale_color(opaque_color(device.scene.toy.albedo), 1.0f - device.scene.toy.albedo.a);

      if (luminance(toy_transparency) == 0.0f)
        return false;

      // Toy can be hit at most twice, compute the intersection and on hit apply the alpha.
      vec3 toy_hit_origin = add_vector(position, scale_vector(dir, toy_dist));
      toy_hit_origin      = add_vector(toy_hit_origin, scale_vector(dir, get_length(toy_hit_origin) * eps * 16.0f));

      const float toy_dist2 = get_toy_distance(toy_hit_origin, dir);

      bool two_hits = false;
      if (toy_dist2 < dist) {
        toy_transparency = mul_color(toy_transparency, toy_transparency);
        two_hits         = true;
      }

      if (!two_hits) {
        // Set ray ior pop values to 1,1 or 0,-1 (max,balance)
        compressed_ior |= (toy_is_inside(position, dir)) ? 0x01010000 : 0x00FF0000;
      }

      light_color = mul_color(light_color, toy_transparency);
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////
// Lighting from Sun
////////////////////////////////////////////////////////////////////

__device__ RGBF optix_compute_light_ray_sun_direct(GBufferData data, const ushort2 index, const vec3 sky_pos) {
  // We have to clamp due to numerical precision issues in the microfacet models.
  data.roughness = fmaxf(data.roughness, BSDF_ROUGHNESS_CLAMP);

  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const vec3 dir_bsdf =
    bsdf_sample_for_light(data, index, QUASI_RANDOM_TARGET_LIGHT_SUN_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);
  RGBF light_bsdf = sky_get_sun_color(sky_pos, dir_bsdf);

  bool is_refraction_bsdf;
  const RGBF value_bsdf = bsdf_evaluate(data, dir_bsdf, BSDF_SAMPLING_GENERAL, is_refraction_bsdf, 1.0f);
  light_bsdf            = mul_color(light_bsdf, value_bsdf);

  if (!sphere_ray_hit(dir_bsdf, sky_pos, device.sun_pos, SKY_SUN_RADIUS)) {
    light_bsdf = get_color(0.0f, 0.0f, 0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample a direction in the sun's solid angle
  ////////////////////////////////////////////////////////////////////

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LIGHT_SUN_RAY, index);

  float solid_angle;
  const vec3 dir_solid_angle = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
  RGBF light_solid_angle     = sky_get_sun_color(sky_pos, dir_solid_angle);

  bool is_refraction_solid_angle;
  const RGBF value_solid_angle = bsdf_evaluate(data, dir_solid_angle, BSDF_SAMPLING_GENERAL, is_refraction_solid_angle, 1.0f);
  light_solid_angle            = mul_color(light_solid_angle, value_solid_angle);

  ////////////////////////////////////////////////////////////////////
  // Resampled Importance Sampling
  ////////////////////////////////////////////////////////////////////

  const float target_pdf_bsdf        = luminance(light_bsdf);
  const float target_pdf_solid_angle = luminance(light_solid_angle);

  // MIS weight multiplied with PDF
  const float mis_weight_bsdf        = solid_angle / (bsdf_sample_for_light_pdf(data, dir_bsdf) * solid_angle + 1.0f);
  const float mis_weight_solid_angle = solid_angle / (bsdf_sample_for_light_pdf(data, dir_solid_angle) * solid_angle + 1.0f);

  const float weight_bsdf        = target_pdf_bsdf * mis_weight_bsdf;
  const float weight_solid_angle = target_pdf_solid_angle * mis_weight_solid_angle;

  const float sum_weights = weight_bsdf + weight_solid_angle;

  float target_pdf;
  vec3 dir;
  RGBF light_color;
  bool is_refraction;
  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_SUN_RIS_RESAMPLING, index) * sum_weights < weight_bsdf) {
    dir           = dir_bsdf;
    target_pdf    = target_pdf_bsdf;
    light_color   = light_bsdf;
    is_refraction = is_refraction_bsdf;
  }
  else {
    dir           = dir_solid_angle;
    target_pdf    = target_pdf_solid_angle;
    light_color   = light_solid_angle;
    is_refraction = is_refraction_solid_angle;
  }

  if (target_pdf == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  light_color = scale_color(light_color, sum_weights / target_pdf);

  ////////////////////////////////////////////////////////////////////
  // Compute Visibility
  ////////////////////////////////////////////////////////////////////

  if (luminance(light_color) == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  unsigned int compressed_ior = ior_compress(is_refraction ? data.ior_out : data.ior_in);

  if (!optix_toy_shadowing(position, dir, FLT_MAX, compressed_ior, light_color))
    return get_color(0.0f, 0.0f, 0.0f);

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  const float3 origin = make_float3(position.x, position.y, position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  // TODO: Add specialized anyhit shaders for non geometry lights
  unsigned int hit_id = LIGHT_ID_SUN;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, FLT_MAX, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  if (device.scene.material.enable_ior_shadowing && optix_evaluate_ior_culling(compressed_ior, index))
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);
  visibility      = mul_color(visibility, volume_integrate_transmittance(position, dir, FLT_MAX));

  return mul_color(light_color, visibility);
}

__device__ RGBF
  optix_compute_light_ray_sun_caustic(const GBufferData data, const ushort2 index, const vec3 sky_pos, const bool is_underwater) {
  ////////////////////////////////////////////////////////////////////
  // Sample a caustic connection vertex using RIS
  ////////////////////////////////////////////////////////////////////

  const vec3 sun_dir                           = normalize_vector(sub_vector(device.sun_pos, sky_pos));
  const CausticsSamplingDomain sampling_domain = caustics_get_domain(data, sun_dir, is_underwater);

  vec3 connection_point;
  float sum_connection_weight = 0.0f;

  const uint32_t num_samples = (device.scene.ocean.amplitude > 0.0f) ? device.scene.ocean.caustics_ris_sample_count : 1;

  // RIS with target weight being the Dirac delta of if the connection point is valid or not.
  for (uint32_t i = 0; i < num_samples; i++) {
    vec3 sample_point;
    float sample_weight;
    if (caustics_find_connection_point(data, index, sampling_domain, is_underwater, i, sample_point, sample_weight)) {
      sum_connection_weight += sample_weight;
      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_CAUSTIC_RESAMPLE, index) * sum_connection_weight < sample_weight) {
        connection_point = sample_point;
      }
    }
  }

  const float connection_weight = (1.0f / num_samples) * sum_connection_weight;

  if (sum_connection_weight == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled connection vertex
  ////////////////////////////////////////////////////////////////////

  vec3 pos_to_ocean = sub_vector(connection_point, data.position);

  const float dist = get_length(pos_to_ocean);
  const vec3 dir   = normalize_vector(pos_to_ocean);

  RGBF light_color = sky_get_sun_color(world_to_sky_transform(connection_point), sun_dir);

  bool is_refraction;
  const RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, is_refraction, connection_weight);
  light_color           = mul_color(light_color, bsdf_value);

  const vec3 normal = scale_vector(ocean_get_normal(connection_point, OCEAN_ITERATIONS_NORMAL_CAUSTICS), (is_underwater) ? -1.0f : 1.0f);

  bool total_reflection;
  const vec3 refraction_dir =
    refract_vector(scale_vector(dir, -1.0f), normal, sampling_domain.ior_in / sampling_domain.ior_out, total_reflection);
  const float reflection_coefficient =
    ocean_reflection_coefficient(normal, dir, refraction_dir, sampling_domain.ior_in, sampling_domain.ior_out);

  light_color = scale_color(light_color, (is_underwater) ? 1.0f - reflection_coefficient : reflection_coefficient);

  // Reduce light intensity based on regularization factor, this is not physically correct in any way
  light_color = scale_color(light_color, 1.0f / device.scene.ocean.caustics_regularization);

  if (luminance(light_color) < eps)
    return get_color(0.0f, 0.0f, 0.0f);

  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  unsigned int hit_id = LIGHT_ID_SUN;

  float3 origin = make_float3(position.x, position.y, position.z);
  float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int compressed_ior = ior_compress(is_underwater ? device.scene.ocean.refractive_index : 1.0f);

  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  if (!optix_toy_shadowing(position, dir, dist, compressed_ior, light_color))
    return get_color(0.0f, 0.0f, 0.0f);

  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  hit_id = LIGHT_ID_SUN;
  origin = make_float3(connection_point.x, connection_point.y, connection_point.z);
  ray    = make_float3(sun_dir.x, sun_dir.y, sun_dir.z);

  if (!optix_toy_shadowing(connection_point, sun_dir, FLT_MAX, compressed_ior, light_color))
    return get_color(0.0f, 0.0f, 0.0f);

  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, FLT_MAX, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);
  visibility      = scale_color(visibility, volume_integrate_transmittance_fog(connection_point, sun_dir, FLT_MAX));

  if (is_underwater) {
    visibility = mul_color(visibility, volume_integrate_transmittance_ocean(position, dir, dist, true));
  }
  else {
    visibility = scale_color(visibility, volume_integrate_transmittance_fog(position, dir, dist));
  }

  return mul_color(light_color, visibility);
}

__device__ RGBF optix_compute_light_ray_sun(const GBufferData data, const ushort2 index) {
  const vec3 sky_pos     = world_to_sky_transform(data.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  if (!sun_visible)
    return get_color(0.0f, 0.0f, 0.0f);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.scene.ocean.active) {
    // TODO: Change the iterations count if necessary.
    is_underwater  = ocean_get_relative_height(data.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;
    sample_direct  = !device.scene.ocean.caustics_active || !is_underwater;
    sample_caustic = device.scene.ocean.caustics_active && data.hit_id != HIT_TYPE_OCEAN;
  }

  RGBF sun_light = get_color(0.0f, 0.0f, 0.0f);

  if (sample_direct) {
    const RGBF direct_light = optix_compute_light_ray_sun_direct(data, index, sky_pos);
    sun_light               = add_color(sun_light, direct_light);
  }

  if (sample_caustic) {
    const RGBF caustic_light = optix_compute_light_ray_sun_caustic(data, index, sky_pos, is_underwater);
    sun_light                = add_color(sun_light, caustic_light);
  }

  return sun_light;
}

////////////////////////////////////////////////////////////////////
// Lighting from Toy
////////////////////////////////////////////////////////////////////

__device__ RGBF optix_compute_light_ray_toy(GBufferData data, const ushort2 index) {
  const bool toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  if (!toy_visible)
    return get_color(0.0f, 0.0f, 0.0f);

  const RGBF toy_emission = scale_color(device.scene.toy.emission, device.scene.toy.material.b);

  if (luminance(toy_emission) == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  // We have to clamp due to numerical precision issues in the microfacet models.
  data.roughness = fmaxf(data.roughness, BSDF_ROUGHNESS_CLAMP);

  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const vec3 dir_bsdf =
    bsdf_sample_for_light(data, index, QUASI_RANDOM_TARGET_LIGHT_TOY_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);
  RGBF light_bsdf = toy_emission;

  bool is_refraction_bsdf;
  const RGBF value_bsdf = bsdf_evaluate(data, dir_bsdf, BSDF_SAMPLING_GENERAL, is_refraction_bsdf, 1.0f);
  light_bsdf            = mul_color(light_bsdf, value_bsdf);

  if (get_toy_distance(data.position, dir_bsdf) == FLT_MAX) {
    light_bsdf = get_color(0.0f, 0.0f, 0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample a direction in the sun's solid angle
  ////////////////////////////////////////////////////////////////////

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LIGHT_TOY_RAY, index);

  const vec3 dir_solid_angle = toy_sample_ray(data.position, random);
  RGBF light_solid_angle     = toy_emission;

  bool is_refraction_solid_angle;
  const RGBF value_solid_angle = bsdf_evaluate(data, dir_solid_angle, BSDF_SAMPLING_GENERAL, is_refraction_solid_angle, 1.0f);
  light_solid_angle            = mul_color(light_solid_angle, value_solid_angle);

  ////////////////////////////////////////////////////////////////////
  // Resampled Importance Sampling
  ////////////////////////////////////////////////////////////////////

  const float solid_angle = toy_get_solid_angle(data.position);

  const float target_pdf_bsdf        = luminance(light_bsdf);
  const float target_pdf_solid_angle = luminance(light_solid_angle);

  // MIS weight multiplied with PDF
  const float mis_weight_bsdf        = solid_angle / (bsdf_sample_for_light_pdf(data, dir_bsdf) * solid_angle + 1.0f);
  const float mis_weight_solid_angle = solid_angle / (bsdf_sample_for_light_pdf(data, dir_solid_angle) * solid_angle + 1.0f);

  const float weight_bsdf        = target_pdf_bsdf * mis_weight_bsdf;
  const float weight_solid_angle = target_pdf_solid_angle * mis_weight_solid_angle;

  const float sum_weights = weight_bsdf + weight_solid_angle;

  float target_pdf;
  vec3 dir;
  RGBF light_color;
  bool is_refraction;
  if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_SUN_RIS_RESAMPLING, index) * sum_weights < weight_bsdf) {
    dir           = dir_bsdf;
    target_pdf    = target_pdf_bsdf;
    light_color   = light_bsdf;
    is_refraction = is_refraction_bsdf;
  }
  else {
    dir           = dir_solid_angle;
    target_pdf    = target_pdf_solid_angle;
    light_color   = light_solid_angle;
    is_refraction = is_refraction_solid_angle;
  }

  if (target_pdf == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  light_color = scale_color(light_color, sum_weights / target_pdf);

  ////////////////////////////////////////////////////////////////////
  // Compute Visibility
  ////////////////////////////////////////////////////////////////////

  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  const float dist = get_toy_distance(position, dir);

  const float3 origin = make_float3(position.x, position.y, position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int hit_id = LIGHT_ID_TOY;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  unsigned int compressed_ior = ior_compress(is_refraction ? data.ior_out : data.ior_in);

  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  if (device.scene.material.enable_ior_shadowing && optix_evaluate_ior_culling(compressed_ior, index))
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);
  visibility      = mul_color(visibility, volume_integrate_transmittance(position, dir, dist));

  return mul_color(light_color, visibility);
}

////////////////////////////////////////////////////////////////////
// Lighting from Geometry
////////////////////////////////////////////////////////////////////

__device__ RGBF optix_compute_light_ray_geometry_single(GBufferData data, const ushort2 index, const uint32_t light_ray_index) {
  if (!device.scene.material.lights_active)
    return get_color(0.0f, 0.0f, 0.0f);

  // We have to clamp due to numerical precision issues in the microfacet models.
  data.roughness = fmaxf(data.roughness, BSDF_ROUGHNESS_CLAMP);

  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const vec3 bsdf_dir = bsdf_sample_for_light(data, index, QUASI_RANDOM_TARGET_LIGHT_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  vec3 position;
  float3 origin, ray;
  float shift;

  shift    = bsdf_sample_is_refraction ? -eps : eps;
  position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

  origin = make_float3(position.x, position.y, position.z);
  ray    = make_float3(bsdf_dir.x, bsdf_dir.y, bsdf_dir.z);

  unsigned int bsdf_light_id = HIT_TYPE_LIGHT_BSDF_HINT;

  float light_search_dist = (bsdf_sample_is_valid) ? FLT_MAX : -1.0f;

  // The compiler has issues with conditional optixTrace, hence we disable them using a negative max dist.
  optixTrace(device.optix_bvh_light, origin, ray, 0.0f, light_search_dist, 0.0f, OptixVisibilityMask(0xFFFF), 0, 0, 0, 0, bsdf_light_id);

  ////////////////////////////////////////////////////////////////////
  // Resample the BSDF direction with NEE based directions
  ////////////////////////////////////////////////////////////////////

  vec3 dir;
  RGBF light_color;
  float dist;
  bool is_refraction;
  const uint32_t light_id = ris_sample_light(
    data, index, light_ray_index, bsdf_light_id, bsdf_dir, bsdf_sample_is_refraction, dir, light_color, dist, is_refraction);

  if (luminance(light_color) == 0.0f || light_id == LIGHT_ID_NONE)
    return get_color(0.0f, 0.0f, 0.0f);

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  origin = make_float3(position.x, position.y, position.z);
  ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int compressed_ior = ior_compress(is_refraction ? data.ior_out : data.ior_in);

  if (!optix_toy_shadowing(position, dir, dist, compressed_ior, light_color))
    return get_color(0.0f, 0.0f, 0.0f);

  unsigned int hit_id = light_id;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  // For triangle lights, we cannot rely on fully opaque OMMs because if we first hit the target light and then execute the closest hit for
  // that, then we will never know if there is an occluder. Similarly, skipping anyhit for fully opaque needs to still terminate the ray so
  // I enforce anyhit.
  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_ENFORCE_ANYHIT, 0, 0, 0, hit_id,
    alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  if (device.scene.material.enable_ior_shadowing && optix_evaluate_ior_culling(compressed_ior, index))
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);
  visibility      = mul_color(visibility, volume_integrate_transmittance(position, dir, dist));

  return mul_color(light_color, visibility);
}

// Shortened from geometry to geo so the function name length would be the same as the other ones.
__device__ RGBF optix_compute_light_ray_geo(const GBufferData data, const ushort2 index) {
  RGBF geometry_light = get_color(0.0f, 0.0f, 0.0f);

  if (device.ris_settings.num_light_rays) {
    for (int j = 0; j < device.ris_settings.num_light_rays; j++) {
      geometry_light = add_color(geometry_light, optix_compute_light_ray_geometry_single(data, index, j));
    }

    geometry_light = scale_color(geometry_light, 1.0f / device.ris_settings.num_light_rays);
  }

  return geometry_light;
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ RGBAF optix_alpha_test(const unsigned int ray_ior) {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);

  // Don't check for IOR when querying a light from a BSDF sample
  if (ray_ior != SKIP_IOR_CHECK) {
    const uint32_t compressed_ior = ior_compress(__ldg(&(device.scene.materials[material_id].refraction_index)));

    // This assumes that IOR is compressed into 8 bits.
    if (device.scene.material.enable_ior_shadowing && compressed_ior != (ray_ior & 0xFF)) {
      // Terminate ray.
      return get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);
    }
  }

  const uint16_t tex = __ldg(&(device.scene.materials[material_id].albedo_map));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v);

    RGBAF albedo;
    albedo.r = tex_value.x;
    albedo.g = tex_value.y;
    albedo.b = tex_value.z;
    albedo.a = tex_value.w;

    return albedo;
  }

  RGBAF albedo;
  albedo.r = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_r)));
  albedo.g = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_g)));
  albedo.b = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_b)));
  albedo.a = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_a)));

  return albedo;
}

extern "C" __global__ void __anyhit__optix() {
  unsigned int target_light = optixGetPayload_0();

  // First check if the target light is a triangle light so we don't unnecessarily load light IDs when sampling the sun or the toy.
  if (target_light < LIGHT_ID_TRIANGLE_ID_LIMIT && target_light == load_triangle_light_id(optixGetPrimitiveIndex())) {
    optixIgnoreIntersection();
  }

  const bool bsdf_sampling_query = (target_light == HIT_TYPE_LIGHT_BSDF_HINT);

  unsigned int ray_ior = (bsdf_sampling_query) ? SKIP_IOR_CHECK : optixGetPayload_3();

  const RGBAF albedo = optix_alpha_test(ray_ior);

  // For finding the hit light of BSDF rays we only care about ignoring fully transparent hits.
  // I don't have OMMs for this BVH.
  if (bsdf_sampling_query) {
    if (albedo.a == 0.0f) {
      optixIgnoreIntersection();
    }

    return;
  }

  if (albedo.a == 1.0f) {
    optixSetPayload_0(HIT_TYPE_REJECT);

    optixTerminateRay();
  }

  int8_t ray_ior_pop_balance = (int8_t) ((ray_ior >> 16) & 0xFF);
  int8_t ray_ior_pop_max     = (int8_t) (ray_ior >> 24);

  ray_ior_pop_balance += (optixIsBackFaceHit()) ? 1 : -1;
  ray_ior_pop_max = max(ray_ior_pop_max, ray_ior_pop_balance);

  ray_ior = (ray_ior & 0xFFFF) | (uint32_t) (ray_ior_pop_balance << 16) | (uint32_t) (ray_ior_pop_max << 24);

  optixSetPayload_3(ray_ior);

  if (albedo.a == 0.0f) {
    optixIgnoreIntersection();
  }

  const RGBF alpha = (device.scene.material.colored_transparency) ? scale_color(opaque_color(albedo), 1.0f - albedo.a)
                                                                  : get_color(1.0f - albedo.a, 1.0f - albedo.a, 1.0f - albedo.a);

  unsigned int alpha_data0 = optixGetPayload_1();
  unsigned int alpha_data1 = optixGetPayload_2();

  RGBF accumulated_alpha = optix_decompress_color(alpha_data0, alpha_data1);
  accumulated_alpha      = mul_color(accumulated_alpha, alpha);
  optix_compress_color(accumulated_alpha, alpha_data0, alpha_data1);

  optixSetPayload_1(alpha_data0);
  optixSetPayload_2(alpha_data1);

  optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__optix() {
  unsigned int target_light = optixGetPayload_0();
  if (target_light == HIT_TYPE_LIGHT_BSDF_HINT) {
    optixSetPayload_0(optixGetPrimitiveIndex());
    return;
  }

  // This is never executed for triangle lights so we don't need to check if the closest hit is the target light.
  optixSetPayload_0(HIT_TYPE_REJECT);
}

#endif /* SHADING_KERNEL && OPTIX_KERNEL */

#endif /* CU_SHADING_KERNEL */
