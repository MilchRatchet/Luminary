#ifndef CU_SHADING_KERNEL_H
#define CU_SHADING_KERNEL_H

#if defined(SHADING_KERNEL) && defined(OPTIX_KERNEL)

#include "bridges.cuh"
#include "bsdf.cuh"
#include "caustics.cuh"
#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "ris.cuh"
#include "sky.cuh"
#include "utils.cuh"

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

  if (!sphere_ray_hit(dir_bsdf, sky_pos, device.sky.sun_pos, SKY_SUN_RADIUS)) {
    light_bsdf = get_color(0.0f, 0.0f, 0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample a direction in the sun's solid angle
  ////////////////////////////////////////////////////////////////////

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_LIGHT_SUN_RAY, index);

  float solid_angle;
  const vec3 dir_solid_angle = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
  RGBF light_solid_angle     = sky_get_sun_color(sky_pos, dir_solid_angle);

  bool is_refraction_solid_angle;
  const RGBF value_solid_angle = bsdf_evaluate(data, dir_solid_angle, BSDF_SAMPLING_GENERAL, is_refraction_solid_angle, 1.0f);
  light_solid_angle            = mul_color(light_solid_angle, value_solid_angle);

  ////////////////////////////////////////////////////////////////////
  // Resampled Importance Sampling
  ////////////////////////////////////////////////////////////////////

  const float target_pdf_bsdf        = color_importance(light_bsdf);
  const float target_pdf_solid_angle = color_importance(light_solid_angle);

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

  light_color = scale_color(light_color, sum_weights / target_pdf);

  float dist = FLT_MAX;

  if (target_pdf == 0.0f) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute Visibility
  ////////////////////////////////////////////////////////////////////

  if (color_importance(light_color) == 0.0f) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
  }
  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  // TODO: Add specialized anyhit shaders for non geometry lights
  const TriangleHandle handle = triangle_handle_get(LIGHT_ID_SUN, 0);

  RGBF visibility = optix_geometry_shadowing(position, dir, dist, handle, index);
  visibility      = mul_color(visibility, volume_integrate_transmittance(position, dir, dist));

  return mul_color(light_color, visibility);
}

__device__ RGBF
  optix_compute_light_ray_sun_caustic(const GBufferData data, const ushort2 index, const vec3 sky_pos, const bool is_underwater) {
  ////////////////////////////////////////////////////////////////////
  // Sample a caustic connection vertex using RIS
  ////////////////////////////////////////////////////////////////////

  float solid_angle;
  const float2 sun_dir_random                  = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAUSTIC_SUN_DIR, index);
  const vec3 sun_dir                           = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, sun_dir_random, solid_angle);
  const CausticsSamplingDomain sampling_domain = caustics_get_domain(data, sun_dir, is_underwater);

  vec3 connection_point;
  float sum_connection_weight = 0.0f;
  float connection_weight;

  if (sampling_domain.fast_path) {
    vec3 sample_point;
    float sample_weight;
    caustics_find_connection_point(data, index, sampling_domain, is_underwater, 0, sample_point, sample_weight);

    sum_connection_weight = sample_weight;
    connection_point      = sample_point;
    connection_weight     = sample_weight;
  }
  else {
    const uint32_t num_samples = device.ocean.caustics_ris_sample_count;

    float resampling_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_CAUSTIC_RESAMPLE, index);

    // RIS with target weight being the Dirac delta of if the connection point is valid or not.
    for (uint32_t i = 0; i < num_samples; i++) {
      vec3 sample_point;
      float sample_weight;
      if (caustics_find_connection_point(data, index, sampling_domain, is_underwater, i, sample_point, sample_weight)) {
        sum_connection_weight += sample_weight;

        const float resampling_probability = sample_weight / sum_connection_weight;
        if (resampling_random < resampling_probability) {
          connection_point = sample_point;

          resampling_random = resampling_random / resampling_probability;
        }
        else {
          resampling_random = (resampling_random - resampling_probability) / (1.0f - resampling_probability);
        }
      }
    }

    connection_weight = (1.0f / num_samples) * sum_connection_weight;

    // Make no mistake. I do in fact have no idea what I am doing. So I just empirically gathered that these
    // weights are correct (they are very unlikely to be correct). I will look into fixing this the moment
    // I start caring.

    // Inspired by the famous factor required for refraction when sampling importance. Note that one of the IOR is 1.0f.
    connection_weight *= device.ocean.refractive_index * device.ocean.refractive_index;

    if (is_underwater) {
      // ... and why not apply it again, we are just sampling some extra importance clearly. And a 2.0f for good measure.
      connection_weight *= device.ocean.refractive_index * device.ocean.refractive_index * 2.0f;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled connection vertex
  ////////////////////////////////////////////////////////////////////

  vec3 pos_to_ocean = sub_vector(connection_point, data.position);

  float dist     = get_length(pos_to_ocean);
  float sun_dist = FLT_MAX;
  const vec3 dir = normalize_vector(pos_to_ocean);

  RGBF light_color = sky_get_sun_color(world_to_sky_transform(connection_point), sun_dir);

  if (sum_connection_weight == 0.0f) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
    sun_dist    = 0.0f;
  }

  bool is_refraction;
  const RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, is_refraction, connection_weight);
  light_color           = mul_color(light_color, bsdf_value);

  const vec3 ocean_normal =
    (sampling_domain.fast_path) ? get_vector(0.0f, 1.0f, 0.0f) : ocean_get_normal(connection_point, OCEAN_ITERATIONS_NORMAL_CAUSTICS);
  const vec3 normal = scale_vector(ocean_normal, (is_underwater) ? -1.0f : 1.0f);

  bool total_reflection;
  const vec3 refraction_dir =
    refract_vector(scale_vector(dir, -1.0f), normal, sampling_domain.ior_in / sampling_domain.ior_out, total_reflection);
  const float reflection_coefficient =
    ocean_reflection_coefficient(normal, dir, refraction_dir, sampling_domain.ior_in, sampling_domain.ior_out);

  light_color = scale_color(light_color, (is_underwater) ? 1.0f - reflection_coefficient : reflection_coefficient);

  if (color_importance(light_color) == 0.0f) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
    sun_dist    = 0.0f;
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
    sun_dist    = 0.0f;
  }

  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  const TriangleHandle handle = triangle_handle_get(LIGHT_ID_SUN, 0);

  RGBF visibility = optix_geometry_shadowing(position, dir, dist, handle, index);
  visibility      = mul_color(visibility, optix_geometry_shadowing(connection_point, sun_dir, sun_dist, handle, index));
  visibility      = scale_color(visibility, volume_integrate_transmittance_fog(connection_point, sun_dir, sun_dist));

  if (is_underwater) {
    visibility = mul_color(visibility, volume_integrate_transmittance_ocean(position, dir, dist, true));
  }
  else {
    visibility = scale_color(visibility, volume_integrate_transmittance_fog(position, dir, dist));
  }

  return mul_color(light_color, visibility);
}

__device__ RGBF optix_compute_light_ray_sun(const GBufferData data, const ushort2 index) {
  if (device.sky.mode == LUMINARY_SKY_MODE_CONSTANT_COLOR)
    return get_color(0.0f, 0.0f, 0.0f);

  const vec3 sky_pos     = world_to_sky_transform(data.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  // This sucks. I want to avoid conditionally executing optixTrace but removing this conditional would hurt performance
  // in night scenes. And in large scenes we may have divergence here during twilight.
  if (!sun_visible)
    return get_color(0.0f, 0.0f, 0.0f);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.ocean.active && data.instance_id != HIT_TYPE_OCEAN) {
    is_underwater  = ocean_get_relative_height(data.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;
    sample_direct  = !is_underwater;
    sample_caustic = device.ocean.caustics_active || is_underwater;
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
// Lighting from Geometry
////////////////////////////////////////////////////////////////////

#ifndef VOLUME_KERNEL

__device__ RGBF optix_compute_light_ray_geometry_single(GBufferData data, const ushort2 index, const uint32_t light_ray_index) {
  // We have to clamp due to numerical precision issues in the microfacet models.
  data.roughness = fmaxf(data.roughness, BSDF_ROUGHNESS_CLAMP);

  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const QuasiRandomTarget bsdf_target = (QuasiRandomTarget) (QUASI_RANDOM_TARGET_LIGHT_BSDF + 2 * light_ray_index);
  const vec3 bsdf_dir                 = bsdf_sample_for_light(data, index, bsdf_target, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  vec3 position;
  float3 origin, ray;
  float shift;

  shift    = bsdf_sample_is_refraction ? -eps : eps;
  position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

  origin = make_float3(position.x, position.y, position.z);
  ray    = make_float3(bsdf_dir.x, bsdf_dir.y, bsdf_dir.z);

  unsigned int bsdf_sample_light_key = HIT_TYPE_LIGHT_BSDF_HINT;
  const float light_search_dist      = (bsdf_sample_is_valid) ? FLT_MAX : -1.0f;

  // The compiler has issues with conditional optixTrace, hence we disable them using a negative max dist.
  // TODO: Add a ray flag to skip anyhit because we don't use it right now.
  OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_TRIANGLE_ID, 0);
  optixTrace(
    device.optix_bvh_light, origin, ray, 0.0f, light_search_dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    OPTIX_SBT_OFFSET_LIGHT_BSDF_TRACE, 0, 0, bsdf_sample_light_key);

  ////////////////////////////////////////////////////////////////////
  // Resample the BSDF direction with NEE based directions
  ////////////////////////////////////////////////////////////////////

  vec3 dir;
  RGBF light_color;
  float dist;
  bool is_refraction;
  const TriangleHandle light_handle = ris_sample_light(
    data, index, light_ray_index, bsdf_sample_light_key, bsdf_dir, bsdf_sample_is_refraction, dir, light_color, dist, is_refraction);

  if (color_importance(light_color) == 0.0f || light_handle.instance_id == LIGHT_ID_NONE) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
    light_color = get_color(0.0f, 0.0f, 0.0f);
    dist        = 0.0f;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute visibility term
  ////////////////////////////////////////////////////////////////////

  position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  RGBF visibility = optix_geometry_shadowing(position, dir, dist, light_handle, index);
  visibility      = mul_color(visibility, volume_integrate_transmittance(position, dir, dist));

  light_color = mul_color(light_color, visibility);

  UTILS_CHECK_NANS(index, light_color);

  return light_color;
}

// Shortened from geometry to geo so the function name length would be the same as the other ones.
__device__ RGBF optix_compute_light_ray_geo(const GBufferData data, const ushort2 index) {
  RGBF geometry_light = get_color(0.0f, 0.0f, 0.0f);

  if (device.settings.light_num_rays) {
    for (int j = 0; j < device.settings.light_num_rays; j++) {
      geometry_light = add_color(geometry_light, optix_compute_light_ray_geometry_single(data, index, j));
    }

    geometry_light = scale_color(geometry_light, 1.0f / device.settings.light_num_rays);
  }

  return geometry_light;
}

#endif /* !VOLUME_KERNEL */

__device__ RGBF optix_compute_light_ray_ambient_sky(
  const GBufferData data, const vec3 ray, const RGBF sample_weight, const bool is_refraction, const ushort2 index) {
  if (device.state.depth < device.settings.max_ray_depth || !device.sky.ambient_sampling)
    return get_color(0.0f, 0.0f, 0.0f);

  // We don't support compute based sky due to register/performance reasons and because
  // we would have to include clouds then aswell.
  if (device.sky.mode == LUMINARY_SKY_MODE_DEFAULT)
    return get_color(0.0f, 0.0f, 0.0f);

  const vec3 position = shift_origin_vector(data.position, data.V, ray, is_refraction);

  RGBF sky_light = sky_color_no_compute(position, ray, data.state, get_pixel_id(index), index);

  const TriangleHandle handle = triangle_handle_get(LIGHT_ID_SUN, 0);

  sky_light = mul_color(sky_light, optix_geometry_shadowing(position, ray, FLT_MAX, handle, index));
  sky_light = mul_color(sky_light, volume_integrate_transmittance(position, ray, FLT_MAX));
  sky_light = mul_color(sky_light, sample_weight);

  UTILS_CHECK_NANS(index, sky_light);

  return sky_light;
}

#endif /* SHADING_KERNEL && OPTIX_KERNEL */

#endif /* CU_SHADING_KERNEL */
