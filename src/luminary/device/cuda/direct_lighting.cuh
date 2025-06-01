#ifndef CU_LUMINARY_DIRECT_LIGHTING_H
#define CU_LUMINARY_DIRECT_LIGHTING_H

#if defined(OPTIX_KERNEL)

#include "bsdf.cuh"
#include "caustics.cuh"
#include "light.cuh"
#include "material.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "random.cuh"
#include "sky.cuh"
#include "utils.cuh"

// #define DIRECT_LIGHTING_NO_SHADOW

struct DirectLightingShadowTask {
  OptixTraceStatus trace_status;
  vec3 origin;
  vec3 ray;
  float limit;
  TriangleHandle target_light;
} typedef DirectLightingShadowTask;

struct DirectLightingBSDFSample {
  uint32_t light_id;
  vec3 ray;
  bool is_refraction;
} typedef DirectLightingBSDFSample;

////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////

__device__ bool direct_lighting_geometry_is_valid(const DeviceTask task) {
  return ((task.state & STATE_FLAG_VOLUME_SCATTERED) == 0);
}

////////////////////////////////////////////////////////////////////
// Shadowing
////////////////////////////////////////////////////////////////////

__device__ RGBF direct_lighting_sun_shadowing(const DirectLightingShadowTask task) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_sun_shadowing(task.origin, task.ray, task.limit, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.origin, task.ray, task.limit));

  return visibility;
#else  /* !DIRECT_LIGHTING_NO_SHADOW */
  return splat_color(1.0f);
#endif /* DIRECT_LIGHTING_NO_SHADOW */
}

__device__ RGBF direct_lighting_shadowing(const DirectLightingShadowTask task) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_geometry_shadowing(task.origin, task.ray, task.limit, task.target_light, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.origin, task.ray, task.limit));

  return visibility;
#else  /* !DIRECT_LIGHTING_NO_SHADOW */
  return splat_color(1.0f);
#endif /* DIRECT_LIGHTING_NO_SHADOW */
}

////////////////////////////////////////////////////////////////////
// Sun
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF
  direct_lighting_sun_direct(MaterialContext<TYPE> ctx, const ushort2 index, const vec3 sky_pos, DirectLightingShadowTask& task) {
  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const vec3 dir_bsdf = bsdf_sample_for_light(ctx, index, RANDOM_TARGET_LIGHT_SUN_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  RGBF light_bsdf         = get_color(0.0f, 0.0f, 0.0f);
  bool is_refraction_bsdf = false;
  if (sphere_ray_hit(dir_bsdf, sky_pos, device.sky.sun_pos, SKY_SUN_RADIUS)) {
    light_bsdf = sky_get_sun_color(sky_pos, dir_bsdf);

    const RGBF value_bsdf = bsdf_evaluate(ctx, dir_bsdf, BSDF_SAMPLING_GENERAL, is_refraction_bsdf, 1.0f);
    light_bsdf            = mul_color(light_bsdf, value_bsdf);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample a direction in the sun's solid angle
  ////////////////////////////////////////////////////////////////////

  const float2 random = random_2D(RANDOM_TARGET_LIGHT_SUN_RAY, index);

  float solid_angle;
  const vec3 dir_solid_angle = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
  RGBF light_solid_angle     = sky_get_sun_color(sky_pos, dir_solid_angle);

  bool is_refraction_solid_angle;
  const RGBF value_solid_angle = bsdf_evaluate(ctx, dir_solid_angle, BSDF_SAMPLING_GENERAL, is_refraction_solid_angle, 1.0f);
  light_solid_angle            = mul_color(light_solid_angle, value_solid_angle);

  ////////////////////////////////////////////////////////////////////
  // Resampled Importance Sampling
  ////////////////////////////////////////////////////////////////////

  const float target_pdf_bsdf        = color_importance(light_bsdf);
  const float target_pdf_solid_angle = color_importance(light_solid_angle);

  // MIS weight multiplied with PDF
  const float mis_weight_bsdf        = solid_angle / (bsdf_sample_for_light_pdf(ctx, dir_bsdf) * solid_angle + 1.0f);
  const float mis_weight_solid_angle = solid_angle / (bsdf_sample_for_light_pdf(ctx, dir_solid_angle) * solid_angle + 1.0f);

  const float weight_bsdf        = target_pdf_bsdf * mis_weight_bsdf;
  const float weight_solid_angle = target_pdf_solid_angle * mis_weight_solid_angle;

  const float sum_weights = weight_bsdf + weight_solid_angle;

  if (sum_weights == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  float target_pdf;
  vec3 dir;
  RGBF light_color;
  bool is_refraction;
  if (random_1D(RANDOM_TARGET_LIGHT_SUN_RESAMPLING, index) * sum_weights < weight_bsdf) {
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

  if (target_pdf == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(ctx, is_refraction)) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  if (color_importance(light_color) == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task.origin       = shift_origin_vector(ctx.position, ctx.V, dir, is_refraction);
  task.ray          = dir;
  task.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task.limit        = FLT_MAX;

  return light_color;
}

template <MaterialType TYPE>
__device__ RGBF direct_lighting_sun_caustic(
  MaterialContext<TYPE> ctx, const ushort2 index, const vec3 sky_pos, const bool is_underwater, DirectLightingShadowTask& task0,
  DirectLightingShadowTask& task1) {
  ////////////////////////////////////////////////////////////////////
  // Sample a caustic connection vertex using RIS
  ////////////////////////////////////////////////////////////////////

  float solid_angle;
  const float2 sun_dir_random                  = random_2D(RANDOM_TARGET_CAUSTIC_SUN_RAY, index);
  const vec3 sun_dir                           = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, sun_dir_random, solid_angle);
  const CausticsSamplingDomain sampling_domain = caustics_get_domain(ctx, sun_dir, is_underwater);

  if (sampling_domain.valid == false)
    return splat_color(0.0f);

  vec3 connection_point;
  float sum_connection_weight = 0.0f;
  float connection_weight;

  if (sampling_domain.fast_path) {
    vec3 sample_point;
    float sample_weight;
    caustics_find_connection_point(ctx, index, sampling_domain, is_underwater, 0, 1, sample_point, sample_weight);

    sum_connection_weight = sample_weight;
    connection_point      = sample_point;
    connection_weight     = sample_weight;
  }
  else {
    const uint32_t num_samples = device.ocean.caustics_ris_sample_count + 1;

    float sum_weights_front = 0.0f;
    float sum_weights_back  = 0.0f;

    uint32_t index_front = (uint32_t) -1;
    uint32_t index_back  = num_samples;

    const float resampling_random = random_1D(RANDOM_TARGET_CAUSTIC_RESAMPLING, index);

    for (uint32_t iteration = 0; iteration <= num_samples; iteration++) {
      const bool compute_front = (sum_weights_front <= resampling_random * (sum_weights_front + sum_weights_back));

      if (!compute_front && iteration == num_samples)
        break;

      uint32_t current_index;
      if (compute_front) {
        current_index = ++index_front;
      }
      else {
        current_index = --index_back;
      }

      // This happens if all samples had a weight of zero
      if (current_index == num_samples)
        break;

      vec3 sample_point;
      float sample_weight = 0.0f;
      const bool valid_hit =
        caustics_find_connection_point(ctx, index, sampling_domain, is_underwater, current_index, num_samples, sample_point, sample_weight);

      if (valid_hit == false || sample_weight == 0.0f)
        continue;

      if (compute_front) {
        connection_point = sample_point;
      }

      // Last iteration cannot add to the weight sums to avoid double counting
      if (iteration == num_samples)
        break;

      if (compute_front) {
        sum_weights_front += sample_weight;
      }
      else {
        sum_weights_back += sample_weight;
      }
    }

    sum_connection_weight = sum_weights_front + sum_weights_back;

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

  vec3 pos_to_ocean = sub_vector(connection_point, ctx.position);

  const float dist = get_length(pos_to_ocean);
  const vec3 dir   = normalize_vector(pos_to_ocean);

  RGBF light_color = sky_get_sun_color(world_to_sky_transform(connection_point), sun_dir);

  if (sum_connection_weight == 0.0f) {
    return splat_color(0.0f);
  }

  bool is_refraction;
  const RGBF bsdf_value = bsdf_evaluate(ctx, dir, BSDF_SAMPLING_GENERAL, is_refraction, connection_weight);
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
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(ctx, is_refraction)) {
    return splat_color(0.0f);
  }

  const vec3 position = shift_origin_vector(ctx.position, ctx.V, dir, is_refraction);

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task0.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task0.origin       = shift_origin_vector(ctx.position, ctx.V, dir, is_refraction);
  task0.ray          = dir;
  task0.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task0.limit        = dist;

  task1.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task1.origin       = connection_point;
  task1.ray          = sun_dir;
  task1.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task1.limit        = FLT_MAX;

  return light_color;
}

////////////////////////////////////////////////////////////////////
// Geometry
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF direct_lighting_geometry_sample(MaterialContext<TYPE> ctx, const ushort2 pixel, DirectLightingShadowTask& task) {
  const LightSampleResult<TYPE> sample = light_sample(ctx, pixel);

  if (sample.handle.instance_id == LIGHT_ID_NONE) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(ctx, sample.is_refraction)) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task.origin       = shift_origin_vector(ctx.position, ctx.V, sample.ray, sample.is_refraction);
  task.ray          = sample.ray;
  task.target_light = sample.handle;
  task.limit        = sample.dist;

  return sample.light_color;
}

template <>
__device__ RGBF
  direct_lighting_geometry_sample<MATERIAL_VOLUME>(MaterialContextVolume ctx, const ushort2 pixel, DirectLightingShadowTask& task) {
  const LightSampleResult<MATERIAL_VOLUME> sample = light_sample(ctx, pixel);

  const RGBF shadowed_result = bridges_sample_apply_shadowing(ctx, sample, pixel);

  ////////////////////////////////////////////////////////////////////
  // Create dummy shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_ABORT;
  task.origin       = ctx.position;
  task.ray          = get_vector(0.0f, 0.0f, 1.0f);
  task.target_light = triangle_handle_get(INSTANCE_ID_INVALID, 0);
  task.limit        = 0.0f;

  return shadowed_result;
}

////////////////////////////////////////////////////////////////////
// Ambient
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF direct_lighting_ambient_sample(MaterialContext<TYPE> ctx, const ushort2 index, DirectLightingShadowTask& task) {
  ////////////////////////////////////////////////////////////////////
  // Early exit
  ////////////////////////////////////////////////////////////////////
  if (device.state.depth < device.settings.max_ray_depth || !device.sky.ambient_sampling) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  // We don't support compute based sky due to register/performance reasons and because
  // we would have to include clouds then aswell.
  if (device.sky.mode == LUMINARY_SKY_MODE_DEFAULT) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample ray
  ////////////////////////////////////////////////////////////////////

  const BSDFSampleInfo<TYPE> bounce_info = bsdf_sample(ctx, index);

  const vec3 task_origin = shift_origin_vector(ctx.position, ctx.V, bounce_info.ray, bounce_info.is_transparent_pass);

  RGBF light_color = sky_color_no_compute(bounce_info.ray, 0);
  light_color      = mul_color(light_color, bounce_info.weight);

  if (color_importance(light_color) == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task.origin       = task_origin;
  task.ray          = bounce_info.ray;
  task.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task.limit        = FLT_MAX;

  return light_color;
}

////////////////////////////////////////////////////////////////////
// Main function
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF direct_lighting_sun(const MaterialContext<TYPE> ctx, const ushort2 index) {
  ////////////////////////////////////////////////////////////////////
  // Decide on method
  ////////////////////////////////////////////////////////////////////

  const vec3 sky_pos     = world_to_sky_transform(ctx.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.ocean.active) {
    is_underwater  = ocean_get_relative_height(ctx.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;
    sample_direct  = !is_underwater;
    sample_caustic = (device.ocean.caustics_active && TYPE == MATERIAL_GEOMETRY) || is_underwater;
  }

  // Sun is not present
  if (device.sky.mode == LUMINARY_SKY_MODE_CONSTANT_COLOR || !sun_visible) {
    sample_direct  = false;
    sample_caustic = false;
  }

  ////////////////////////////////////////////////////////////////////
  // Execute sun lighting
  ////////////////////////////////////////////////////////////////////

  DirectLightingShadowTask task_caustic0;
  task_caustic0.trace_status = OPTIX_TRACE_STATUS_ABORT;
  DirectLightingShadowTask task_caustic1;
  task_caustic1.trace_status = OPTIX_TRACE_STATUS_ABORT;
  RGBF light_caustic         = splat_color(0.0f);

  if (sample_caustic) {
    light_caustic = direct_lighting_sun_caustic(ctx, index, sky_pos, is_underwater, task_caustic0, task_caustic1);
  }

  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(task_caustic0));
  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(task_caustic1));

  DirectLightingShadowTask task_direct;
  task_direct.trace_status = OPTIX_TRACE_STATUS_ABORT;
  RGBF light_direct        = splat_color(0.0f);

  if (sample_direct) {
    light_direct = direct_lighting_sun_direct(ctx, index, sky_pos, task_direct);
  }

  light_direct = mul_color(light_direct, direct_lighting_sun_shadowing(task_direct));

  return add_color(light_caustic, light_direct);
}

template <MaterialType TYPE>
__device__ RGBF direct_lighting_geometry(const MaterialContext<TYPE> ctx, const ushort2 index) {
  // if (is_center_pixel(index) == false)
  //   return get_color(0.0f, 0.0f, 0.0f);

  DirectLightingShadowTask task;
  RGBF light = direct_lighting_geometry_sample<TYPE>(ctx, index, task);

  light = mul_color(light, direct_lighting_shadowing(task));

  return light;
}

template <>
__device__ RGBF direct_lighting_geometry<MATERIAL_VOLUME>(const MaterialContextVolume ctx, const ushort2 index) {
  DirectLightingShadowTask task;
  RGBF light = direct_lighting_geometry_sample(ctx, index, task);

  return light;
}

template <MaterialType TYPE>
__device__ RGBF direct_lighting_ambient(const MaterialContext<TYPE> ctx, const ushort2 index) {
  DirectLightingShadowTask task;
  RGBF light = direct_lighting_ambient_sample(ctx, index, task);

  light = mul_color(light, direct_lighting_sun_shadowing(task));

  return light;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_DIRECT_LIGHTING_H */
