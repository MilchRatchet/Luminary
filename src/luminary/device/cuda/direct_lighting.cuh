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
  VolumeType volume_type;
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

__device__ RGBF direct_lighting_sun_shadowing(const TriangleHandle handle, const DirectLightingShadowTask task) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_sun_shadowing(handle, task.origin, task.ray, task.limit, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.volume_type, task.origin, task.ray, task.limit));

  return visibility;
#else  /* !DIRECT_LIGHTING_NO_SHADOW */
  return splat_color(1.0f);
#endif /* DIRECT_LIGHTING_NO_SHADOW */
}

__device__ RGBF direct_lighting_shadowing(const TriangleHandle handle, const DirectLightingShadowTask task) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_geometry_shadowing(handle, task.origin, task.ray, task.limit, task.target_light, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.volume_type, task.origin, task.ray, task.limit));

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
  const vec3 dir_bsdf = bsdf_sample_for_sun(ctx, index, bsdf_sample_is_refraction, bsdf_sample_is_valid);

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

  const float2 random = random_2D(MaterialContext<TYPE>::RANDOM_DL_SUN::RAY, index);

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
  const float mis_weight_bsdf        = solid_angle / (bsdf_sample_for_sun_pdf(ctx, dir_bsdf) * solid_angle + 1.0f);
  const float mis_weight_solid_angle = solid_angle / (bsdf_sample_for_sun_pdf(ctx, dir_solid_angle) * solid_angle + 1.0f);

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
  if (random_1D(MaterialContext<TYPE>::RANDOM_DL_SUN::RESAMPLING, index) * sum_weights < weight_bsdf) {
    dir         = dir_bsdf;
    target_pdf  = target_pdf_bsdf;
    light_color = light_bsdf;
  }
  else {
    dir         = dir_solid_angle;
    target_pdf  = target_pdf_solid_angle;
    light_color = light_solid_angle;
  }

  light_color = scale_color(light_color, sum_weights / target_pdf);

  if (target_pdf == 0.0f) {
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
  task.origin       = ctx.position;
  task.ray          = dir;
  task.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task.limit        = FLT_MAX;
  task.volume_type  = ctx.volume_type;

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
  const float2 sun_dir_random                  = random_2D(MaterialContext<TYPE>::RANDOM_DL_SUN::CAUSTIC_SUN_RAY, index);
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

    const float resampling_random = random_1D(MaterialContext<TYPE>::RANDOM_DL_SUN::CAUSTIC_RESAMPLING, index);

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

  if (sum_connection_weight == 0.0f) {
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled connection vertex
  ////////////////////////////////////////////////////////////////////

  vec3 pos_to_ocean = sub_vector(connection_point, ctx.position);

  const float dist = get_length(pos_to_ocean);
  const vec3 dir   = normalize_vector(pos_to_ocean);

  RGBF light_color = sky_get_sun_color(world_to_sky_transform(connection_point), sun_dir);

  bool is_refraction;
  const RGBF bsdf_value = bsdf_evaluate(ctx, dir, BSDF_SAMPLING_GENERAL, is_refraction, connection_weight);
  light_color           = mul_color(light_color, bsdf_value);

  const vec3 ocean_normal =
    (sampling_domain.fast_path) ? get_vector(0.0f, 1.0f, 0.0f) : ocean_get_normal(connection_point, OCEAN_ITERATIONS_NORMAL_CAUSTICS);
  const vec3 normal = scale_vector(ocean_normal, (is_underwater) ? -1.0f : 1.0f);

  bool total_reflection;
  const vec3 refraction_dir          = refract_vector(scale_vector(dir, -1.0f), normal, sampling_domain.ior, total_reflection);
  const float reflection_coefficient = ocean_reflection_coefficient(normal, dir, refraction_dir, sampling_domain.ior);

  light_color = scale_color(light_color, (is_underwater) ? 1.0f - reflection_coefficient : reflection_coefficient);

  if (color_importance(light_color) == 0.0f) {
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task0.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task0.origin       = ctx.position;
  task0.ray          = dir;
  task0.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task0.limit        = dist;
  task0.volume_type  = ctx.volume_type;

  task1.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task1.origin       = connection_point;
  task1.ray          = sun_dir;
  task1.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task1.limit        = FLT_MAX;
  task1.volume_type  = (device.fog.active) ? VOLUME_TYPE_FOG : VOLUME_TYPE_NONE;

  return light_color;
}

////////////////////////////////////////////////////////////////////
// Geometry
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF direct_lighting_geometry_sample(MaterialContext<TYPE> ctx, const ushort2 pixel, DirectLightingShadowTask& task) {
  const LightSampleResult<TYPE> sample = light_sample(ctx, pixel);

  const bool valid_sample = sample.handle.instance_id != INSTANCE_ID_INVALID;

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = (valid_sample) ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  task.origin       = ctx.position;
  task.ray          = sample.ray;
  task.target_light = sample.handle;
  task.limit        = sample.dist;
  task.volume_type  = ctx.volume_type;

  return (valid_sample) ? sample.light_color : splat_color(0.0f);
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
  task.origin       = get_vector(0.0f, 0.0f, 0.0f);
  task.ray          = get_vector(0.0f, 0.0f, 1.0f);
  task.target_light = triangle_handle_get(INSTANCE_ID_INVALID, 0);
  task.limit        = 0.0f;
  task.volume_type  = VOLUME_TYPE_NONE;

  return shadowed_result;
}

////////////////////////////////////////////////////////////////////
// Ambient
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ RGBF direct_lighting_ambient_sample(
  MaterialContext<TYPE> ctx, const ushort2 index, DirectLightingShadowTask& task1, DirectLightingShadowTask& task2) {
  ////////////////////////////////////////////////////////////////////
  // Early exit
  ////////////////////////////////////////////////////////////////////

  // We don't support compute based sky due to register/performance reasons and because
  // we would have to include clouds then aswell.
  if (device.sky.mode == LUMINARY_SKY_MODE_DEFAULT) {
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Sample ray
  ////////////////////////////////////////////////////////////////////

  const BSDFSampleInfo<TYPE> bounce_info = bsdf_sample<MaterialContext<TYPE>::RANDOM_DL_AMBIENT>(ctx, index);

  // No ocean caustics reflection sampling for ambient DL
  if ((ctx.volume_type != VOLUME_TYPE_OCEAN) && device.ocean.active && (bounce_info.ray.y < 0.0f)) {
    return splat_color(0.0f);
  }

  RGBF light_color = sky_color_no_compute(ctx.position, bounce_info.ray, 0);
  light_color      = mul_color(light_color, bounce_info.weight);

  if (color_importance(light_color) == 0.0f) {
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task1.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task1.origin       = ctx.position;
  task1.ray          = bounce_info.ray;
  task1.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task1.limit        = FLT_MAX;
  task1.volume_type  = ctx.volume_type;

  ////////////////////////////////////////////////////////////////////
  // Ocean caustic refraction task
  ////////////////////////////////////////////////////////////////////

  // Add a second ray for ocean refraction
  if ((ctx.volume_type == VOLUME_TYPE_OCEAN) && (bounce_info.ray.y > 0.0f)) {
    const float ocean_intersection_dist = ocean_intersection_solver(ctx.position, bounce_info.ray, 0.0f, FLT_MAX);

    const vec3 ocean_intersection = add_vector(ctx.position, scale_vector(bounce_info.ray, ocean_intersection_dist));

    // Ocean normal points up, we come from below, so flip it
    const vec3 ocean_normal = scale_vector(ocean_get_normal(ocean_intersection), -1.0f);
    const vec3 ocean_V      = scale_vector(bounce_info.ray, -1.0f);

    bool total_reflection;
    const vec3 refraction = refract_vector(ocean_V, ocean_normal, device.ocean.refractive_index, total_reflection);

    if (total_reflection) {
      task1.trace_status = OPTIX_TRACE_STATUS_ABORT;
      return splat_color(0.0f);
    }

    const float fresnel_term = bsdf_fresnel(ocean_normal, ocean_V, refraction, device.ocean.refractive_index);

    light_color = scale_color(light_color, 1.0f - fresnel_term);

    task1.limit = ocean_intersection_dist;

    task2.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
    task2.origin       = ocean_intersection;
    task2.ray          = refraction;
    task2.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
    task2.limit        = FLT_MAX;
    task2.volume_type  = (device.fog.active) ? VOLUME_TYPE_FOG : VOLUME_TYPE_NONE;
  }

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

  const vec3 sky_pos = world_to_sky_transform(ctx.position);

  const bool sun_below_horizon = sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const bool inside_earth      = get_length(sky_pos) < SKY_EARTH_RADIUS;
  const bool sun_visible       = (sun_below_horizon == false) && (inside_earth == false);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.ocean.active) {
    is_underwater  = ctx.volume_type == VOLUME_TYPE_OCEAN;
    sample_direct  = !is_underwater;
    sample_caustic = (device.ocean.caustics_active && TYPE == MATERIAL_GEOMETRY) || is_underwater;
  }

  // Sun is not present
  if (device.sky.mode == LUMINARY_SKY_MODE_CONSTANT_COLOR || sun_visible == false) {
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

  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(ctx.get_handle(), task_caustic0));
  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(ctx.get_handle(), task_caustic1));

  DirectLightingShadowTask task_direct;
  task_direct.trace_status = OPTIX_TRACE_STATUS_ABORT;
  RGBF light_direct        = splat_color(0.0f);

  if (sample_direct) {
    light_direct = direct_lighting_sun_direct(ctx, index, sky_pos, task_direct);
  }

  light_direct = mul_color(light_direct, direct_lighting_sun_shadowing(ctx.get_handle(), task_direct));

  return add_color(light_caustic, light_direct);
}

template <MaterialType TYPE>
__device__ RGBF direct_lighting_geometry(const MaterialContext<TYPE> ctx, const ushort2 index) {
  DirectLightingShadowTask task;
  RGBF light = direct_lighting_geometry_sample<TYPE>(ctx, index, task);

  light = mul_color(light, direct_lighting_shadowing(ctx.get_handle(), task));

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
  DirectLightingShadowTask task1;
  task1.trace_status = OPTIX_TRACE_STATUS_ABORT;
  DirectLightingShadowTask task2;
  task2.trace_status = OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;
  RGBF light         = direct_lighting_ambient_sample(ctx, index, task1, task2);

  light = mul_color(light, direct_lighting_sun_shadowing(ctx.get_handle(), task1));
  light = mul_color(light, direct_lighting_sun_shadowing(ctx.get_handle(), task2));

  return light;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_DIRECT_LIGHTING_H */
