#ifndef CU_LUMINARY_DIRECT_LIGHTING_H
#define CU_LUMINARY_DIRECT_LIGHTING_H

#ifdef OPTIX_KERNEL

#include "bridges.cuh"
#include "bsdf.cuh"
#include "caustics.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "random.cuh"
#include "ris.cuh"
#include "sky.cuh"
#include "utils.cuh"

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
// BSDF Sampling
////////////////////////////////////////////////////////////////////

__device__ DirectLightingBSDFSample direct_lighting_get_bsdf_sample(GBufferData data, const ushort2 index, const uint32_t id) {
  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const QuasiRandomTarget bsdf_target = (QuasiRandomTarget) (QUASI_RANDOM_TARGET_LIGHT_BSDF + 2 * id);
  const vec3 bsdf_dir                 = bsdf_sample_for_light(data, index, bsdf_target, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  vec3 position;
  float shift;

  shift    = bsdf_sample_is_refraction ? -eps : eps;
  position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

  // The compiler has issues with conditional optixTrace, hence we disable them using a negative max dist.
  const OptixTraceStatus light_trace_status = (bsdf_sample_is_valid) ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;

  OptixKernelFunctionLightBSDFTracePayload payload;
  payload.triangle_id = HIT_TYPE_LIGHT_BSDF_HINT;

  optixKernelFunctionLightBSDFTrace(
    device.optix_bvh_light, position, bsdf_dir, 0.0f, FLT_MAX, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    light_trace_status, payload);

  DirectLightingBSDFSample sample;
  sample.light_id      = payload.triangle_id;
  sample.ray           = bsdf_dir;
  sample.is_refraction = bsdf_sample_is_refraction;

  return sample;
}

////////////////////////////////////////////////////////////////////
// Shadowing
////////////////////////////////////////////////////////////////////

__device__ RGBF direct_lighting_sun_shadowing(const DirectLightingShadowTask task) {
  RGBF visibility = optix_sun_shadowing(task.origin, task.ray, task.limit, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.origin, task.ray, task.limit));

  return visibility;
}

__device__ RGBF direct_lighting_shadowing(const DirectLightingShadowTask task) {
  RGBF visibility = optix_geometry_shadowing(task.origin, task.ray, task.limit, task.target_light, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.origin, task.ray, task.limit));

  return visibility;
}

////////////////////////////////////////////////////////////////////
// Sun
////////////////////////////////////////////////////////////////////

__device__ RGBF direct_lighting_sun_direct(GBufferData data, const ushort2 index, const vec3 sky_pos, DirectLightingShadowTask& task) {
  ////////////////////////////////////////////////////////////////////
  // Sample a direction using BSDF importance sampling
  ////////////////////////////////////////////////////////////////////

  bool bsdf_sample_is_refraction, bsdf_sample_is_valid;
  const vec3 dir_bsdf =
    bsdf_sample_for_light(data, index, QUASI_RANDOM_TARGET_LIGHT_SUN_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  RGBF light_bsdf         = get_color(0.0f, 0.0f, 0.0f);
  bool is_refraction_bsdf = false;
  if (sphere_ray_hit(dir_bsdf, sky_pos, device.sky.sun_pos, SKY_SUN_RADIUS)) {
    light_bsdf = sky_get_sun_color(sky_pos, dir_bsdf);

    const RGBF value_bsdf = bsdf_evaluate(data, dir_bsdf, BSDF_SAMPLING_GENERAL, is_refraction_bsdf, 1.0f);
    light_bsdf            = mul_color(light_bsdf, value_bsdf);
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

  if (sum_weights == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

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

  if (target_pdf == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
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
  task.origin       = shift_origin_vector(data.position, data.V, dir, is_refraction);
  task.ray          = dir;
  task.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task.limit        = FLT_MAX;

  return light_color;
}

__device__ RGBF direct_lighting_sun_caustic(
  GBufferData data, const ushort2 index, const vec3 sky_pos, const bool is_underwater, DirectLightingShadowTask& task0,
  DirectLightingShadowTask& task1) {
  ////////////////////////////////////////////////////////////////////
  // Sample a caustic connection vertex using RIS
  ////////////////////////////////////////////////////////////////////

  float solid_angle;
  const float2 sun_dir_random                  = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAUSTIC_SUN_DIR, index);
  const vec3 sun_dir                           = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, sun_dir_random, solid_angle);
  const CausticsSamplingDomain sampling_domain = caustics_get_domain(data, sun_dir, is_underwater);

  if (sampling_domain.valid == false)
    return splat_color(0.0f);

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
    const uint32_t num_samples = device.ocean.caustics_ris_sample_count + 1;

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

  const float dist = get_length(pos_to_ocean);
  const vec3 dir   = normalize_vector(pos_to_ocean);

  RGBF light_color = sky_get_sun_color(world_to_sky_transform(connection_point), sun_dir);

  if (sum_connection_weight == 0.0f) {
    return splat_color(0.0f);
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
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
    return splat_color(0.0f);
  }

  const vec3 position = shift_origin_vector(data.position, data.V, dir, is_refraction);

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task0.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task0.origin       = shift_origin_vector(data.position, data.V, dir, is_refraction);
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

#ifndef VOLUME_KERNEL
__device__ RGBF direct_lighting_geometry_single(
  GBufferData data, const ushort2 index, const uint32_t id, const DirectLightingBSDFSample bsdf_sample, DirectLightingShadowTask& task) {
  if (LIGHTS_ARE_PRESENT == false) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Resample the BSDF direction with NEE based directions
  ////////////////////////////////////////////////////////////////////

  vec3 dir;
  RGBF light_color;
  float dist;
  bool is_refraction;
  const TriangleHandle light_handle = ris_sample_light(
    data, index, id, bsdf_sample.light_id, bsdf_sample.ray, bsdf_sample.is_refraction, dir, light_color, dist, is_refraction);

  if (color_importance(light_color) == 0.0f || light_handle.instance_id == LIGHT_ID_NONE) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  // Transparent pass through rays are not allowed.
  if (bsdf_is_pass_through_ray(is_refraction, data.ior_in, data.ior_out)) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task.origin       = shift_origin_vector(data.position, data.V, dir, is_refraction);
  task.ray          = dir;
  task.target_light = light_handle;
  task.limit        = dist;

  return light_color;
}
#endif

////////////////////////////////////////////////////////////////////
// Ambient
////////////////////////////////////////////////////////////////////

__device__ RGBF direct_lighting_ambient_sample(GBufferData data, const ushort2 index, DirectLightingShadowTask& task) {
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

#ifndef PHASE_KERNEL
  BSDFSampleInfo bounce_info;
  const vec3 ray = bsdf_sample(data, index, bounce_info);

  RGBF light_color = bounce_info.weight;

  const vec3 task_origin = shift_origin_vector(data.position, data.V, ray, bounce_info.is_transparent_pass);

  if (color_importance(light_color) == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }
#else
  const vec3 ray         = bsdf_sample_volume(data, index);
  const vec3 task_origin = data.position;
  RGBF light_color       = splat_color(1.0f);
#endif

  light_color = mul_color(light_color, sky_color_no_compute(ray));

  if (color_importance(light_color) == 0.0f) {
    task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    return splat_color(0.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Create shadow task
  ////////////////////////////////////////////////////////////////////

  task.trace_status = OPTIX_TRACE_STATUS_EXECUTE;
  task.origin       = task_origin;
  task.ray          = ray;
  task.target_light = triangle_handle_get(HIT_TYPE_SKY, 0);
  task.limit        = FLT_MAX;

  return light_color;
}

////////////////////////////////////////////////////////////////////
// Main function
////////////////////////////////////////////////////////////////////

__device__ RGBF direct_lighting_sun(const GBufferData data, const ushort2 index) {
  ////////////////////////////////////////////////////////////////////
  // Decide on method
  ////////////////////////////////////////////////////////////////////

  const vec3 sky_pos     = world_to_sky_transform(data.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.ocean.active && data.instance_id != HIT_TYPE_OCEAN) {
    is_underwater  = ocean_get_relative_height(data.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;
    sample_direct  = !is_underwater;
    sample_caustic = device.ocean.caustics_active || is_underwater;
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
    light_caustic = direct_lighting_sun_caustic(data, index, sky_pos, is_underwater, task_caustic0, task_caustic1);
  }

  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(task_caustic0));
  light_caustic = mul_color(light_caustic, direct_lighting_sun_shadowing(task_caustic1));

  DirectLightingShadowTask task_direct;
  task_direct.trace_status = OPTIX_TRACE_STATUS_ABORT;
  RGBF light_direct        = splat_color(0.0f);

  if (sample_direct) {
    light_direct = direct_lighting_sun_direct(data, index, sky_pos, task_direct);
  }

  light_direct = mul_color(light_direct, direct_lighting_sun_shadowing(task_direct));

  return add_color(light_caustic, light_direct);
}

#ifdef PHASE_KERNEL
__device__ RGBF direct_lighting_sun_phase(const GBufferData data, const ushort2 index) {
  ////////////////////////////////////////////////////////////////////
  // Decide on method
  ////////////////////////////////////////////////////////////////////

  const vec3 sky_pos     = world_to_sky_transform(data.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  bool sample_direct  = true;
  bool sample_caustic = false;
  bool is_underwater  = false;

  if (device.ocean.active && data.instance_id != HIT_TYPE_OCEAN) {
    is_underwater  = ocean_get_relative_height(data.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;
    sample_direct  = !is_underwater;
    sample_caustic = is_underwater;
  }

  // Sun is not present
  if (device.sky.mode == LUMINARY_SKY_MODE_CONSTANT_COLOR || !sun_visible) {
    sample_direct  = false;
    sample_caustic = false;
  }

  ////////////////////////////////////////////////////////////////////
  // Execute sun lighting
  ////////////////////////////////////////////////////////////////////

  DirectLightingShadowTask task0;
  task0.trace_status = OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;
  DirectLightingShadowTask task1;
  task1.trace_status = OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;
  RGBF light         = splat_color(0.0f);

  if (sample_caustic) {
    light = direct_lighting_sun_caustic(data, index, sky_pos, is_underwater, task0, task1);
  }

  if (sample_direct) {
    light = direct_lighting_sun_direct(data, index, sky_pos, task0);
  }

  light = mul_color(light, direct_lighting_sun_shadowing(task0));
  light = mul_color(light, direct_lighting_sun_shadowing(task1));

  return light;
}
#endif

#ifndef VOLUME_KERNEL
__device__ RGBF direct_lighting_geometry(const GBufferData data, const ushort2 index) {
  RGBF sum_light = splat_color(0.0f);

  for (uint32_t j = 0; j < device.settings.light_num_rays; j++) {
    const DirectLightingBSDFSample bsdf_sample = direct_lighting_get_bsdf_sample(data, index, j);

    DirectLightingShadowTask task;
    RGBF light = direct_lighting_geometry_single(data, index, j, bsdf_sample, task);

    light = mul_color(light, direct_lighting_shadowing(task));

    sum_light = add_color(sum_light, light);
  }

  return scale_color(sum_light, 1.0f / device.settings.light_num_rays);
}

#endif

__device__ RGBF direct_lighting_geometry_bridges(const DeviceTask task, const VolumeType volume_type, const VolumeDescriptor volume) {
  RGBF light = splat_color(0.0f);

#ifdef VOLUME_KERNEL
  if (((task.state & STATE_FLAG_DELTA_PATH) != 0) && (device.ocean.triangle_light_contribution || volume_type != VOLUME_TYPE_OCEAN)) {
    light = bridges_sample(task, volume);
  }
#endif

  return light;
}

__device__ RGBF direct_lighting_ambient(const GBufferData data, const ushort2 index) {
  DirectLightingShadowTask task;
  RGBF light = direct_lighting_ambient_sample(data, index, task);

  light = mul_color(light, direct_lighting_sun_shadowing(task));

  return light;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_DIRECT_LIGHTING_H */
