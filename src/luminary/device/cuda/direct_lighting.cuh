#ifndef CU_LUMINARY_DIRECT_LIGHTING_H
#define CU_LUMINARY_DIRECT_LIGHTING_H

#include "bsdf.cuh"
#include "caustics.cuh"
#include "light.cuh"
#include "light_bridges.cuh"
#include "light_shadow.cuh"
#include "material.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "sky.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Sun
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ DeviceTaskDirectLightSun direct_lighting_sun_direct(MaterialContext<TYPE> ctx, const ushort2 index, const vec3 sky_pos) {
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
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
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
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
  }

  if (color_importance(light_color) == 0.0f) {
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
  }

  ////////////////////////////////////////////////////////////////////
  // Volume transmittance
  ////////////////////////////////////////////////////////////////////

  light_color = mul_color(light_color, volume_integrate_transmittance(ctx.volume_type, ctx.position, dir, FLT_MAX));

  ////////////////////////////////////////////////////////////////////
  // Create task
  ////////////////////////////////////////////////////////////////////

  DeviceTaskDirectLightSun task;
  task.light_color = record_pack(light_color);
  task.ray         = ray_pack(dir);

  return task;
}

template <MaterialType TYPE>
__device__ DeviceTaskDirectLightSun direct_lighting_sun_caustic(MaterialContext<TYPE> ctx, const ushort2 index, const vec3 sky_pos) {
  // Caustics are only for underwater starting with v1.2.0
  constexpr bool IS_UNDERWATER = true;

  ////////////////////////////////////////////////////////////////////
  // Sample a caustic connection vertex using RIS
  ////////////////////////////////////////////////////////////////////

  float solid_angle;
  const float2 sun_dir_random                  = random_2D(MaterialContext<TYPE>::RANDOM_DL_SUN::CAUSTIC_SUN_RAY, index);
  const vec3 sun_dir                           = sample_sphere(device.sky.sun_pos, SKY_SUN_RADIUS, sky_pos, sun_dir_random, solid_angle);
  const CausticsSamplingDomain sampling_domain = caustics_get_domain(ctx, sun_dir, IS_UNDERWATER);

  if (sampling_domain.valid == false) {
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
  }

  vec3 connection_point;
  float sum_connection_weight = 0.0f;
  float connection_weight;

  if (sampling_domain.fast_path) {
    vec3 sample_point;
    float sample_weight;
    caustics_find_connection_point(ctx, index, sampling_domain, IS_UNDERWATER, 0, 1, sample_point, sample_weight);

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
        caustics_find_connection_point(ctx, index, sampling_domain, IS_UNDERWATER, current_index, num_samples, sample_point, sample_weight);

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

    if (IS_UNDERWATER) {
      // ... and why not apply it again, we are just sampling some extra importance clearly. And a 2.0f for good measure.
      connection_weight *= device.ocean.refractive_index * device.ocean.refractive_index * 2.0f;
    }
  }

  if (sum_connection_weight == 0.0f) {
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
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
  const vec3 normal = scale_vector(ocean_normal, (IS_UNDERWATER) ? -1.0f : 1.0f);

  bool total_reflection;
  const vec3 refraction_dir          = refract_vector(scale_vector(dir, -1.0f), normal, sampling_domain.ior, total_reflection);
  const float reflection_coefficient = ocean_reflection_coefficient(normal, dir, refraction_dir, sampling_domain.ior);

  light_color = scale_color(light_color, (IS_UNDERWATER) ? 1.0f - reflection_coefficient : reflection_coefficient);

  if (color_importance(light_color) == 0.0f) {
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;
    return task;
  }

  ////////////////////////////////////////////////////////////////////
  // Volume transmittance
  ////////////////////////////////////////////////////////////////////

  const VolumeType second_volume = (device.fog.active) ? VOLUME_TYPE_FOG : VOLUME_TYPE_NONE;

  light_color = mul_color(light_color, volume_integrate_transmittance(ctx.volume_type, ctx.position, dir, dist));
  light_color = mul_color(light_color, volume_integrate_transmittance(second_volume, connection_point, sun_dir, FLT_MAX));

  ////////////////////////////////////////////////////////////////////
  // Create task
  ////////////////////////////////////////////////////////////////////

  DeviceTaskDirectLightSun task;
  task.light_color = record_pack(light_color);
  task.ray         = ray_pack(dir);

  return task;
}

////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////

__device__ bool direct_lighting_geometry_is_allowed(const DeviceTask& task) {
  bool allow_geometry_lighting = true;

  allow_geometry_lighting &= LIGHTS_ARE_PRESENT == true;
  allow_geometry_lighting &= (task.state & STATE_FLAG_VOLUME_SCATTERED) == 0;

  return allow_geometry_lighting;
}

__device__ bool direct_lighting_sun_is_allowed(const DeviceTask& task) {
  bool allow_sun_lighting = true;

  allow_sun_lighting &= device.sky.mode != LUMINARY_SKY_MODE_CONSTANT_COLOR;

  return allow_sun_lighting;
}

__device__ bool direct_lighting_sun_is_allowed(const MaterialContextVolume& ctx) {
  bool allow_sun_lighting = true;

  allow_sun_lighting &= device.sky.mode != LUMINARY_SKY_MODE_CONSTANT_COLOR;
  allow_sun_lighting &= ctx.volume_type != VOLUME_TYPE_NONE;

  return allow_sun_lighting;
}

__device__ bool direct_lighting_ambient_is_allowed(const DeviceTask& task) {
  bool allow_ambient_lighting = true;

  allow_ambient_lighting &= device.sky.mode != LUMINARY_SKY_MODE_DEFAULT;

  return allow_ambient_lighting;
}

__device__ bool direct_lighting_ambient_is_allowed(const MaterialContextVolume& ctx) {
  bool allow_ambient_lighting = true;

  allow_ambient_lighting &= device.sky.mode != LUMINARY_SKY_MODE_DEFAULT;
  allow_ambient_lighting &= ctx.volume_type != VOLUME_TYPE_NONE;

  return allow_ambient_lighting;
}

__device__ bool direct_lighting_bridges_is_allowed(const MaterialContextVolume& ctx) {
  bool allow_geometry_lighting = true;

  allow_geometry_lighting &= LIGHTS_ARE_PRESENT == true;
  allow_geometry_lighting &= (ctx.state & STATE_FLAG_DELTA_PATH) != 0;
  allow_geometry_lighting &= (ctx.state & STATE_FLAG_VOLUME_SCATTERED) == 0;
  allow_geometry_lighting &= ctx.volume_type != VOLUME_TYPE_NONE;
  allow_geometry_lighting &= (ctx.volume_type != VOLUME_TYPE_OCEAN) || device.ocean.triangle_light_contribution;

  return allow_geometry_lighting;
}

////////////////////////////////////////////////////////////////////
// Main functions
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ DeviceTaskDirectLightGeo direct_lighting_geometry_create_task(const MaterialContext<TYPE>& ctx, const ushort2 pixel) {
  LightSampleResult<TYPE> sample = light_sample(ctx, pixel);

  ////////////////////////////////////////////////////////////////////
  // Volume transmittance
  ////////////////////////////////////////////////////////////////////

  if constexpr (TYPE != MATERIAL_VOLUME) {
    const RGBF transmittance = volume_integrate_transmittance(ctx.volume_type, ctx.position, sample.ray, sample.dist);
    sample.light_color       = mul_color(sample.light_color, transmittance);
  }

  ////////////////////////////////////////////////////////////////////
  // Create task
  ////////////////////////////////////////////////////////////////////

  DeviceTaskDirectLightGeo task;
  task.light_id    = sample.light_id;
  task.light_color = sample.light_color;
  task.ray         = sample.ray;
  task.dist        = sample.dist;

  return task;
}

template <MaterialType TYPE>
__device__ DeviceTaskDirectLightSun direct_lighting_sun_create_task(const MaterialContext<TYPE> ctx, const ushort2 pixel) {
  const vec3 sky_pos = world_to_sky_transform(ctx.position);

  const bool sun_below_horizon = sph_ray_hit_p0(normalize_vector(sub_vector(device.sky.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const bool inside_earth      = get_length(sky_pos) < SKY_EARTH_RADIUS;
  const bool sun_visible       = (sun_below_horizon == false) && (inside_earth == false);

  // Sun is not present
  if (sun_visible == false) {
    DeviceTaskDirectLightSun task;
    task.light_color = PACKED_RECORD_BLACK;

    return task;
  }

  DeviceTaskDirectLightSun task;
  if (ctx.volume_type == VOLUME_TYPE_OCEAN) {
    task = direct_lighting_sun_caustic(ctx, pixel, sky_pos);
  }
  else {
    task = direct_lighting_sun_direct(ctx, pixel, sky_pos);
  }

  return task;
}

template <MaterialType TYPE>
__device__ DeviceTaskDirectLightAmbient
  direct_lighting_ambient_create_task(const MaterialContext<TYPE>& ctx, const BSDFSampleInfo<TYPE>& bounce_sample, const ushort2 index) {
  ////////////////////////////////////////////////////////////////////
  // Compute ambient color
  ////////////////////////////////////////////////////////////////////

  RGBF light_color = sky_color_no_compute(ctx.position, bounce_sample.ray, 0);
  light_color      = mul_color(light_color, bounce_sample.weight);

  ////////////////////////////////////////////////////////////////////
  // Create task
  ////////////////////////////////////////////////////////////////////

  DeviceTaskDirectLightAmbient task;
  task.light_color = record_pack(light_color);
  task.ray         = ray_pack(bounce_sample.ray);

  return task;
}

__device__ DeviceTaskDirectLightBridges direct_lighting_bridges_create_task(const MaterialContextVolume& ctx, const ushort2 index) {
  LightSampleResult<MATERIAL_VOLUME> sample = light_sample(ctx, index);

  ////////////////////////////////////////////////////////////////////
  // Create task
  ////////////////////////////////////////////////////////////////////

  DeviceTaskDirectLightBridges task;
  task.light_id    = sample.light_id;
  task.light_color = sample.light_color;
  task.rotation    = quaternion_pack(sample.rotation);
  task.scale       = sample.scale;
  task.seed        = sample.seed;

  return task;
}

#ifdef OPTIX_KERNEL

__device__ RGBF direct_lighting_geometry_evaluate_task(
  const DeviceTask& task, const DeviceTaskTrace& trace, const DeviceTaskDirectLightGeo& direct_light_task, const bool is_allowed) {
  const bool sample_is_valid = (direct_light_task.light_id != LIGHT_ID_INVALID) && is_allowed;

  ShadowTraceTask shadow_task;
  shadow_task.trace_status = sample_is_valid ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  shadow_task.origin       = task.origin;
  shadow_task.ray          = direct_light_task.ray;
  shadow_task.limit        = direct_light_task.dist;
  shadow_task.target_light =
    (sample_is_valid) ? device.ptrs.light_tree_tri_handle_map[direct_light_task.light_id] : TRIANGLE_HANDLE_INVALID;

  const RGBF visibility = shadow_evaluate(shadow_task, trace.handle);

  return mul_color(direct_light_task.light_color, visibility);
}

__device__ RGBF direct_lighting_sun_evaluate_task(
  const DeviceTask& task, const DeviceTaskTrace& trace, const DeviceTaskDirectLightSun& direct_light_task, const bool is_allowed) {
  const vec3 ray = ray_unpack(direct_light_task.ray);

  float limit = FLT_MAX;

  if (task.volume_id == VOLUME_TYPE_OCEAN && ray.y > 0.0f) {
    const float dist = (OCEAN_MAX_HEIGHT - task.origin.y) / ray.y;

    limit = (dist > 0.0f) ? dist : FLT_MAX;
  }

  const bool sample_is_valid = (direct_light_task.light_color.x != 0 || direct_light_task.light_color.y != 0) && is_allowed;

  ShadowTraceTask shadow_task;
  shadow_task.trace_status = sample_is_valid ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  shadow_task.origin       = task.origin;
  shadow_task.ray          = ray;
  shadow_task.limit        = limit;
  shadow_task.target_light = TRIANGLE_HANDLE_INVALID;

  RGBF visibility = shadow_evaluate_sun(shadow_task, trace.handle);

  RGBF light_color = record_unpack(direct_light_task.light_color);
  light_color      = mul_color(light_color, visibility);

  if (sample_is_valid && task.volume_id == VOLUME_TYPE_OCEAN && limit != FLT_MAX) {
    const vec3 ocean_pos = add_vector(task.origin, scale_vector(ray, limit));

    const bool fast_path = caustics_is_fast_path<MATERIAL_GEOMETRY>(task.state);

    // Ocean normal points up, we come from below, so flip it
    const vec3 ocean_normal = (fast_path) ? get_vector(0.0f, -1.0f, 0.0f) : scale_vector(ocean_get_normal(ocean_pos), -1.0f);
    const vec3 ocean_V      = scale_vector(ray, -1.0f);

    bool total_reflection;
    const vec3 refraction = refract_vector(ocean_V, ocean_normal, device.ocean.refractive_index, total_reflection);

    if (total_reflection) {
      shadow_task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    }

    shadow_task.origin       = ocean_pos;
    shadow_task.ray          = refraction;
    shadow_task.limit        = FLT_MAX;
    shadow_task.target_light = TRIANGLE_HANDLE_INVALID;
  }
  else {
    shadow_task.trace_status = OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;
  }

  light_color = mul_color(light_color, shadow_evaluate_sun(shadow_task, TRIANGLE_HANDLE_INVALID));

  return light_color;
}

__device__ RGBF direct_lighting_ambient_evaluate_task(
  const DeviceTask& task, const DeviceTaskTrace& trace, const DeviceTaskDirectLightAmbient& direct_light_task, const bool is_allowed) {
  const vec3 ray = ray_unpack(direct_light_task.ray);

  float limit = FLT_MAX;

  if (task.volume_id == VOLUME_TYPE_OCEAN && ray.y > 0.0f) {
    const float dist = (OCEAN_MAX_HEIGHT - task.origin.y) / ray.y;

    limit = (dist > 0.0f) ? dist : FLT_MAX;
  }

  const bool sample_is_valid = (direct_light_task.light_color.x != 0 || direct_light_task.light_color.y != 0) && is_allowed;

  ShadowTraceTask shadow_task;
  shadow_task.trace_status = sample_is_valid ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  shadow_task.origin       = task.origin;
  shadow_task.ray          = ray;
  shadow_task.limit        = limit;
  shadow_task.target_light = TRIANGLE_HANDLE_INVALID;

  RGBF visibility = shadow_evaluate_sun(shadow_task, trace.handle);

  RGBF light_color = record_unpack(direct_light_task.light_color);
  light_color      = mul_color(light_color, visibility);
  light_color      = mul_color(light_color, volume_integrate_transmittance((VolumeType) task.volume_id, task.origin, ray, limit));

  if (sample_is_valid && task.volume_id == VOLUME_TYPE_OCEAN && limit != FLT_MAX) {
    const vec3 ocean_pos = add_vector(task.origin, scale_vector(ray, limit));

    // Ocean normal points up, we come from below, so flip it
    const vec3 ocean_normal = scale_vector(ocean_get_normal(ocean_pos), -1.0f);
    const vec3 ocean_V      = scale_vector(ray, -1.0f);

    bool total_reflection;
    const vec3 refraction = refract_vector(ocean_V, ocean_normal, device.ocean.refractive_index, total_reflection);

    if (total_reflection) {
      shadow_task.trace_status = OPTIX_TRACE_STATUS_ABORT;
    }

    const VolumeType second_volume = (device.fog.active) ? VOLUME_TYPE_FOG : VOLUME_TYPE_NONE;

    const float fresnel_term = bsdf_fresnel(ocean_normal, ocean_V, refraction, device.ocean.refractive_index);

    light_color = scale_color(light_color, 1.0f - fresnel_term);
    light_color = mul_color(light_color, volume_integrate_transmittance(second_volume, ocean_pos, refraction, FLT_MAX));

    shadow_task.origin       = ocean_pos;
    shadow_task.ray          = refraction;
    shadow_task.limit        = FLT_MAX;
    shadow_task.target_light = TRIANGLE_HANDLE_INVALID;
  }
  else {
    shadow_task.trace_status = OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;
  }

  light_color = mul_color(light_color, shadow_evaluate_sun(shadow_task, TRIANGLE_HANDLE_INVALID));

  return light_color;
}

__device__ RGBF direct_lighting_bridges_evaluate_task(
  const MaterialContextVolume& ctx, const DeviceTaskDirectLightBridges& direct_light_task, const ushort2 index, const bool is_allowed) {
  const bool sample_is_valid = (direct_light_task.light_id != LIGHT_ID_INVALID) && is_allowed;

  return bridges_sample_apply_shadowing(ctx, direct_light_task, index, sample_is_valid);
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_DIRECT_LIGHTING_H */
