#ifndef CU_CLOUD_H
#define CU_CLOUD_H

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky_integration.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

//
// The code of this file was initially based on the cloud rendering in https://github.com/turanszkij/WickedEngine.
// It follows the basic ideas of using raymarching with noise based density.
// The noise texture creation is very similar to that found in the Wicked Engine.
// The clouds are divided into a 3 tropospheric layers: low, mid and top.
// Low level layer: Stratus, Stratocumulus and Cumulus clouds.
// Mid level layer: Altostratus and Altocumulus clouds.
// Top level layer: Cirrus, Cirrostratus and Cirrocumulus clouds.
//
// Note:  Cirrus clouds have a special shape that is not simple to reproduce using noise.
//        As a result, they are not currently implemented. Possible solutions include
//        computing a texture specifically for it but it is unclear how to achieve
//        a huge number of variations using that. As an alternative, I tested
//        Cirrostratus fibratus clouds. However, they don't mix well with other
//        top level clouds, causing a look that looks like interpolation artifacts.
//        Thus, the top layer currently consists of only cirrocumulus clouds and
//        cirrostratus nebulosus clouds.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Hil16]
// Sébastien Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", in Physically Based Shading in Theory and
// Practice course, SIGGRAPH 2016. https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
//

// [Sch22]
// Andrew Schneider, "Nubis, Evolved: Real-Time Volumetric Clouds for Skies, Environments, and VFX", in Advances in Real-Time Rendering in
// Games cource, SIGGRAPH 2022.
//

////////////////////////////////////////////////////////////////////
// Integrator functions
////////////////////////////////////////////////////////////////////

template <CloudLayerType LAYER_TYPE>
LUMINARY_FUNCTION float cloud_extinction(const vec3 origin, const vec3 ray) {
  const float iter_step = 1.0f / device.cloud.shadow_steps;

  float optical_depth = 0.0f;

  for (float i = 0.0f; i < 1.0f; i += iter_step) {
    float t0 = i;
    float t1 = i + iter_step;
    t0       = t0 * t0;
    t1       = t1 * t1;

    const float step_size = t1 - t0;
    const float reach     = t0 + step_size * 0.5f;

    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height<LAYER_TYPE>(pos);

    if (height > 1.0f || height < 0.0f)
      break;

    const CloudWeather weather = cloud_weather<LAYER_TYPE>(pos, height);

    if (cloud_significant_point<LAYER_TYPE>(height, weather)) {
      optical_depth -= cloud_density<LAYER_TYPE>(pos, height, weather, 0.0f) * step_size;
    }
  }

  optical_depth *= CLOUD_EXTINCTION_DENSITY;

  return expf(optical_depth);
}

/*
 * Returns an RGBAF where the RGB is the radiance and the A is the greyscale transmittance.
 */
template <CloudLayerType LAYER_TYPE>
LUMINARY_FUNCTION CloudRenderResult clouds_compute(vec3 origin, vec3 ray, float start, float dist, const PathID& path_id) {
  if (dist < 0.0f || start == FLT_MAX) {
    CloudRenderResult result;
    result.radiance      = spectrum_set1(0.0f);
    result.transmittance = 1.0f;
    result.hit_dist      = FLT_MAX;

    return result;
  }

  uint32_t step_count;
  if constexpr (LAYER_TYPE == CLOUD_LAYER_LOW) {
    const float span = device.cloud.low.height_max - device.cloud.low.height_min;
    dist             = fminf(8.0f * span, dist);
    step_count       = device.cloud.steps;
  }
  else if constexpr (LAYER_TYPE == CLOUD_LAYER_MID) {
    const float span = device.cloud.mid.height_max - device.cloud.mid.height_min;
    dist             = fminf(8.0f * span, dist);
    step_count       = device.cloud.steps / 4;
  }
  else if constexpr (LAYER_TYPE == CLOUD_LAYER_TOP) {
    const float span = device.cloud.top.height_max - device.cloud.top.height_min;
    dist             = fminf(8.0f * span, dist);
    step_count       = device.cloud.steps / 8;
  }

  start = fmaxf(0.0f, start);

  const float step_size = dist / step_count;
  float reach = start + step_size * remap(random_1D(RANDOM_TARGET_CLOUD_STEP_OFFSET + LAYER_TYPE, path_id), 0.0f, 1.0f, 0.1f, 0.9f);

  const float sun_solid_angle = sample_sphere_solid_angle(device.sky.sun_pos, SKY_SUN_RADIUS, add_vector(origin, scale_vector(ray, reach)));

  const JendersieEonParams params = jendersie_eon_phase_parameters(device.cloud.droplet_diameter);

  float transmittance           = 1.0f;
  float scattered_sun_light     = 0.0f;
  float scattered_ambient_light = 0.0f;
  float hit_dist                = FLT_MAX;
  bool hit                      = false;

  const float2 ambient_r        = random_2D(RANDOM_TARGET_CLOUD_DIR, path_id);
  const vec3 ambient_ray        = sample_ray_sphere(2.0f * ambient_r.x - 1.0f, ambient_r.y);
  const float ambient_cos_angle = dot_product(ray, ambient_ray);

  const float color_sampling_dist = random_1D(RANDOM_TARGET_CLOUD_COLOR_STEP + LAYER_TYPE, path_id);
  const vec3 color_sampling_pos   = add_vector(origin, scale_vector(ray, color_sampling_dist));

  const vec3 sun_ray        = normalize_vector(sub_vector(device.sky.sun_pos, color_sampling_pos));
  const float sun_cos_angle = dot_product(ray, sun_ray);

  const bool sun_visible = sph_ray_hit_p0(sun_ray, color_sampling_pos, SKY_EARTH_RADIUS) == false;

  for (uint32_t step_id = 0; step_id < step_count; step_id++) {
    if (reach >= start + dist)
      break;

    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height<LAYER_TYPE>(pos);

    if (height < 0.0f || height > 1.0f) {
      reach += step_size;
      continue;
    }

    const CloudWeather weather = cloud_weather<LAYER_TYPE>(pos, height);

    if (cloud_significant_point<LAYER_TYPE>(height, weather) == false) {
      reach += step_size;
      continue;
    }

    const float density = cloud_density<LAYER_TYPE>(pos, height, weather, 0.0f);

    if (density == 0.0f) {
      reach += step_size;
      continue;
    }

    if (hit == false) {
      hit_dist = reach;
      hit      = true;
    }

    float scattering            = density * CLOUD_SCATTERING_DENSITY;
    const float extinction_init = fmaxf(density * CLOUD_EXTINCTION_DENSITY, 0.0001f);
    float ambient_extinction    = cloud_extinction<LAYER_TYPE>(pos, ambient_ray);
    float sun_extinction        = (sun_visible) ? cloud_extinction<LAYER_TYPE>(pos, sun_ray) : 0.0f;
    float phase_factor          = 1.0f;

    const float step_trans_base = expf(-extinction_init * step_size);
    float step_trans_running    = step_trans_base;
    float rcp_extinction        = 1.0f / extinction_init;

    for (int oct = 0; oct < device.cloud.octaves; oct++) {
      const float sun_phase     = jendersie_eon_phase_function(sun_cos_angle, params, phase_factor);
      const float ambient_phase = jendersie_eon_phase_function(ambient_cos_angle, params, phase_factor);

      const float sun_color_i     = sun_extinction * sun_phase * sun_solid_angle * scattering;
      const float ambient_color_i = ambient_extinction * ambient_phase * 4.0f * PI * scattering;

      const float sun_s_term     = (sun_color_i - sun_color_i * step_trans_running) * rcp_extinction;
      const float ambient_s_term = (ambient_color_i - ambient_color_i * step_trans_running) * rcp_extinction;

      scattered_sun_light += sun_s_term * transmittance;
      scattered_ambient_light += ambient_s_term * transmittance;

      // Scale factors for the next octave
      scattering *= CLOUD_OCTAVE_SCATTERING_FACTOR;
      phase_factor *= CLOUD_OCTAVE_PHASE_FACTOR;
      step_trans_running = sqrtf(step_trans_running);
      rcp_extinction *= 2.0f;
      sun_extinction     = sqrtf(sun_extinction);
      ambient_extinction = sqrtf(ambient_extinction);
    }

    transmittance *= step_trans_base;

    if (transmittance < 0.001f) {
      transmittance = 0.0f;
      break;
    }

    reach += step_size;
  }

  const Spectrum sun_radiance     = (sun_visible) ? sky_get_sun_color_spectral(color_sampling_pos, sun_ray, false) : spectrum_set1(0.0f);
  const Spectrum ambient_radiance = sky_get_color_spectral<false>(color_sampling_pos, ambient_ray, FLT_MAX, device.sky.steps, path_id);

  CloudRenderResult result;
  result.radiance =
    spectrum_add(spectrum_scale(sun_radiance, scattered_sun_light), spectrum_scale(ambient_radiance, scattered_ambient_light));
  result.transmittance = transmittance;
  result.hit_dist      = hit_dist;

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION float clouds_render(
  vec3 origin, const vec3 ray, const float limit, const PathID& path_id, RGBF& color, RGBF& record, float& transmittance_cloud_only) {
  float2 intersections[3];
  CloudRenderResult results[3];

  intersections[0] = cloud_get_lowlayer_intersection(origin, ray, limit);
  results[0]       = clouds_compute<CLOUD_LAYER_LOW>(origin, ray, intersections[0].x, intersections[0].y, path_id);

  intersections[1] = cloud_get_midlayer_intersection(origin, ray, limit);
  results[1]       = clouds_compute<CLOUD_LAYER_MID>(origin, ray, intersections[1].x, intersections[1].y, path_id);

  intersections[2] = cloud_get_toplayer_intersection(origin, ray, limit);
  results[2]       = clouds_compute<CLOUD_LAYER_TOP>(origin, ray, intersections[2].x, intersections[2].y, path_id);

  const bool less01 = intersections[0].x <= intersections[1].x;
  const bool less02 = intersections[0].x <= intersections[2].x;
  const bool less12 = intersections[1].x <= intersections[2].x;

  int order[3];
  if (less01) {
    if (less02) {
      order[0] = 0;
      order[1] = (less12) ? 1 : 2;
      order[2] = (less12) ? 2 : 1;
    }
    else {
      order[0] = 2;
      order[1] = (less01) ? 0 : 1;
      order[2] = (less01) ? 1 : 0;
    }
  }
  else if (less12) {
    order[0] = 1;
    order[1] = (less02) ? 0 : 2;
    order[2] = (less02) ? 2 : 0;
  }
  else {
    order[0] = 2;
    order[1] = (less01) ? 0 : 1;
    order[2] = (less01) ? 1 : 0;
  }

  constexpr uint32_t NUM_STEPS = 128;

  float prev_start      = 0.0f;
  uint32_t prev_step_id = 0;

  SkyIntegrationParams params = sky_get_integration_params(origin, ray, limit, NUM_STEPS, path_id);

  Spectrum radiance      = spectrum_set1(0.0f);
  Spectrum transmittance = spectrum_set1(1.0f);

  for (int i = 0; i < 3; i++) {
    const CloudRenderResult result = results[order[i]];

    if (result.hit_dist == FLT_MAX)
      continue;

    if (device.cloud.atmosphere_scattering) {
      uint32_t next_step_id = sky_get_nearest_step_id(params, result.hit_dist);

      Spectrum segment_radiance = sky_compute_atmosphere<false, true>(transmittance, origin, ray, params, prev_step_id, next_step_id);
      radiance                  = spectrum_add(radiance, segment_radiance);

      prev_step_id = next_step_id;
    }

    radiance      = spectrum_add(radiance, spectrum_mul(result.radiance, transmittance));
    transmittance = spectrum_scale(transmittance, result.transmittance);
    transmittance_cloud_only *= result.transmittance;

    prev_start = result.hit_dist;
  }

  if (device.cloud.atmosphere_scattering)
    radiance = spectrum_add(radiance, sky_compute_atmosphere<true, true>(transmittance, origin, ray, params, prev_step_id, NUM_STEPS));

  color  = add_color(color, mul_color(sky_evaluate_radiance_from_spectrum(radiance), record));
  record = mul_color(record, sky_evaluate_transmittance_from_spectrum(transmittance));

  return prev_start;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void cloud_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(i, TASK_STATE_BUFFER_INDEX_PRESORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    float depth              = trace.depth;
    const float sky_max_dist = world_to_sky_scale(depth);
    vec3 sky_origin          = world_to_sky_transform(task.origin);

    RGBF record = record_unpack(throughput.record);
    RGBF color  = get_color(0.0f, 0.0f, 0.0f);

    float cloud_transmittance;
    const float cloud_offset = clouds_render(sky_origin, task.ray, sky_max_dist, task.path_id, color, record, cloud_transmittance);

    if (depth == FLT_MAX) {
      record = splat_color(0.0f);
    }
    else {
      // Move past the clouds
      if (cloud_offset != FLT_MAX && cloud_offset > 0.0f) {
        const float cloud_world_offset = sky_to_world_scale(cloud_offset);

        task.origin = add_vector(task.origin, scale_vector(task.ray, cloud_world_offset));
        task_store(task_base_address, task);

        depth -= cloud_world_offset;
        task_trace_depth_store(task_base_address, depth);
      }
    }

    task_throughput_record_store(task_base_address, record_pack(record));

    write_beauty_buffer(color, throughput.results_index);
  }
}

#endif /* CU_CLOUD_H */
