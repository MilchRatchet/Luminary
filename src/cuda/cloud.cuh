#ifndef CU_CLOUD_H
#define CU_CLOUD_H

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky.cuh"
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

__device__ float cloud_extinction(const vec3 origin, const vec3 ray, const CloudLayerType layer) {
  const float iter_step = 1.0f / device.scene.sky.cloud.shadow_steps;

  float optical_depth = 0.0f;

  for (float i = 0.0f; i < 1.0f; i += iter_step) {
    float t0 = i;
    float t1 = i + iter_step;
    t0       = t0 * t0;
    t1       = t1 * t1;

    const float step_size = t1 - t0;
    const float reach     = t0 + step_size * 0.5f;

    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos, layer);

    if (height > 1.0f || height < 0.0f)
      break;

    const CloudWeather weather = cloud_weather(pos, height, layer);

    if (cloud_significant_point(height, weather, layer)) {
      optical_depth -= cloud_density(pos, height, weather, 0.0f, layer) * step_size;
    }
  }

  optical_depth *= CLOUD_EXTINCTION_DENSITY;

  return expf(optical_depth);
}

/*
 * Returns an RGBAF where the RGB is the radiance and the A is the greyscale transmittance.
 */
__device__ CloudRenderResult
  clouds_compute(vec3 origin, vec3 ray, float start, float dist, const CloudLayerType layer, const ushort2 pixel) {
  if (dist < 0.0f || start == FLT_MAX) {
    CloudRenderResult result;

    result.scattered_light = get_color(0.0f, 0.0f, 0.0f);
    result.transmittance   = 1.0f;
    result.hit_dist        = start;

    return result;
  }

  int step_count;
  switch (layer) {
    case CLOUD_LAYER_LOW: {
      const float span = device.scene.sky.cloud.low.height_max - device.scene.sky.cloud.low.height_min;
      dist             = fminf(6.0f * span, dist);
      step_count       = device.scene.sky.cloud.steps * __saturatef(dist / (6.0f * span));
    } break;
    case CLOUD_LAYER_MID: {
      const float span = device.scene.sky.cloud.mid.height_max - device.scene.sky.cloud.mid.height_min;
      dist             = fminf(6.0f * span, dist);
      step_count       = (device.scene.sky.cloud.steps / 4) * __saturatef(dist / (6.0f * span));
    } break;
    case CLOUD_LAYER_TOP: {
      const float span = device.scene.sky.cloud.top.height_max - device.scene.sky.cloud.top.height_min;
      dist             = fminf(6.0f * span, dist);
      step_count       = (device.scene.sky.cloud.steps / 8) * __saturatef(dist / (6.0f * span));
    } break;
    default: {
      CloudRenderResult result;

      result.scattered_light = get_color(0.0f, 0.0f, 0.0f);
      result.transmittance   = 1.0f;
      result.hit_dist        = start;

      return result;
    }
  }

  step_count += 8.0f * quasirandom_sequence_1D(QUASI_RANDOM_TARGET_CLOUD_STEP_COUNT + layer, pixel);

  start = fmaxf(0.0f, start);

  const float step_size     = dist / step_count;
  const float random_offset = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_CLOUD_STEP_OFFSET + layer, pixel);
  float reach               = start + (0.1f + random_offset * 0.9f) * step_size;

  const float sun_solid_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, add_vector(origin, scale_vector(ray, reach)));

  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.sky.cloud.droplet_diameter);

  float transmittance  = 1.0f;
  RGBF scattered_light = get_color(0.0f, 0.0f, 0.0f);
  float hit_dist       = start;
  bool hit             = false;

  for (int i = 0; i < step_count; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    if (!hit) {
      hit_dist = reach;
    }

    const float height = cloud_height(pos, layer);

    if (height < 0.0f || height > 1.0f) {
      break;
    }

    const CloudWeather weather = cloud_weather(pos, height, layer);

    if (!cloud_significant_point(height, weather, layer)) {
      reach += step_size;
      continue;
    }

    const float density = cloud_density(pos, height, weather, 0.0f, layer);

    if (density > 0.0f) {
      hit = true;

      float2 ambient_r = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CLOUD_DIR + i, pixel);
      ambient_r.x      = 2.0f * ambient_r.x - 1.0f;

      const vec3 ambient_ray = sample_ray_sphere(ambient_r.x, ambient_r.y);
      RGBF ambient_color     = sky_get_color(pos, ambient_ray, FLT_MAX, false, device.scene.sky.steps / 2, pixel);

      float ambient_extinction      = cloud_extinction(pos, ambient_ray, layer);
      const float ambient_cos_angle = dot_product(ray, ambient_ray);

      RGBF sun_color;
      float sun_extinction;
      float sun_cos_angle;

      const vec3 sun_ray = normalize_vector(sub_vector(device.sun_pos, pos));

      const int sun_visible = !sph_ray_hit_p0(sun_ray, pos, SKY_EARTH_RADIUS);
      if (sun_visible) {
        sun_color = sky_get_sun_color(pos, sun_ray);

        sun_cos_angle = dot_product(ray, sun_ray);

        sun_extinction = cloud_extinction(pos, sun_ray, layer);
      }
      else {
        sun_color      = get_color(0.0f, 0.0f, 0.0f);
        sun_extinction = 1.0f;
        sun_cos_angle  = 0.0f;
      }

      float scattering   = density * CLOUD_SCATTERING_DENSITY;
      float extinction   = fmaxf(density * CLOUD_EXTINCTION_DENSITY, 0.0001f);
      float phase_factor = 1.0f;
      for (int i = 0; i < device.scene.sky.cloud.octaves; i++) {
        scattering *= CLOUD_OCTAVE_SCATTERING_FACTOR;
        extinction *= CLOUD_OCTAVE_EXTINCTION_FACTOR;

        const float sun_phase     = jendersie_eon_phase_function(sun_cos_angle, params, phase_factor);
        const float ambient_phase = jendersie_eon_phase_function(ambient_cos_angle, params, phase_factor);
        phase_factor *= CLOUD_OCTAVE_PHASE_FACTOR;

        const RGBF sun_color_i     = scale_color(sun_color, sun_extinction * sun_phase * sun_solid_angle);
        const RGBF ambient_color_i = scale_color(ambient_color, ambient_extinction * ambient_phase * 4.0f * PI);

        sun_extinction     = sqrtf(sun_extinction);
        ambient_extinction = sqrtf(ambient_extinction);

        RGBF S = add_color(sun_color_i, ambient_color_i);
        S      = scale_color(S, scattering);

        const float step_trans = expf(-extinction * step_size);

        S               = scale_color(sub_color(S, scale_color(S, step_trans)), 1.0f / extinction);
        scattered_light = add_color(scattered_light, scale_color(S, transmittance));
      }

      transmittance *= expf(-density * CLOUD_EXTINCTION_DENSITY * step_size);

      if (transmittance < 0.1f) {
        transmittance = 0.0f;
        break;
      }
    }

    reach += step_size;
  }

  CloudRenderResult result;

  result.scattered_light = scattered_light;
  result.transmittance   = transmittance;
  result.hit_dist        = hit_dist;

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ float clouds_render(
  vec3 origin, const vec3 ray, const float limit, const ushort2 pixel, RGBF& color, RGBF& transmittance, float& transmittance_cloud_only) {
  float2 intersections[3];
  CloudRenderResult results[3];

  intersections[0] = cloud_get_lowlayer_intersection(origin, ray, limit);
  results[0]       = clouds_compute(origin, ray, intersections[0].x, intersections[0].y, CLOUD_LAYER_LOW, pixel);

  intersections[1] = cloud_get_midlayer_intersection(origin, ray, limit);
  results[1]       = clouds_compute(origin, ray, intersections[1].x, intersections[1].y, CLOUD_LAYER_MID, pixel);

  intersections[2] = cloud_get_toplayer_intersection(origin, ray, limit);
  results[2]       = clouds_compute(origin, ray, intersections[2].x, intersections[2].y, CLOUD_LAYER_TOP, pixel);

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

  float prev_start = 0.0f;

  for (int i = 0; i < 3; i++) {
    const CloudRenderResult result = results[order[i]];

    if (result.hit_dist == FLT_MAX)
      break;

#ifndef SHADING_KERNEL
    if (device.scene.sky.cloud.atmosphere_scattering) {
      color  = add_color(color, sky_trace_inscattering(origin, ray, result.hit_dist - prev_start, transmittance, pixel));
      origin = add_vector(origin, scale_vector(ray, result.hit_dist - prev_start));
    }
#endif /* SHADING_KERNEL */

    color         = add_color(color, mul_color(result.scattered_light, transmittance));
    transmittance = scale_color(transmittance, result.transmittance);
    transmittance_cloud_only *= result.transmittance;

    prev_start = result.hit_dist;
  }

  return prev_start;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void clouds_render_tasks() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset         = get_task_address(i);
    TraceTask task           = load_trace_task(device.ptrs.trace_tasks + offset);
    const float depth        = __ldcs((float*) (device.ptrs.trace_results + offset));
    vec3 sky_origin          = world_to_sky_transform(task.origin);
    const float sky_max_dist = (depth == device.scene.camera.far_clip_distance) ? FLT_MAX : world_to_sky_scale(depth);
    const int pixel          = task.index.y * device.width + task.index.x;

    RGBF record = load_RGBF(device.ptrs.records + pixel);
    RGBF color  = get_color(0.0f, 0.0f, 0.0f);

    float cloud_transmittance;
    const float cloud_offset = clouds_render(sky_origin, task.ray, sky_max_dist, task.index, color, record, cloud_transmittance);

    if (device.scene.sky.cloud.atmosphere_scattering) {
      if (cloud_offset != FLT_MAX && cloud_offset > 0.0f) {
        const float cloud_world_offset = sky_to_world_scale(cloud_offset);

        task.origin = add_vector(task.origin, scale_vector(task.ray, cloud_world_offset));
        store_trace_task(device.ptrs.trace_tasks + offset, task);

        if (depth != device.scene.camera.far_clip_distance) {
          __stcs((float*) (device.ptrs.trace_results + offset), depth - cloud_world_offset);
        }
      }
    }

    store_RGBF(device.ptrs.records + pixel, record);
    write_beauty_buffer(color, pixel);
  }
}

#endif /* CU_CLOUD_H */
