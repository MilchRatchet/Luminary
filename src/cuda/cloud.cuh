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
__device__ CloudRenderResult cloud_render(const vec3 origin, const vec3 ray, float start, float dist, const CloudLayerType layer) {
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
      step_count       = device.scene.sky.cloud.steps * __saturatef(dist / (6.0f * span)) + white_noise() - 0.5f;
    } break;
    case CLOUD_LAYER_MID: {
      const float span = device.scene.sky.cloud.mid.height_max - device.scene.sky.cloud.mid.height_min;
      dist             = fminf(6.0f * span, dist);
      step_count       = (device.scene.sky.cloud.steps / 4) * __saturatef(dist / (6.0f * span)) + white_noise() - 0.5f;
    } break;
    case CLOUD_LAYER_TOP: {
      const float span = device.scene.sky.cloud.top.height_max - device.scene.sky.cloud.top.height_min;
      dist             = fminf(6.0f * span, dist);
      step_count       = (device.scene.sky.cloud.steps / 8) * __saturatef(dist / (6.0f * span)) + white_noise() - 0.5f;
    } break;
    default: {
      CloudRenderResult result;

      result.scattered_light = get_color(0.0f, 0.0f, 0.0f);
      result.transmittance   = 1.0f;
      result.hit_dist        = start;

      return result;
    }
  }

  start = fmaxf(0.0f, start);

  const float step_size = dist / step_count;
  float reach           = start + (0.1f + white_noise() * 0.9f) * step_size;

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

      const float ambient_r1 = 2.0f * PI * white_noise() - PI;
      const float ambient_r2 = 2.0f * PI * white_noise();

      const vec3 ambient_ray = angles_to_direction(ambient_r1, ambient_r2);
      RGBF ambient_color     = sky_get_color(pos, ambient_ray, FLT_MAX, false, device.scene.sky.steps / 2);
      ambient_color          = scale_color(ambient_color, 4.0f * PI);

      float ambient_extinction      = cloud_extinction(pos, ambient_ray, layer);
      const float ambient_cos_angle = dot_product(ray, ambient_ray);

      RGBF sun_color;
      float sun_extinction;
      float sun_cos_angle;

      const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, pos)), pos, SKY_EARTH_RADIUS);
      if (sun_visible) {
        const vec3 sun_ray = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, pos);

        const float sun_light_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, pos);

        sun_color = sky_get_sun_color(pos, sun_ray);
        sun_color = scale_color(sun_color, sun_light_angle);

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

        const float sun_phase     = cloud_dual_lobe_henvey_greenstein(sun_cos_angle, phase_factor);
        const float ambient_phase = cloud_dual_lobe_henvey_greenstein(ambient_cos_angle, phase_factor);

        phase_factor *= CLOUD_OCTAVE_PHASE_FACTOR;

        const RGBF sun_color_i     = scale_color(sun_color, sun_extinction * sun_phase);
        const RGBF ambient_color_i = scale_color(ambient_color, ambient_extinction * ambient_phase);

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

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 5) void clouds_render_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset         = get_task_address(i);
    TraceTask task           = load_trace_task(device.trace_tasks + offset);
    const float depth        = __ldcs((float*) (device.ptrs.trace_results + offset));
    vec3 sky_origin          = world_to_sky_transform(task.origin);
    const float sky_max_dist = (depth == device.scene.camera.far_clip_distance) ? FLT_MAX : world_to_sky_scale(depth);
    const int pixel          = task.index.y * device.width + task.index.x;

    float2 intersections[3];
    CloudRenderResult results[3];

    intersections[0] = cloud_get_lowlayer_intersection(sky_origin, task.ray, sky_max_dist);
    results[0]       = cloud_render(sky_origin, task.ray, intersections[0].x, intersections[0].y, CLOUD_LAYER_LOW);

    intersections[1] = cloud_get_midlayer_intersection(sky_origin, task.ray, sky_max_dist);
    results[1]       = cloud_render(sky_origin, task.ray, intersections[1].x, intersections[1].y, CLOUD_LAYER_MID);

    intersections[2] = cloud_get_toplayer_intersection(sky_origin, task.ray, sky_max_dist);
    results[2]       = cloud_render(sky_origin, task.ray, intersections[2].x, intersections[2].y, CLOUD_LAYER_TOP);

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

    RGBF record = RGBAhalf_to_RGBF(load_RGBAhalf(device.records + pixel));
    RGBF color  = RGBAhalf_to_RGBF(load_RGBAhalf(device.ptrs.frame_buffer + pixel));

    for (int i = 0; i < 3; i++) {
      const CloudRenderResult result = results[order[i]];

      if (result.hit_dist == FLT_MAX)
        break;

      if (device.iteration_type != TYPE_LIGHT && device.scene.sky.cloud.atmosphere_scattering) {
        color      = add_color(color, sky_trace_inscattering(sky_origin, task.ray, result.hit_dist - prev_start, record));
        sky_origin = add_vector(sky_origin, scale_vector(task.ray, result.hit_dist - prev_start));
      }

      color  = add_color(color, mul_color(result.scattered_light, record));
      record = scale_color(record, result.transmittance);

      prev_start = result.hit_dist;
    }

    if (device.iteration_type != TYPE_LIGHT && device.scene.sky.cloud.atmosphere_scattering) {
      const float cloud_offset = prev_start;

      if (cloud_offset != FLT_MAX && cloud_offset > 0.0f) {
        const float cloud_world_offset = sky_to_world_scale(cloud_offset);

        task.origin = add_vector(task.origin, scale_vector(task.ray, cloud_world_offset));
        store_trace_task(device.trace_tasks + offset, task);

        if (depth != device.scene.camera.far_clip_distance) {
          __stcs((float*) (device.ptrs.trace_results + offset), depth - cloud_world_offset);
        }
      }
    }

    store_RGBAhalf(device.records + pixel, RGBF_to_RGBAhalf(record));
    store_RGBAhalf(device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(color));
  }
}

#endif /* CU_CLOUD_H */
