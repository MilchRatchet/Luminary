#ifndef CU_CLOUD_H
#define CU_CLOUD_H

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

//
// The code of this file is based on the cloud rendering in https://github.com/turanszkij/WickedEngine.
// It follows the basic ideas of using raymarching with noise based density.
// The clouds are divided into a tropospheric layer, containing cumulus and alto clouds, and a cirrus cloud layer.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Hil16]
// Sébastien Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", in Physically Based Shading in Theory and
// Practice course, SIGGRAPH 2016. https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering

// [Sch22]
// Andrew Schneider, "Nubis, Evolved: Real-Time Volumetric Clouds for Skies, Environments, and VFX", in Advances in Real-Time Rendering in
// Games cource, SIGGRAPH 2022.
//

////////////////////////////////////////////////////////////////////
// Integrator functions
////////////////////////////////////////////////////////////////////

__device__ float cloud_extinction(const vec3 origin, const vec3 ray) {
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

    const float height = cloud_height(pos);

    if (height > 1.0f || height < 0.0f)
      break;

    const CloudWeather weather = cloud_weather(pos, height);

    if (cloud_significant_point(height, weather)) {
      optical_depth -= cloud_density(pos, height, weather, 0.0f) * step_size;
    }
  }

  optical_depth *= CLOUD_EXTINCTION_DENSITY;

  return expf(optical_depth);
}

/*
 * Returns an RGBAF where the RGB is the radiance and the A is the greyscale transmittance.
 */
__device__ RGBAF cloud_render_tropospheric(const vec3 origin, const vec3 ray, float start, float dist) {
  if (dist < 0.0f || start == FLT_MAX) {
    return RGBAF_set(0.0f, 0.0f, 0.0f, 1.0f);
  }

  start = fmaxf(0.0f, start);
  dist  = fminf(30.0f, dist);

  const int step_count = device.scene.sky.cloud.steps * __saturatef(dist / 15.0f);

  const int big_step_mult = 2;
  const float big_step    = big_step_mult;

  const float step_size = dist / step_count;

  float reach = start + (white_noise() + 0.1f) * step_size;

  float transmittance = 1.0f;
  RGBF scatteredLight = get_color(0.0f, 0.0f, 0.0f);

  for (int i = 0; i < step_count; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height < 0.0f || height > 1.0f) {
      break;
    }

    const CloudWeather weather = cloud_weather(pos, height);

    if (!cloud_significant_point(height, weather)) {
      i += big_step_mult - 1;
      reach += step_size * big_step;
      continue;
    }

    const float density = cloud_density(pos, height, weather, 0.0f);

    if (density > 0.0f) {
      const float ambient_r1 = 2.0f * PI * white_noise();
      const float ambient_r2 = 2.0f * PI * white_noise();

      const vec3 ambient_ray = angles_to_direction(ambient_r1, ambient_r2);
      RGBF ambient_color     = sky_get_color(pos, ambient_ray, FLT_MAX, false, device.scene.sky.steps / 2);
      ambient_color          = scale_color(ambient_color, 4.0f * PI);

      float ambient_extinction      = cloud_extinction(pos, ambient_ray);
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

        sun_extinction = cloud_extinction(pos, sun_ray);
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
        S      = scale_color(S, cloud_powder(scattering, step_size));

        const float step_trans = expf(-extinction * step_size);

        S              = scale_color(sub_color(S, scale_color(S, step_trans)), 1.0f / extinction);
        scatteredLight = add_color(scatteredLight, scale_color(S, transmittance));
      }

      // Update transmittance
      transmittance *= expf(-density * CLOUD_EXTINCTION_DENSITY * step_size);

      if (transmittance < 0.01f) {
        transmittance = 0.0f;
        break;
      }
    }

    reach += step_size;
  }

  RGBAF result;
  result.r = scatteredLight.r;
  result.g = scatteredLight.g;
  result.b = scatteredLight.b;
  result.a = transmittance;

  return result;
}

__device__ RGBAF cloud_render_cirrus(const vec3 origin, const vec3 ray, const float start, const float dist) {
  RGBAF result;
  result.r = 0.0f;
  result.g = 0.0f;
  result.b = 0.0f;
  result.a = 1.0f;

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

    const float2 tropo_int   = cloud_get_tropolayer_intersection(sky_origin, task.ray, sky_max_dist);
    const RGBAF tropo_result = cloud_render_tropospheric(sky_origin, task.ray, tropo_int.x, tropo_int.y);

    const float2 cirrus_int   = cloud_get_cirruslayer_intersection(sky_origin, task.ray, sky_max_dist);
    const RGBAF cirrus_result = cloud_render_cirrus(sky_origin, task.ray, cirrus_int.x, cirrus_int.y);

    const bool tropo_first = (tropo_int.x <= cirrus_int.x);

    const float start1  = (tropo_first) ? tropo_int.x : cirrus_int.x;
    const float start2  = (tropo_first) ? cirrus_int.x : tropo_int.x;
    const RGBAF result1 = (tropo_first) ? tropo_result : cirrus_result;
    const RGBAF result2 = (tropo_first) ? cirrus_result : tropo_result;

    if (start1 < FLT_MAX) {
      RGBF record = RGBAhalf_to_RGBF(load_RGBAhalf(device.records + pixel));
      RGBF color  = RGBAhalf_to_RGBF(load_RGBAhalf(device.ptrs.frame_buffer + pixel));

      if (device.iteration_type != TYPE_LIGHT) {
        color      = add_color(color, sky_trace_inscattering(sky_origin, task.ray, start1, record));
        sky_origin = add_vector(sky_origin, scale_vector(task.ray, start1));
      }

      color  = add_color(color, mul_color(opaque_color(result1), record));
      record = scale_color(record, result1.a);

      if (start2 < FLT_MAX) {
        if (device.iteration_type != TYPE_LIGHT) {
          color = add_color(color, sky_trace_inscattering(sky_origin, task.ray, start2 - start1, record));
        }

        color  = add_color(color, mul_color(opaque_color(result2), record));
        record = scale_color(record, result2.a);
      }

      if (device.iteration_type != TYPE_LIGHT) {
        const float cloud_offset = (start2 == FLT_MAX) ? start1 : start2;

        if (cloud_offset != FLT_MAX) {
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
}

#endif /* CU_CLOUD_H */
