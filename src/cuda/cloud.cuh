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

__device__ CloudExtinctionOctaves cloud_extinction(const vec3 origin, const vec3 ray) {
  CloudExtinctionOctaves extinction;

  for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
    extinction.E[i] = 0.0f;
  }

  const float iter_step = 1.0f / device.scene.sky.cloud.shadow_steps;

  // Sometimes the shadow ray goes below the cloud layer but misses the earth and reenters the cloud layer
  float offset = 0.0f;

  for (float i = 0.0f; i < 1.0f; i += iter_step) {
    float t0 = offset + i;
    float t1 = offset + i + iter_step;
    t0       = t0 * t0;
    t1       = t1 * t1;

    const float step_size = t1 - t0;
    const float reach     = t0 + step_size * 0.5f;

    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height > 1.0f)
      break;

    if (height < 0.0f) {
      const float h_min = world_to_sky_scale(device.scene.sky.cloud.height_min) + SKY_EARTH_RADIUS;
      offset += sph_ray_int_p0(ray, pos, h_min);
      continue;
    }

    const vec3 weather = cloud_weather(pos, height);

    if (weather.x > 0.05f) {
      const float density = CLOUD_EXTINCTION_DENSITY * cloud_density(pos, height, weather);

      float octave_factor = 1.0f;

      for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
        extinction.E[i] -= octave_factor * density * step_size;
        octave_factor *= CLOUD_OCTAVE_EXTINCTION_FACTOR;
      }
    }
  }

  for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
    extinction.E[i] = expf(extinction.E[i]);
  }

  return extinction;
}

__device__ RGBAF cloud_render(const vec3 origin, const vec3 ray, const float start, float dist) {
  dist = fminf(30.0f, dist);

  const int step_count = device.scene.sky.cloud.steps * __saturatef(dist / 15.0f);

  const int big_step_mult = 2;
  const float big_step    = big_step_mult;

  const float step_size = dist / step_count;

  float reach = start + (white_noise() + 0.1f) * step_size;

  float transmittance = 1.0f;
  RGBF scatteredLight = get_color(0.0f, 0.0f, 0.0f);

  const float ambient_r1 = 2.0f * PI * white_noise();
  const float ambient_r2 = 2.0f * PI * white_noise();

  const vec3 ambient_ray        = angles_to_direction(ambient_r1, ambient_r2);
  const float ambient_cos_angle = dot_product(ray, ambient_ray);

  CloudPhaseOctaves ambient_phase;
  float phase_factor = 1.0f;
  for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
    ambient_phase.P[i] = cloud_dual_lobe_henvey_greenstein(ambient_cos_angle, phase_factor);
    phase_factor *= CLOUD_OCTAVE_PHASE_FACTOR;
  }

  const float sun_light_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, add_vector(origin, scale_vector(ray, reach)));

  for (int i = 0; i < step_count; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height < 0.0f || height > 1.0f) {
      break;
    }

    const vec3 weather = cloud_weather(pos, height);

    if (weather.x < 0.05f) {
      i += big_step_mult - 1;
      reach += step_size * big_step;
      continue;
    }

    const float density = cloud_density(pos, height, weather);

    if (density > 0.0f) {
      RGBF sun_color;
      CloudPhaseOctaves sun_phase;
      CloudExtinctionOctaves sun_extinction;

      const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, pos)), pos, SKY_EARTH_RADIUS);
      if (sun_visible) {
        const vec3 sun_ray = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, pos);

        const float sun_cos_angle = dot_product(ray, sun_ray);

        float phase_factor = 1.0f;
        for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
          sun_phase.P[i] = cloud_dual_lobe_henvey_greenstein(sun_cos_angle, phase_factor);
          phase_factor *= CLOUD_OCTAVE_PHASE_FACTOR;
        }

        sun_extinction = cloud_extinction(pos, sun_ray);

        sun_color = sky_get_sun_color(pos, sun_ray);
        sun_color = scale_color(sun_color, sun_light_angle);
      }
      else {
        sun_color = get_color(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
          sun_phase.P[i]      = 0.0f;
          sun_extinction.E[i] = 1.0f;
        }
      }

      // Ambient light
      const CloudExtinctionOctaves ambient_extinction = cloud_extinction(pos, ambient_ray);

      RGBF ambient_color = sky_get_color(pos, ambient_ray, FLT_MAX, false, device.scene.sky.steps / 2);
      ambient_color      = scale_color(ambient_color, 4.0f * PI);

      float scattering_factor = 1.0f;
      float extinction_factor = 1.0f;
      for (int i = 0; i < CLOUD_SCATTERING_OCTAVES; i++) {
        const float scattering = scattering_factor * density * CLOUD_SCATTERING_DENSITY;
        const float extinction = fmaxf(extinction_factor * density * CLOUD_EXTINCTION_DENSITY, 0.0001f);

        scattering_factor *= CLOUD_OCTAVE_SCATTERING_FACTOR;
        extinction_factor *= CLOUD_OCTAVE_EXTINCTION_FACTOR;

        const RGBF sun_color_i     = scale_color(sun_color, sun_extinction.E[i] * sun_phase.P[i]);
        const RGBF ambient_color_i = scale_color(ambient_color, ambient_extinction.E[i] * ambient_phase.P[i]);

        RGBF S = add_color(sun_color_i, ambient_color_i);
        S      = scale_color(S, scattering);
        S      = scale_color(S, cloud_powder(scattering, step_size));
        S      = scale_color(S, cloud_weather_wetness(weather));

        const float step_trans = expf(-extinction * step_size);

        S              = scale_color(sub_color(S, scale_color(S, step_trans)), 1.0f / extinction);
        scatteredLight = add_color(scatteredLight, scale_color(S, transmittance));
      }

      // Update transmittance
      transmittance *= expf(-density * CLOUD_EXTINCTION_DENSITY * step_size);

      if (transmittance < 0.005f) {
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

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ void trace_clouds(const vec3 origin, const vec3 ray, const float start, const float distance, ushort2 index) {
  int pixel = index.x + index.y * device.width;

  if (distance <= 0.0f)
    return;

  RGBAF result = cloud_render(origin, ray, start, distance);

  if ((result.r + result.g + result.b) != 0.0f) {
    RGBF color  = RGBAhalf_to_RGBF(device.ptrs.frame_buffer[pixel]);
    RGBF record = RGBAhalf_to_RGBF(device.records[pixel]);

    color.r += result.r * record.r;
    color.g += result.g * record.g;
    color.b += result.b * record.b;

    device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(color);
  }

  if (result.a != 1.0f) {
    RGBAhalf record = load_RGBAhalf(device.records + pixel);
    record          = scale_RGBAhalf(record, __float2half(result.a));
    store_RGBAhalf(device.records + pixel, record);
  }
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 4) void clouds_render_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset  = get_task_address(i);
    TraceTask task    = load_trace_task(device.trace_tasks + offset);
    const float depth = __ldcs((float*) (device.ptrs.trace_results + offset));

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float sky_max_dist = (depth == device.scene.camera.far_clip_distance) ? FLT_MAX : world_to_sky_scale(depth);
    const float2 params      = cloud_get_intersection(sky_origin, task.ray, sky_max_dist);

    const bool cloud_hit = (params.x < FLT_MAX && params.y > 0.0f);

    if (cloud_hit) {
      trace_clouds(sky_origin, task.ray, params.x, params.y, task.index);

      task.origin = add_vector(task.origin, scale_vector(task.ray, sky_to_world_scale(params.x)));
      store_trace_task(device.trace_tasks + offset, task);
    }
  }
}

#endif /* CU_CLOUD_H */
