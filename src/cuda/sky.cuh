#ifndef CU_SKY_H
#define CU_SKY_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "stars.h"
#include "utils.cuh"

#define SKY_RAYLEIGH_SCATTERING get_color(5.8f * 0.001f, 13.558f * 0.001f, 33.1f * 0.001f)
#define SKY_MIE_SCATTERING get_color(3.996f * 0.001f, 3.996f * 0.001f, 3.996f * 0.001f)
#define SKY_OZONE_SCATTERING 0.0f

#define SKY_RAYLEIGH_EXTINCTION SKY_RAYLEIGH_SCATTERING
#define SKY_MIE_EXTINCTION scale_color(SKY_MIE_SCATTERING, 1.11f)
#define SKY_OZONE_EXTINCTION get_color(0.65f * 0.001f, 1.881f * 0.001f, 0.085f * 0.001f)

#define SKY_RAYLEIGH_DISTRIBUTION 0.125f
#define SKY_MIE_DISTRIBUTION 0.833f

__device__ float sky_rayleigh_phase(const float cos_angle) {
  return 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);
}

__device__ float sky_density_falloff(const float height, const float density_falloff) {
  return expf(-height * density_falloff);
}

__device__ float sky_ozone_density(const float height) {
  if (!device_scene.sky.ozone_absorption)
    return 0.0f;

  if (height > 25.0f) {
    return fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.04f);
  }
  else {
    return fmaxf(0.1f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);
  }
}

__device__ float sky_height(const vec3 point) {
  return get_length(point) - SKY_EARTH_RADIUS;
}

__device__ RGBF sky_extinction(const vec3 origin, const vec3 ray, const float start, const float length) {
  if (length <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  const int steps       = device_scene.sky.shadow_steps;
  const float step_size = length / steps;
  RGBF density          = get_color(0.0f, 0.0f, 0.0f);
  float reach           = start + white_noise() * step_size;

  for (int i = 0; i < steps; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height           = sky_height(pos);
    const float density_rayleigh = sky_density_falloff(height, SKY_RAYLEIGH_DISTRIBUTION);
    const float density_mie      = sky_density_falloff(height, SKY_MIE_DISTRIBUTION);
    const float density_ozone    = sky_ozone_density(height);

    RGBF D = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    D      = add_color(D, scale_color(SKY_MIE_EXTINCTION, density_mie));
    D      = add_color(D, scale_color(SKY_OZONE_EXTINCTION, density_ozone));

    density = add_color(density, D);

    reach += step_size;
  }

  density = scale_color(density, -device_scene.sky.base_density * 0.5f * step_size);

  return get_color(expf(density.r), expf(density.g), expf(density.b));
}

/*
 * Computes the start and length of a ray path through atmosphere.
 * @param origin Start point of ray in sky space.
 * @param ray Direction of ray
 * @result 2 floats, first value is the start, second value is the length of the path.
 */
__device__ float2 sky_compute_path(const vec3 origin, const vec3 ray, const float min_height, const float max_height) {
  const float height = get_length(origin);

  if (height <= min_height)
    return make_float2(0.0f, -FLT_MAX);

  float distance;
  float start = 0.0f;
  if (height > max_height) {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    const float atmo_dist2 = sph_ray_int_back_p0(ray, origin, max_height);

    distance = fminf(earth_dist - atmo_dist, atmo_dist2 - atmo_dist);
    start    = atmo_dist;
  }
  else {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    distance               = fminf(earth_dist, atmo_dist);
  }

  return make_float2(start, distance);
}

__device__ RGBF
  sky_compute_atmosphere(RGBF& transmittance_out, const vec3 origin, const vec3 ray, const float limit, const bool celestials) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX)
    return result;

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  RGBF transmittance = get_color(1.0f, 1.0f, 1.0f);

  if (distance > 0.0f) {
    const int steps       = device_scene.sky.steps;
    const float step_size = distance / steps;
    float reach           = start + step_size * white_noise();

    for (int i = 0; i < steps; i++) {
      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float height = sky_height(pos);
      if (height < 0.0f || height > SKY_ATMO_HEIGHT) {
        reach += step_size;
        continue;
      }

      const float light_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, pos);
      const vec3 ray_scatter  = normalize_vector(sub_vector(device_sun, pos));

      const float scatter_distance =
        (sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS)) ? 0.0f : sph_ray_int_p0(ray_scatter, pos, SKY_ATMO_RADIUS);

      // If scatter_distance is 0.0 then all light is extinct
      // This is not very beautiful but it seems to be the easiest approach
      const RGBF extinction_sun = sky_extinction(pos, ray_scatter, 0.0f, scatter_distance);

      const float density_rayleigh = sky_density_falloff(height, SKY_RAYLEIGH_DISTRIBUTION);
      const float density_mie      = sky_density_falloff(height, SKY_MIE_DISTRIBUTION);
      const float density_ozone    = sky_ozone_density(height);

      const float cos_angle = fmaxf(0.0f, dot_product(ray, ray_scatter));

      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = henvey_greenstein(cos_angle, 0.8f);

      // Amount of light that reached pos
      RGBF S = scale_color(device_scene.sky.sun_color, device_scene.sky.sun_strength);
      S      = mul_color(S, extinction_sun);

      // Amount of light that gets scattered towards camera at pos
      RGBF scattering = scale_color(SKY_RAYLEIGH_SCATTERING, density_rayleigh * phase_rayleigh);
      scattering      = add_color(scattering, scale_color(SKY_MIE_SCATTERING, density_mie * phase_mie));
      scattering      = scale_color(scattering, device_scene.sky.base_density * 0.5f * light_angle);

      S = mul_color(S, scattering);

      RGBF extinction = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      extinction      = add_color(extinction, scale_color(SKY_MIE_EXTINCTION, density_mie));
      extinction      = add_color(extinction, scale_color(SKY_OZONE_EXTINCTION, density_ozone));
      extinction      = scale_color(extinction, device_scene.sky.base_density * 0.5f);

      // Amount of light that gets lost along this step
      RGBF step_transmittance;
      step_transmittance.r = expf(-step_size * extinction.r);
      step_transmittance.g = expf(-step_size * extinction.g);
      step_transmittance.b = expf(-step_size * extinction.b);

      // Amount of light that gets scattered towards camera along this step
      S = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));

      result = add_color(result, mul_color(S, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);

      reach += step_size;
    }
  }

  if (celestials) {
    const float sun_hit   = sphere_ray_intersection(ray, origin, device_sun, SKY_SUN_RADIUS);
    const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);
    const float moon_hit  = sphere_ray_intersection(ray, origin, device_moon, SKY_MOON_RADIUS);

    if (earth_hit > sun_hit && moon_hit > sun_hit) {
      const vec3 sun_hit_pos  = add_vector(origin, scale_vector(ray, sun_hit));
      const float limb_factor = 1.0f + dot_product(normalize_vector(sub_vector(sun_hit_pos, device_sun)), ray);
      const float mu          = sqrtf(1.0f - limb_factor * limb_factor);

      const RGBF limb_color = get_color(0.397f, 0.503f, 0.652f);

      const RGBF limb_darkening = get_color(powf(mu, limb_color.r), powf(mu, limb_color.g), powf(mu, limb_color.b));

      RGBF S = mul_color(transmittance, scale_color(device_scene.sky.sun_color, device_scene.sky.sun_strength));
      S      = mul_color(S, limb_darkening);

      result = add_color(result, S);
    }
    else if (earth_hit > moon_hit) {
      vec3 moon_pos   = add_vector(origin, scale_vector(ray, moon_hit));
      vec3 normal     = normalize_vector(sub_vector(moon_pos, device_moon));
      vec3 bounce_ray = normalize_vector(sub_vector(device_sun, moon_pos));

      if (!sphere_ray_hit(bounce_ray, moon_pos, get_vector(0.0f, 0.0f, 0.0f), SKY_EARTH_RADIUS) && dot_product(normal, bounce_ray) > 0.0f) {
        const float light_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, moon_pos);
        const float weight      = device_scene.sky.sun_strength * device_scene.sky.moon_albedo * light_angle;

        result = add_color(result, mul_color(transmittance, scale_color(device_scene.sky.sun_color, weight)));
      }
    }

    if (sun_hit == FLT_MAX && earth_hit == FLT_MAX && moon_hit == FLT_MAX) {
      float ray_altitude = asinf(ray.y);
      float ray_azimuth  = atan2f(-ray.z, -ray.x) + PI;

      int x = (int) (ray_azimuth * 10.0f);
      int y = (int) ((ray_altitude + PI * 0.5f) * 10.0f);

      int grid = x + y * STARS_GRID_LD;

      int a = device_scene.sky.stars_offsets[grid];
      int b = device_scene.sky.stars_offsets[grid + 1];

      for (int i = a; i < b; i++) {
        Star star     = device_scene.sky.stars[i];
        vec3 star_pos = angles_to_direction(star.altitude, star.azimuth);

        if (sphere_ray_hit(ray, get_vector(0.0f, 0.0f, 0.0f), star_pos, star.radius)) {
          result = add_color(result, scale_color(transmittance, star.intensity * device_scene.sky.stars_intensity));
        }
      }
    }
  }

  transmittance_out = mul_color(transmittance_out, transmittance);

  return result;
}

__device__ RGBF sky_get_color(const vec3 origin, const vec3 ray, const float limit, const bool celestials) {
  RGBF unused = get_color(0.0f, 0.0f, 0.0f);

  RGBF result = sky_compute_atmosphere(unused, origin, ray, limit, celestials);

  return result;
}

__device__ void sky_trace_inscattering(const vec3 origin, const vec3 ray, const float limit, ushort2 index) {
  int pixel = index.x + index.y * device_width;

  RGBAhalf record = load_RGBAhalf(device_records + pixel);

  RGBF new_record = RGBAhalf_to_RGBF(record);

  RGBAhalf inscattering = RGBF_to_RGBAhalf(sky_compute_atmosphere(new_record, origin, ray, limit, false));

  if (any_RGBAhalf(inscattering)) {
    store_RGBAhalf(
      device.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.frame_buffer + pixel), mul_RGBAhalf(inscattering, record)));
  }

  store_RGBAhalf(device_records + pixel, RGBF_to_RGBAhalf(new_record));
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 8) void process_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 6 + 2];
  const int task_offset = device.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    const RGBAhalf record = load_RGBAhalf(device_records + pixel);
    const vec3 origin     = world_to_sky_transform(task.origin);
    const uint32_t light  = device.light_sample_history[pixel];

    const RGBAhalf sky =
      mul_RGBAhalf(RGBF_to_RGBAhalf(sky_get_color(origin, task.ray, FLT_MAX, proper_light_sample(light, LIGHT_ID_SUN))), record);

    store_RGBAhalf(device.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.frame_buffer + pixel), sky));
    write_albedo_buffer(RGBAhalf_to_RGBF(sky), pixel);
    write_normal_buffer(get_vector(0.0f, 0.0f, 0.0f), pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_debug_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 6 + 2];
  const int task_offset = device.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      store_RGBAhalf(
        device.frame_buffer + pixel, RGBF_to_RGBAhalf(sky_get_color(world_to_sky_transform(task.origin), task.ray, FLT_MAX, true)));
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value = __saturatef((1.0f / device_scene.camera.far_clip_distance) * 2.0f);
      store_RGBAhalf(device.frame_buffer + pixel, get_RGBAhalf(value, value, value, value));
    }
  }
}

#endif /* CU_SKY_H */
