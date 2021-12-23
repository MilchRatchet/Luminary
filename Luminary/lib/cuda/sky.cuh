#ifndef CU_SKY_H
#define CU_SKY_H

#include <cuda_runtime_api.h>

#include "math.cuh"
#include "stars.h"
#include "utils.cuh"

#define EARTH_RADIUS 6371.0f

__device__ float get_length_to_border(const vec3 origin, vec3 ray, const float atmosphere_end) {
  if (ray.y < 0.0f)
    ray = scale_vector(ray, -1.0f);
  const float a = dot_product(origin, ray);
  return -a + sqrtf(a * a - dot_product(origin, origin) + atmosphere_end * atmosphere_end);
}

__device__ float density_at_height(const float height, const float density_falloff) {
  return expf(-height * density_falloff);
}

__device__ float height_at_point(const vec3 point) {
  return get_length(point) - EARTH_RADIUS;
}

__device__ float get_earth_intersection(const vec3 origin, const vec3 ray) {
  const float a = dot_product(origin, ray);
  const float b = get_length(origin);
  return a * a - b * b + EARTH_RADIUS * EARTH_RADIUS;
}

__device__ float get_optical_depth(const vec3 origin, const vec3 ray, const float start, const float length) {
  if (length <= 0.0f)
    return 0.0f;

  const int steps       = device_scene.sky.steps;
  const float step_size = length / steps;
  float depth           = 0.0f;
  vec3 point            = origin;

  point = add_vector(point, scale_vector(ray, start + 0.125f * step_size));

  for (int i = 0; i < steps; i++) {
    depth += density_at_height(height_at_point(point), 0.125f) * step_size;

    point = add_vector(point, scale_vector(ray, step_size));
  }

  return depth;
}

__device__ RGBF get_transmittance_fast(const float optical_depth, const RGBF rayleigh, const float mie) {
  RGBF transmittance;
  transmittance.r = expf(-optical_depth * (rayleigh.r + 1.11f * mie));
  transmittance.g = expf(-optical_depth * (rayleigh.g + 1.11f * mie));
  transmittance.b = expf(-optical_depth * (rayleigh.b + 1.11f * mie));

  return transmittance;
}

__device__ RGBF get_sky_color(vec3 pos, const vec3 ray) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  const float overall_density = device_scene.sky.base_density;

  const RGBF scatter = get_color(5.8f * 0.001f * overall_density, 13.558f * 0.001f * overall_density, 33.1f * 0.001f * overall_density);

  const float mie_scatter = 3.996f * 0.001f * overall_density;

  const RGBF ozone_absorbtion =
    get_color(0.65f * 0.001f * overall_density, 1.881f * 0.001f * overall_density, 0.085f * 0.001f * overall_density);

  const float sun_dist   = 149597870.0f;
  const float sun_radius = 696340.0f;

  const float sky_intensity = 6.0f;

  const vec3 sun_normalized = device_sun;
  vec3 sun                  = scale_vector(sun_normalized, sun_dist);

  sun.y -= EARTH_RADIUS;

  vec3 moon = angles_to_direction(device_scene.sky.moon_altitude, device_scene.sky.moon_azimuth);
  moon      = scale_vector(moon, 384399.0f);
  moon.y -= EARTH_RADIUS;

  const float moon_radius = 1737.4f;

  const float atmosphere_height = 100.0f;
  const float atmo_radius       = atmosphere_height + EARTH_RADIUS;

  pos = scale_vector(pos, 0.001f);

  vec3 origin = get_vector(pos.x, EARTH_RADIUS + pos.y, pos.z);

  const vec3 origin_default = origin;

  const float height = get_length(origin);

  if (height <= EARTH_RADIUS)
    return result;

  float distance;
  float start = 0.0f;
  if (height > atmo_radius) {
    float earth_dist = sphere_ray_intersection(ray, origin, get_vector(0.0f, 0.0f, 0.0f), EARTH_RADIUS);
    float atmo_dist  = sphere_ray_intersection(ray, origin, get_vector(0.0f, 0.0f, 0.0f), atmo_radius);
    float atmo_dist2 = sphere_ray_intersect_back(ray, origin, get_vector(0.0f, 0.0f, 0.0f), atmo_radius);

    distance = fminf(earth_dist - atmo_dist, atmo_dist2 - atmo_dist);
    start    = atmo_dist;
  }
  else {
    float earth_dist = sphere_ray_intersection(ray, origin, get_vector(0.0f, 0.0f, 0.0f), EARTH_RADIUS);
    float atmo_dist  = sphere_ray_intersection(ray, origin, get_vector(0.0f, 0.0f, 0.0f), atmo_radius);
    distance         = fminf(earth_dist, atmo_dist);
  }

  if (distance > 0.0f) {
    const int steps       = device_scene.sky.steps;
    const float step_size = distance / steps;
    float reach           = 0.0f;

    reach += step_size * 0.125f;

    origin = add_vector(origin, scale_vector(ray, start + 0.125f * step_size));

    for (int i = 0; i < steps; i++) {
      const vec3 ray_scatter = normalize_vector(sub_vector(sun, origin));

      if (sphere_ray_hit(ray_scatter, origin, moon, moon_radius * 0.5f))
        continue;

      const float optical_depth =
        get_optical_depth(origin_default, ray, start, reach)
        + get_optical_depth(
          origin, ray_scatter, 0.0f, sphere_ray_intersection(ray_scatter, origin, get_vector(0.0f, 0.0f, 0.0f), atmo_radius));

      const float height = height_at_point(origin);

      const float local_density = density_at_height(height, device_scene.sky.rayleigh_falloff);
      const float mie_density   = density_at_height(height, device_scene.sky.mie_falloff);
      const float ozone_density = fmaxf(0.0f, 1.0f - fabsf(height - atmosphere_height * 0.25f) * 0.066666667f);

      RGBF transmittance;
      transmittance.r = expf(-optical_depth * (scatter.r + ozone_density * ozone_absorbtion.r + 1.11f * mie_scatter));
      transmittance.g = expf(-optical_depth * (scatter.g + ozone_density * ozone_absorbtion.g + 1.11f * mie_scatter));
      transmittance.b = expf(-optical_depth * (scatter.b + ozone_density * ozone_absorbtion.b + 1.11f * mie_scatter));

      float cos_angle = dot_product(ray, ray_scatter);

      const float rayleigh = 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);

      const float g   = 0.8f;
      const float mie = 1.5f * (1.0f + cos_angle * cos_angle) * (1.0f - g * g)
                        / (4.0f * 3.1415926535f * (2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * cos_angle, 1.5f));

      result.r += transmittance.r * (local_density * scatter.r * rayleigh + mie_density * mie_scatter * mie);
      result.g += transmittance.g * (local_density * scatter.g * rayleigh + mie_density * mie_scatter * mie);
      result.b += transmittance.b * (local_density * scatter.b * rayleigh + mie_density * mie_scatter * mie);

      reach += step_size;

      origin = add_vector(origin, scale_vector(ray, step_size));
    }

    result = scale_color(result, step_size);
  }

  origin = origin_default;

  const float sun_hit   = sphere_ray_intersection(ray, origin, sun, sun_radius);
  const float earth_hit = sphere_ray_intersection(ray, origin, get_vector(0.0f, 0.0f, 0.0f), EARTH_RADIUS);
  const float moon_hit  = sphere_ray_intersection(ray, origin, moon, moon_radius);

  if (earth_hit > sun_hit && moon_hit > sun_hit) {
    const float optical_depth = get_optical_depth(origin_default, ray, start, distance);
    const RGBF transmittance  = get_transmittance_fast(optical_depth, scatter, mie_scatter);

    result.r += transmittance.r * device_scene.sky.sun_strength;
    result.g += transmittance.g * device_scene.sky.sun_strength;
    result.b += transmittance.b * device_scene.sky.sun_strength;
  }
  else if (earth_hit > moon_hit) {
    vec3 moon_pos = add_vector(origin, scale_vector(ray, moon_hit));

    vec3 normal = normalize_vector(sub_vector(moon_pos, moon));

    vec3 bounce_ray = normalize_vector(sub_vector(sun, moon_pos));

    if (!sphere_ray_hit(bounce_ray, moon_pos, get_vector(0.0f, 0.0f, 0.0f), EARTH_RADIUS) && dot_product(normal, bounce_ray) > 0.0f) {
      const float optical_depth = get_optical_depth(origin_default, ray, start, distance);
      const RGBF transmittance  = get_transmittance_fast(optical_depth, scatter, mie_scatter);

      result.r += transmittance.r * device_scene.sky.sun_strength * device_scene.sky.moon_albedo;
      result.g += transmittance.g * device_scene.sky.sun_strength * device_scene.sky.moon_albedo;
      result.b += transmittance.b * device_scene.sky.sun_strength * device_scene.sky.moon_albedo;
    }
  }

  result = mul_color(result, device_scene.sky.sun_color);

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

      float star_hit = sphere_ray_intersection(ray, get_vector(0.0f, 0.0f, 0.0f), star_pos, star.radius);

      if (star_hit < FLT_MAX) {
        const float optical_depth = get_optical_depth(origin_default, ray, start, distance);
        const RGBF transmittance  = get_transmittance_fast(optical_depth, scatter, mie_scatter);

        result.r += transmittance.r * star.intensity * device_scene.sky.stars_intensity;
        result.g += transmittance.g * star.intensity * device_scene.sky.stars_intensity;
        result.b += transmittance.b * star.intensity * device_scene.sky.stars_intensity;
      }
    }
  }

  result = scale_color(result, sky_intensity);

  return result;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 5 + 2];
  const int task_offset = device.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    const RGBF record = device_records[pixel];
    RGBF sky          = get_sky_color(task.origin, task.ray);
    sky               = mul_color(sky, record);

    const uint32_t light = device.light_sample_history[pixel];

    if (device_iteration_type != TYPE_LIGHT || light == 0) {
      device.frame_buffer[pixel] = add_color(device.frame_buffer[pixel], sky);
    }

    write_albedo_buffer(sky, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 5 + 2];
  const int task_offset = device.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO || device_shading_mode == SHADING_LIGHTSOURCE) {
      device.frame_buffer[pixel] = get_sky_color(task.origin, task.ray);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / device_scene.camera.far_clip_distance) * 2.0f);
      device.frame_buffer[pixel] = get_color(value, value, value);
    }
  }
}

#endif /* CU_SKY_H */
