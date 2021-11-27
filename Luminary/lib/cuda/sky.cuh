#ifndef CU_SKY_H
#define CU_SKY_H

#include <cuda_runtime_api.h>

#include "math.cuh"
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
  return (get_length(point) - EARTH_RADIUS);
}

__device__ float get_earth_intersection(const vec3 origin, const vec3 ray) {
  const float a = dot_product(origin, ray);
  const float b = get_length(origin);
  return a * a - b * b + EARTH_RADIUS * EARTH_RADIUS;
}

__device__ float get_earth_shadowing(const vec3 origin, const vec3 ray, const vec3 sun, const float atmosphere_length) {
  if (sun.y >= 0.0f)
    return 1.0f;

  const int steps = 8;
  float a         = atmosphere_length * 0.5f;
  float step_size = a;

  vec3 p = add_vector(origin, scale_vector(ray, a));

  for (int i = 0; i < steps; i++) {
    const vec3 ray_scatter = normalize_vector(sub_vector(sun, p));

    float delta = get_earth_intersection(p, ray_scatter);

    if (fabsf(delta) < eps)
      break;

    a += (delta > 0.0f) ? step_size : -step_size;
    step_size *= 0.5f;

    if (a < 0.0f) {
      return 1.0f;
    }

    p = add_vector(origin, scale_vector(ray, a));
  }

  return fmaxf(0.0f, 1.0f - (a / atmosphere_length));
}

__device__ float get_optical_depth(const vec3 origin, const vec3 ray, const float length) {
  if (length == 0.0f)
    return 0.0f;

  const int steps       = 8;
  const float step_size = length / steps;
  float depth           = 0.0f;
  vec3 point            = origin;

  point = add_vector(point, scale_vector(ray, 0.125f * step_size));

  for (int i = 0; i < steps; i++) {
    depth += density_at_height(height_at_point(point), 0.125f) * step_size;

    point = add_vector(point, scale_vector(ray, step_size));
  }

  return depth;
}

__device__ RGBF get_sky_color(const vec3 ray) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);
  ;

  if (ray.y < 0.0f) {
    return result;
  }

  const float angular_diameter = 0.009f;

  const float overall_density = device_scene.sky.base_density;

  RGBF scatter;
  scatter.r = 5.8f * 0.001f * overall_density;
  scatter.g = 13.558f * 0.001f * overall_density;
  scatter.b = 33.1f * 0.001f * overall_density;

  const float mie_scatter = 3.996f * 0.001f * overall_density;

  RGBF ozone_absorbtion;
  ozone_absorbtion.r = 0.65f * 0.001f * overall_density;
  ozone_absorbtion.g = 1.881f * 0.001f * overall_density;
  ozone_absorbtion.b = 0.085f * 0.001f * overall_density;

  const float sun_dist      = 150000000.0f;
  const float sun_intensity = 6.0f;

  const vec3 sun_normalized = device_sun;
  const vec3 sun            = scale_vector(sun_normalized, sun_dist);

  const float atmosphere_height = 100.0f;

  vec3 origin = get_vector(0.0f, EARTH_RADIUS, 0.0f);

  const vec3 origin_default = origin;

  const float limit           = get_length_to_border(origin, ray, EARTH_RADIUS + atmosphere_height);
  const float earth_shadowing = get_earth_shadowing(origin, ray, sun, limit);
  const int steps             = 8;
  const float step_size       = limit / steps;
  float reach                 = 0.0f;

  reach += step_size * 0.125f;

  origin = add_vector(origin, scale_vector(ray, 0.125f * step_size));

  for (int i = 0; i < steps; i++) {
    const vec3 ray_scatter = normalize_vector(sub_vector(sun, origin));

    const float optical_depth =
      get_optical_depth(origin_default, ray, reach)
      + get_optical_depth(origin, ray_scatter, get_length_to_border(origin, ray_scatter, EARTH_RADIUS + atmosphere_height));

    const float height = height_at_point(origin);

    const float local_density = density_at_height(height, device_scene.sky.rayleigh_falloff);
    const float mie_density   = density_at_height(height, device_scene.sky.mie_falloff);
    // The tent function is disabled atm, first argument 0.0f to activate
    const float ozone_density = fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);

    RGBF transmittance;
    transmittance.r = expf(-optical_depth * (scatter.r + ozone_density * ozone_absorbtion.r + 1.11f * mie_scatter));
    transmittance.g = expf(-optical_depth * (scatter.g + ozone_density * ozone_absorbtion.g + 1.11f * mie_scatter));
    transmittance.b = expf(-optical_depth * (scatter.b + ozone_density * ozone_absorbtion.b + 1.11f * mie_scatter));

    float cos_angle = dot_product(ray, ray_scatter);

    cos_angle = cosf(fmaxf(0.0f, acosf(cos_angle) - angular_diameter));

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

  result.r *= step_size * earth_shadowing;
  result.g *= step_size * earth_shadowing;
  result.b *= step_size * earth_shadowing;

  const vec3 ray_sun = normalize_vector(sub_vector(sun, origin_default));

  float cos_angle = dot_product(ray, ray_sun);
  cos_angle       = cosf(fmaxf(0.0f, acosf(cos_angle) - angular_diameter));

  if (cos_angle >= 0.99999f) {
    const float optical_depth = get_optical_depth(origin_default, ray, limit);

    const float height = height_at_point(origin_default);

    const float ozone_density = fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);

    RGBF transmittance;
    transmittance.r = expf(-optical_depth * (scatter.r + ozone_density * ozone_absorbtion.r + 1.11f * mie_scatter));
    transmittance.g = expf(-optical_depth * (scatter.g + ozone_density * ozone_absorbtion.g + 1.11f * mie_scatter));
    transmittance.b = expf(-optical_depth * (scatter.b + ozone_density * ozone_absorbtion.b + 1.11f * mie_scatter));

    result.r += transmittance.r * cos_angle * device_scene.sky.sun_strength;
    result.g += transmittance.g * cos_angle * device_scene.sky.sun_strength;
    result.b += transmittance.b * cos_angle * device_scene.sky.sun_strength;
  }

  result.r *= sun_intensity * device_scene.sky.sun_color.r;
  result.g *= sun_intensity * device_scene.sky.sun_color.g;
  result.b *= sun_intensity * device_scene.sky.sun_color.b;

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
    RGBF sky          = get_sky_color(task.ray);
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

    if (device_shading_mode == SHADING_ALBEDO) {
      device.frame_buffer[pixel] = get_sky_color(task.ray);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / device_scene.camera.far_clip_distance) * 2.0f);
      device.frame_buffer[pixel] = get_color(value, value, value);
    }
  }
}

#endif /* CU_SKY_H */
