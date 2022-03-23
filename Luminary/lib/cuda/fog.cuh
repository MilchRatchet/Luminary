#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"

__device__ float get_intersection_fog(vec3 origin, vec3 ray, float random) {
  float height = device_scene.fog.height + device_scene.fog.falloff;
  float dist   = (fabsf(ray.y) < eps) ? FLT_MAX : (height - origin.y) / ray.y;

  float max_dist = FLT_MAX;
  float min_dist = 0.0f;

  if (dist < 0.0f) {
    if (origin.y > height) {
      max_dist = -FLT_MAX;
    }
    else {
      max_dist = FLT_MAX;
    }
  }
  else {
    if (origin.y > height) {
      min_dist = dist;
    }
    else {
      max_dist = dist;
    }
  }

  max_dist = fminf(device_scene.fog.dist, max_dist);

  float t = fmaxf(0.0f, min_dist) + logf(random) / (-device_scene.fog.scattering * 0.001f);

  return (t < min_dist || t > max_dist) ? FLT_MAX : t;
}

__device__ float get_fog_depth(float y, float ry, float depth) {
  float height = device_scene.fog.height + device_scene.fog.falloff;

  if (y >= height && ry >= 0.0f)
    return 0.0f;

  if (y < height && ry <= 0.0f)
    return fminf(device_scene.fog.dist, depth);

  if (y < height) {
    return fminf(device_scene.fog.dist, fminf(((height - y) / ry), depth));
  }

  return fmaxf(0.0f, fminf(device_scene.fog.dist, depth - ((height - y) / ry)));
}

__device__ float get_fog_density(float base_density, float height) {
  if (height > device_scene.fog.height) {
    base_density = (height < device_scene.fog.height + device_scene.fog.falloff)
                     ? lerp(base_density, 0.0f, (height - device_scene.fog.height) / device_scene.fog.falloff)
                     : 0.0f;
  }

  return base_density;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.task_counts[id * 5 + 4];
  const int task_offset  = device.task_offsets[id * 5 + 4];
  int light_trace_count  = device.light_trace_count[id];
  int bounce_trace_count = device.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    if (task.state & DEPTH_LEFT) {
      TraceTask continue_task;
      continue_task.origin = task.position;
      continue_task.ray    = ray;
      continue_task.index  = task.index;
      continue_task.state  = task.state;

      store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), continue_task);

      LightSample light = load_light_sample(device.light_samples, pixel);
      light             = brdf_finalize_light_sample(light, task.position);

      if (light.weight <= 0.0f) {
        continue;
      }

      float alpha = blue_noise(task.index.x, task.index.y, task.state, 99);
      float gamma = 2.0f * PI * blue_noise(task.index.x, task.index.y, task.state, 98);

      vec3 out_ray = brdf_sample_light_ray(light, task.position);
      float angle  = dot_product(ray, out_ray);
      float g      = device_scene.fog.anisotropy;

      float weight  = (4.0f * PI * powf(1.0f + g * g - 2.0f * g * angle, 1.5f)) / (1.0f - g * g);
      float density = get_fog_density(device_scene.fog.scattering, task.position.y);
      weight *= density * 0.001f;
      weight *= light.weight;

      RGBF record                        = device_records[pixel];
      device.light_records[pixel]        = scale_color(record, weight);
      device.light_sample_history[pixel] = light.id;

      task.state = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      TraceTask light_task;
      light_task.origin = task.position;
      light_task.ray    = out_ray;
      light_task.index  = task.index;
      light_task.state  = task.state;

      store_trace_task(device.light_trace + get_task_address(light_trace_count++), light_task);
    }
  }

  device.light_trace_count[id]  = light_trace_count;
  device.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 5 + 4];
  const int task_offset = device.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO || device_shading_mode == SHADING_NORMAL) {
      device.frame_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / task.distance) * 2.0f);
      device.frame_buffer[pixel] = get_color(value, value, value);
    }
  }
}

#endif /* CU_FOG_H */
