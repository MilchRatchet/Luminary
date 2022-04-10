#ifndef CU_TOY_H
#define CU_TOY_H

#include "math.cuh"

/*
 * Requirement:
 *      - Position should be the middle of the shape
 *      - Scale should be the radius of the shape
 */

__device__ float get_toy_distance(const vec3 origin, const vec3 ray) {
  switch (device_scene.toy.shape) {
    case TOY_SPHERE:
      return sphere_ray_intersection(ray, origin, device_scene.toy.position, device_scene.toy.scale);
  }

  return FLT_MAX;
}

__device__ vec3 toy_sphere_normal(const vec3 position) {
  const vec3 diff = sub_vector(position, device_scene.toy.position);
  vec3 normal     = normalize_vector(diff);

  if (get_length(diff) < device_scene.toy.scale)
    normal = scale_vector(normal, -1.0f);

  return normal;
}

__device__ vec3 get_toy_normal(const vec3 position) {
  switch (device_scene.toy.shape) {
    case TOY_SPHERE:
      return toy_sphere_normal(position);
  }

  return normalize_vector(position);
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.task_counts[id * 6 + 3];
  const int task_offset  = device.task_offsets[id * 5 + 3];
  int light_trace_count  = device.light_trace_count[id];
  int bounce_trace_count = device.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    vec3 normal     = get_toy_normal(task.position);
    int from_inside = 0;

    if (dot_product(normal, task.ray) > 0.0f) {
      normal      = scale_vector(normal, -1.0f);
      from_inside = 1;
    }

    RGBAF albedo          = device_scene.toy.albedo;
    const float roughness = (1.0f - device_scene.toy.material.r);
    const float metallic  = device_scene.toy.material.g;
    const float intensity = device_scene.toy.material.b;
    RGBF emission         = get_color(device_scene.toy.emission.r, device_scene.toy.emission.g, device_scene.toy.emission.b);
    emission              = scale_color(emission, intensity);

    if (albedo.a < device_scene.material.alpha_cutoff)
      albedo.a = 0.0f;

    if (device_scene.toy.flashlight_mode) {
      const vec3 dir            = normalize_vector(rotate_vector_by_quaternion(get_vector(0.0f, 0.0f, -1.0f), device_camera_rotation));
      const float angle         = -dot_product(dir, task.ray);
      const float dir_intensity = remap01(angle, 0.85f, 1.0f);
      emission                  = scale_color(emission, dir_intensity);
    }

    RGBF record = device_records[pixel];

    if (albedo.a > 0.0f && device_scene.toy.emissive) {
      write_albedo_buffer(emission, pixel);

      if (!isnan(record.r) && !isinf(record.r) && !isnan(record.g) && !isinf(record.g) && !isnan(record.b) && !isinf(record.b)) {
        emission = mul_color(emission, record);

        const uint32_t light = device.light_sample_history[pixel];

        if (proper_light_sample(light, LIGHT_ID_TOY)) {
          device.frame_buffer[pixel] = add_color(device.frame_buffer[pixel], emission);
        }
      }
    }

    if (white_noise() > albedo.a) {
      task.position = add_vector(task.position, scale_vector(task.ray, 2.0f * eps));

      record.r *= (albedo.r * albedo.a + 1.0f - albedo.a);
      record.g *= (albedo.g * albedo.a + 1.0f - albedo.a);
      record.b *= (albedo.b * albedo.a + 1.0f - albedo.a);

      const float alpha = blue_noise(task.index.x, task.index.y, task.state, 2);
      const float beta  = 2.0f * PI * blue_noise(task.index.x, task.index.y, task.state, 3);

      if (device_scene.toy.refractive_index != 1.0f) {
        const float refraction_index = (from_inside) ? device_scene.toy.refractive_index : 1.0f / device_scene.toy.refractive_index;

        task.ray = brdf_sample_ray_refraction(record, opaque_color(albedo), normal, task.ray, roughness, refraction_index, alpha, beta);
      }

      TraceTask new_task;
      new_task.origin = task.position;
      new_task.ray    = task.ray;
      new_task.index  = task.index;
      new_task.state  = task.state;

      switch (device_iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          device.bounce_records[pixel] = record;
          store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f)
            break;
          device.light_records[pixel] = scale_color(record, 2.0f);
          device.state_buffer[pixel] |= STATE_LIGHT_OCCUPIED;
          store_trace_task(device.light_trace + get_task_address(light_trace_count++), new_task);
          break;
      }
    }
    else if (device_iteration_type != TYPE_LIGHT) {
      write_albedo_buffer(get_color(albedo.r, albedo.g, albedo.b), pixel);

      const vec3 V               = scale_vector(task.ray, -1.0f);
      const int use_light_sample = (roughness > 0.1f || metallic < 0.9f) && !(device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

      task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));
      task.state    = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      if (use_light_sample) {
        LightSample light = load_light_sample(device.light_samples, pixel);

        light = brdf_finalize_light_sample(light, task.position);

        if (light.weight > 0.0f) {
          task.ray = brdf_sample_light_ray(light, task.position);

          const RGBF light_record =
            mul_color(record, scale_color(brdf_evaluate(opaque_color(albedo), V, task.ray, normal, roughness, metallic), light.weight));

          TraceTask light_task;
          light_task.origin = task.position;
          light_task.ray    = task.ray;
          light_task.index  = task.index;
          light_task.state  = task.state;

          if (color_any(light_record)) {
            device.light_records[pixel] = light_record;
            light_history_buffer_entry  = light.id;
            store_trace_task(device.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }
      }

      if (!(device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED))
        device.light_sample_history[pixel] = light_history_buffer_entry;

      RGBF bounce_record = record;

      const bool valid_bounce =
        brdf_sample_ray(task.ray, bounce_record, task.index, task.state, opaque_color(albedo), V, normal, normal, roughness, metallic);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = task.ray;
      bounce_task.index  = task.index;
      bounce_task.state  = task.state;

      if (valid_bounce && validate_trace_task(bounce_task, bounce_record)) {
        device.bounce_records[pixel] = bounce_record;
        store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
    }
  }

  device.light_trace_count[id]  = light_trace_count;
  device.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 6 + 3];
  const int task_offset = device.task_offsets[id * 5 + 3];

  for (int i = 0; i < task_count; i++) {
    const ToyTask task = load_toy_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      device.frame_buffer[pixel] = get_color(device_scene.toy.albedo.r, device_scene.toy.albedo.g, device_scene.toy.albedo.b);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float dist           = get_length(sub_vector(device_scene.camera.pos, task.position));
      const float value          = __saturatef((1.0f / dist) * 2.0f);
      device.frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      vec3 normal = get_toy_normal(task.position);

      if (dot_product(normal, task.ray) > 0.0f) {
        normal = scale_vector(normal, -1.0f);
      }

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      device.frame_buffer[pixel] = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
    }
  }
}

#endif /* CU_TOY_H */
