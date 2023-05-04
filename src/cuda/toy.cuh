#ifndef CU_TOY_H
#define CU_TOY_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "toy_utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 3];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 3];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 normal     = get_toy_normal(task.position);
    int from_inside = 0;

    if (dot_product(normal, task.ray) > 0.0f) {
      normal      = scale_vector(normal, -1.0f);
      from_inside = 1;
    }

    RGBAF albedo          = device.scene.toy.albedo;
    const float roughness = (1.0f - device.scene.toy.material.r);
    const float metallic  = device.scene.toy.material.g;
    const float intensity = device.scene.toy.material.b;
    RGBAhalf emission     = get_RGBAhalf(device.scene.toy.emission.r, device.scene.toy.emission.g, device.scene.toy.emission.b, 0.0f);
    emission              = scale_RGBAhalf(emission, intensity);

    if (albedo.a < device.scene.material.alpha_cutoff)
      albedo.a = 0.0f;

    if (device.scene.toy.flashlight_mode) {
      const vec3 dir    = normalize_vector(rotate_vector_by_quaternion(get_vector(0.0f, 0.0f, -1.0f), device.emitter.camera_rotation));
      const float angle = -dot_product(dir, task.ray);
      const float dir_intensity = remap01(angle, 0.85f, 1.0f);
      emission                  = scale_RGBAhalf(emission, dir_intensity);
    }

    RGBAhalf record = load_RGBAhalf(device.records + pixel);

    if (albedo.a > 0.0f && device.scene.toy.emissive) {
      emission = scale_RGBAhalf(emission, albedo.a);

      write_albedo_buffer(RGBAhalf_to_RGBF(emission), pixel);

      emission = mul_RGBAhalf(emission, record);

      const uint32_t light = device.ptrs.light_sample_history[pixel];

      if (proper_light_sample(light, LIGHT_ID_TOY)) {
        store_RGBAhalf(device.ptrs.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.ptrs.frame_buffer + pixel), emission));
      }
    }

    write_normal_buffer(normal, pixel);

    const vec3 V      = scale_vector(task.ray, -1.0f);
    BRDFInstance brdf = brdf_get_instance(RGBAF_to_RGBAhalf(albedo), V, normal, roughness, metallic);

    if (albedo.a < 1.0f && white_noise() > albedo.a) {
      task.position = add_vector(task.position, scale_vector(task.ray, 2.0f * eps));

      brdf.term = mul_RGBAhalf(brdf.term, RGBF_to_RGBAhalf(opaque_color(albedo)));

      if (device.scene.toy.refractive_index != 1.0f && device.iteration_type != TYPE_LIGHT) {
        const float alpha = white_noise();
        const float beta  = 2.0f * PI * white_noise();

        const float refraction_index = (from_inside) ? device.scene.toy.refractive_index : 1.0f / device.scene.toy.refractive_index;

        brdf = brdf_sample_ray_refraction(brdf, refraction_index, alpha, beta);
      }
      else {
        brdf.L = task.ray;
      }

      record = mul_RGBAhalf(record, brdf.term);

      TraceTask new_task;
      new_task.origin = task.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBAhalf(device.ptrs.bounce_records + pixel, record);
          store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f)
            break;
          store_RGBAhalf(device.ptrs.light_records + pixel, scale_RGBAhalf(record, 2.0f));
          store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
          device.ptrs.state_buffer[pixel] |= STATE_LIGHT_OCCUPIED;
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      const int is_mirror = material_is_mirror(roughness, metallic);

      if (!is_mirror)
        write_albedo_buffer(get_color(albedo.r, albedo.g, albedo.b), pixel);

      const int use_light_sample = !is_mirror && !(device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

      task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      if (use_light_sample) {
        LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

        const float light_weight = brdf_light_sample_shading_weight(light) * light.solid_angle;

        if (light_weight > 0.0f) {
          brdf.L = brdf_sample_light_ray(light, task.position);

          const RGBAhalf light_record = mul_RGBAhalf(record, scale_RGBAhalf(brdf_evaluate(brdf).term, light_weight));

          TraceTask light_task;
          light_task.origin = task.position;
          light_task.ray    = brdf.L;
          light_task.index  = task.index;

          if (any_RGBAhalf(light_record)) {
            store_RGBAhalf(device.ptrs.light_records + pixel, light_record);
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }
      }

      if (!(device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED))
        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;

      brdf                   = brdf_sample_ray(brdf, task.index);
      RGBAhalf bounce_record = mul_RGBAhalf(record, brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = brdf.L;
      bounce_task.index  = task.index;

      if (validate_trace_task(bounce_task, bounce_record)) {
        store_RGBAhalf(device.ptrs.bounce_records + pixel, bounce_record);
        store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
    }
  }

  device.ptrs.light_trace_count[id]  = light_trace_count;
  device.ptrs.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 3];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 3];

  for (int i = 0; i < task_count; i++) {
    const ToyTask task = load_toy_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      store_RGBAhalf(device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(opaque_color(device.scene.toy.albedo)));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      store_RGBAhalf(device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(get_color(value, value, value)));
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = get_toy_normal(task.position);

      if (dot_product(normal, task.ray) > 0.0f) {
        normal = scale_vector(normal, -1.0f);
      }

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      store_RGBAhalf(
        device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z))));
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const RGBAhalf color = get_RGBAhalf(1.0f, 0.63f, 0.0f, 0.0f);

      store_RGBAhalf(device.ptrs.frame_buffer + pixel, color);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      RGBF color;
      if (device.scene.toy.emissive) {
        color = get_color(100.0f, 100.0f, 100.0f);
      }
      else {
        color = scale_color(opaque_color(device.scene.toy.albedo), 0.1f);
      }

      store_RGBAhalf(device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(color));
    }
  }
}

#endif /* CU_TOY_H */
