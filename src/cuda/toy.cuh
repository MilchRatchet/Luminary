#ifndef CU_TOY_H
#define CU_TOY_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "state.cuh"
#include "toy_utils.cuh"

__global__ void toy_generate_g_buffer() {
  const int task_count  = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 3];
  const int task_offset = device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 3];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 normal = get_toy_normal(task.position);

    if (dot_product(normal, task.ray) > 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }

    uint32_t flags = 0;

    if (device.scene.toy.albedo.a < 1.0f && white_noise() > device.scene.toy.albedo.a) {
      flags |= G_BUFFER_TRANSPARENT_PASS;
    }

    if (!(flags & G_BUFFER_TRANSPARENT_PASS) && !state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      flags |= G_BUFFER_REQUIRES_SAMPLING;
    }

    if (flags & G_BUFFER_TRANSPARENT_PASS) {
      task.position = add_vector(task.position, scale_vector(task.ray, eps * get_length(task.position)));
    }
    else {
      task.position = add_vector(task.position, scale_vector(task.ray, -eps * get_length(task.position)));
    }

    RGBF emission;
    if (device.scene.toy.emissive) {
      emission = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
    }
    else {
      emission = get_color(0.0f, 0.0f, 0.0f);
    }

    GBufferData data;
    data.hit_id    = TOY_HIT;
    data.albedo    = device.scene.toy.albedo;
    data.emission  = emission;
    data.normal    = normal;
    data.position  = task.position;
    data.V         = scale_vector(task.ray, -1.0f);
    data.roughness = (1.0f - device.scene.toy.material.r);
    data.metallic  = device.scene.toy.material.g;
    data.flags     = flags;

    store_g_buffer_data(data, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 3];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 3];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    const GBufferData data = load_g_buffer_data(pixel);

    const bool is_inside = toy_is_inside(task.position);
    const vec3 normal    = data.normal;

    RGBAF albedo          = device.scene.toy.albedo;
    const float roughness = (1.0f - device.scene.toy.material.r);
    const float metallic  = device.scene.toy.material.g;
    const float intensity = device.scene.toy.material.b;
    RGBF emission         = get_color(device.scene.toy.emission.r, device.scene.toy.emission.g, device.scene.toy.emission.b);
    emission              = scale_color(emission, intensity);

    if (albedo.a < device.scene.material.alpha_cutoff)
      albedo.a = 0.0f;

    if (device.scene.toy.flashlight_mode) {
      const vec3 dir    = normalize_vector(rotate_vector_by_quaternion(get_vector(0.0f, 0.0f, -1.0f), device.emitter.camera_rotation));
      const float angle = -dot_product(dir, task.ray);
      const float dir_intensity = remap01(angle, 0.85f, 1.0f);
      emission                  = scale_color(emission, dir_intensity);
    }

    RGBF record = load_RGBF(device.records + pixel);

    if (albedo.a > 0.0f && device.scene.toy.emissive) {
      emission = scale_color(emission, albedo.a);

      write_albedo_buffer(emission, pixel);

      emission = mul_color(emission, record);

      if (device.iteration_type == TYPE_BOUNCE) {
        const float mis_weight = device.ptrs.mis_buffer[pixel];
        emission               = scale_color(emission, mis_weight);
      }

      const uint32_t light = device.ptrs.light_sample_history[pixel];

      if (proper_light_sample(light, LIGHT_ID_TOY)) {
        store_RGBF(device.ptrs.frame_buffer + pixel, add_color(load_RGBF(device.ptrs.frame_buffer + pixel), emission));
      }
    }

    write_normal_buffer(normal, pixel);

    BRDFInstance brdf = brdf_get_instance(albedo, data.V, normal, roughness, metallic);

    if (data.flags & G_BUFFER_TRANSPARENT_PASS) {
      brdf.term = mul_color(brdf.term, opaque_color(albedo));

      if (device.scene.toy.refractive_index != 1.0f && device.iteration_type != TYPE_LIGHT) {
        const float refraction_index = (is_inside) ? device.scene.toy.refractive_index : 1.0f / device.scene.toy.refractive_index;

        brdf = brdf_sample_ray_refraction(brdf, refraction_index, white_noise(), white_noise());
      }
      else {
        brdf.L = task.ray;
      }

      record = mul_color(record, brdf.term);

      TraceTask new_task;
      new_task.origin = data.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBF(device.ptrs.bounce_records + pixel, record);
          store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f) {
            if (state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
              store_RGBF(device.ptrs.light_records + pixel, scale_color(record, 2.0f));
              store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
            }
          }
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      if (!material_is_mirror(roughness, metallic))
        write_albedo_buffer(get_color(albedo.r, albedo.g, albedo.b), pixel);

      bool bounce_is_specular;
      BRDFInstance bounce_brdf = brdf_sample_ray(brdf, bounce_is_specular);

      float bounce_mis_weight = 1.0f;

      if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
        uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

        LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

        if (light.weight > 0.0f) {
          BRDFInstance brdf_sample = brdf_apply_sample(brdf, light, data.position);

          const RGBF light_record = mul_color(record, brdf_sample.term);

          TraceTask light_task;
          light_task.origin = data.position;
          light_task.ray    = brdf_sample.L;
          light_task.index  = task.index;

          if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
            const float light_mis_weight = (bounce_is_specular) ? data.roughness * data.roughness : 1.0f;
            bounce_mis_weight            = 1.0f - light_mis_weight;

            store_RGBF(device.ptrs.light_records + pixel, scale_color(light_record, light_mis_weight));
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }

        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
      }

      RGBF bounce_record = mul_color(record, bounce_brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = data.position;
      bounce_task.ray    = bounce_brdf.L;
      bounce_task.index  = task.index;

      if (validate_trace_task(bounce_task, bounce_record)) {
        device.ptrs.mis_buffer[pixel] = bounce_mis_weight;
        store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
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
      store_RGBF(device.ptrs.frame_buffer + pixel, opaque_color(device.scene.toy.albedo));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(value, value, value));
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = get_toy_normal(task.position);

      if (dot_product(normal, task.ray) > 0.0f) {
        normal = scale_vector(normal, -1.0f);
      }

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(1.0f, 0.63f, 0.0f));
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      RGBF color;
      if (device.scene.toy.emissive) {
        color = get_color(100.0f, 100.0f, 100.0f);
      }
      else {
        color = scale_color(opaque_color(device.scene.toy.albedo), 0.1f);
      }

      store_RGBF(device.ptrs.frame_buffer + pixel, color);
    }
  }
}

#endif /* CU_TOY_H */
