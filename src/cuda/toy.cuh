#ifndef CU_TOY_H
#define CU_TOY_H

#include "bsdf.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "state.cuh"
#include "toy_utils.cuh"

__device__ GBufferData toy_generate_g_buffer(const ToyTask task, const int pixel) {
  vec3 normal = get_toy_normal(task.position);

  if (dot_product(normal, task.ray) > 0.0f) {
    normal = scale_vector(normal, -1.0f);
  }

  uint32_t flags = G_BUFFER_REQUIRES_SAMPLING;

  RGBF emission;
  if (device.scene.toy.emissive) {
    emission = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
  }
  else {
    emission = get_color(0.0f, 0.0f, 0.0f);
  }

  if (toy_is_inside(task.position)) {
    flags |= G_BUFFER_REFRACTION_IS_INSIDE;
  }

  const IORStackMethod ior_stack_method =
    (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior = ior_stack_interact(device.scene.toy.refractive_index, pixel, ior_stack_method);

  GBufferData data;
  data.hit_id             = HIT_TYPE_TOY;
  data.albedo             = device.scene.toy.albedo;
  data.emission           = emission;
  data.normal             = normal;
  data.position           = task.position;
  data.V                  = scale_vector(task.ray, -1.0f);
  data.roughness          = (1.0f - device.scene.toy.material.r);
  data.metallic           = device.scene.toy.material.g;
  data.flags              = flags;
  data.ior_in             = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? device.scene.toy.refractive_index : ray_ior;
  data.ior_out            = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ray_ior : device.scene.toy.refractive_index;
  data.colored_dielectric = 1;

  return data;
}

LUMINARY_KERNEL void process_toy_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_TOY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    GBufferData data = toy_generate_g_buffer(task, pixel);

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    if (data.albedo.a > 0.0f && color_any(data.emission)) {
      write_albedo_buffer(add_color(data.emission, opaque_color(data.albedo)), pixel);

      if (state_peek(pixel, STATE_FLAG_BOUNCE_LIGHTING)) {
        RGBF emission = mul_color(data.emission, record);
        write_beauty_buffer(emission, pixel);
      }
    }

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    // uint32_t light_history_buffer_entry = LIGHT_ID_ANY;
    //
    // LightSample light = restir_sample_reservoir(data, record, task.index);
    //
    // if (light.weight > 0.0f) {
    //  RGBF light_weight;
    //  bool is_transparent_pass;
    //  const vec3 light_ray = restir_apply_sample_shading(data, light, task.index, light_weight, is_transparent_pass);
    //
    //  const RGBF light_record = mul_color(record, light_weight);
    //
    //  const float shift           = (is_transparent_pass) ? -eps : eps;
    //  const vec3 shifted_position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));
    //
    //  TraceTask light_task;
    //  light_task.origin = shifted_position;
    //  light_task.ray    = light_ray;
    //  light_task.index  = task.index;
    //
    //  const float light_mis_weight = mis_weight_light_sampled(data, light_ray, bounce_info, light);
    //  store_RGBF(device.ptrs.records + pixel, scale_color(light_record, light_mis_weight));
    //  light_history_buffer_entry = light.id;
    //  store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
    //}
    //
    // device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    const float shift = (bounce_info.is_transparent_pass) ? -eps : eps;
    data.position     = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_in, pixel, ior_stack_method);
    }

    TraceTask bounce_task;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records + pixel, bounce_record);
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);

      state_release(pixel, STATE_FLAG_BOUNCE_LIGHTING);

      // MISData mis_data;
      // mis_data.light_target_pdf_normalization = light.target_pdf_normalization;
      // mis_data.bsdf_marginal                  = bsdf_marginal;
      //
      // mis_store_data(data, record, mis_data, bounce_ray, pixel);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

// LUMINARY_KERNEL void process_toy_light_tasks() {
//   const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOY];
//   const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_TOY];
//
//   for (int i = 0; i < task_count; i++) {
//     ToyTask task    = load_toy_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
//     const int pixel = task.index.y * device.width + task.index.x;
//
//     GBufferData data = toy_generate_g_buffer(task, pixel);
//
//     RGBF record = load_RGBF(device.ptrs.records + pixel);
//
//     if (color_any(data.emission)) {
//       const uint32_t light = device.ptrs.light_sample_history[pixel];
//
//       if (proper_light_sample(light, LIGHT_ID_TOY)) {
//         write_beauty_buffer(mul_color(data.emission, record), pixel);
//       }
//     }
//   }
// }

LUMINARY_KERNEL void process_debug_toy_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_TOY];

  for (int i = 0; i < task_count; i++) {
    const ToyTask task = load_toy_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      write_beauty_buffer(opaque_color(device.scene.toy.albedo), pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = get_toy_normal(task.position);

      if (dot_product(normal, task.ray) > 0.0f) {
        normal = scale_vector(normal, -1.0f);
      }

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      write_beauty_buffer(get_color(1.0f, 0.63f, 0.0f), pixel, true);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      RGBF color;
      if (device.scene.toy.emissive) {
        color = get_color(100.0f, 100.0f, 100.0f);
      }
      else {
        color = scale_color(opaque_color(device.scene.toy.albedo), 0.1f);
      }

      write_beauty_buffer(color, pixel, true);
    }
  }
}

#endif /* CU_TOY_H */
