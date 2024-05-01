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
