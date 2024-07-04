#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"

//
// In this ocean implementation the surface shape is defined by a function based on the shadertoy by
// Alexander Alekseev aka TDM (https://www.shadertoy.com/view/Ms2SD1).
// The intersection of the ray with the surface is handled through a ray marcher that uses an
// approximate Lipschitz factor of the surface function to obtain a function similar to an SDF.
// The shading of the ocean and the water beneath is based on
// M. Droske, J. Hanika, J. Vorba, A. Weidlich, M. Sabbadin, "Path Tracing in Production: The Path of Water", ACM SIGGRAPH 2023 Courses,
// 2023.
// The water is handled by the volume implementation.
//

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void process_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel  = task.index.y * device.width + task.index.x;

    vec3 normal = ocean_get_normal(task.position);

    const bool inside_water = dot_product(task.ray, normal) > 0.0f;

    const IORStackMethod ior_stack_method = (inside_water) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
    const float ray_ior                   = ior_stack_interact(device.scene.ocean.refractive_index, pixel, ior_stack_method);

    float ior_in, ior_out;
    if (inside_water) {
      normal  = scale_vector(normal, -1.0f);
      ior_in  = device.scene.ocean.refractive_index;
      ior_out = ray_ior;
    }
    else {
      ior_in  = ray_ior;
      ior_out = device.scene.ocean.refractive_index;
    }

    write_normal_buffer(normal, pixel);

    const vec3 V = scale_vector(task.ray, -1.0f);

    bool total_reflection;
    const vec3 refraction_dir = refract_vector(V, normal, ior_in / ior_out, total_reflection);

    vec3 bounce_ray;
    vec3 bias_direction;
    if (total_reflection) {
      bias_direction = V;
      bounce_ray     = reflect_vector(V, normal);
    }
    else {
      const float reflection_coefficient = ocean_reflection_coefficient(normal, task.ray, refraction_dir, ior_in, ior_out);

      if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY, task.index) < reflection_coefficient) {
        bias_direction = V;
        bounce_ray     = reflect_vector(V, normal);
      }
      else {
        bounce_ray     = refraction_dir;
        bias_direction = bounce_ray;

        const IORStackMethod ior_stack_method = (inside_water) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
        ior_stack_interact(ior_out, pixel, ior_stack_method);
      }
    }

    TraceTask new_task;
    new_task.origin = add_vector(task.position, scale_vector(bias_direction, 2.0f * eps * (1.0f + get_length(task.position))));
    new_task.ray    = bounce_ray;
    new_task.index  = task.index;

    store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), new_task);
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void process_debug_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];

  for (int i = 0; i < task_count; i++) {
    ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel  = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = ocean_get_normal(task.position);

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      write_beauty_buffer(get_color(0.0f, 0.0f, 1.0f), pixel, true);
    }
  }
}

#endif /* CU_OCEAN_H */
