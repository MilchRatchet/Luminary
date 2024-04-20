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

__device__ vec3 ocean_get_normal(const vec3 p) {
  const float d = (OCEAN_LIPSCHITZ + get_length(p)) * eps;

  // Sobel filter
  float h[8];
  h[0] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[1] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[2] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[3] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[4] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[5] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[6] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[7] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);

  vec3 normal;
  normal.x = -((h[7] + 2.0f * h[4] + h[2]) - (h[5] + 2.0f * h[3] + h[0])) / 8.0f;
  normal.y = d;
  normal.z = -((h[0] + 2.0f * h[1] + h[2]) - (h[5] + 2.0f * h[6] + h[7])) / 8.0f;

  return normalize_vector(normal);
}

/*
 * This uses the actual Fresnel equations to compute the reflection coefficient under the following assumptions:
 *  - The media are not magnetic.
 *  - The light is not polarized.
 *  - The IORs are wavelength independent.
 */
__device__ float ocean_reflection_coefficient(
  const vec3 normal, const vec3 ray, const vec3 refraction, const float index_in, const float index_out) {
  const float NdotV = -dot_product(ray, normal);
  const float NdotT = -dot_product(refraction, normal);

  const float s_pol_term1 = index_in * NdotV;
  const float s_pol_term2 = index_out * NdotT;

  const float p_pol_term1 = index_in * NdotT;
  const float p_pol_term2 = index_out * NdotV;

  float reflection_s_pol = (s_pol_term1 - s_pol_term2) / (s_pol_term1 + s_pol_term2);
  float reflection_p_pol = (p_pol_term1 - p_pol_term2) / (p_pol_term1 + p_pol_term2);

  reflection_s_pol *= reflection_s_pol;
  reflection_p_pol *= reflection_p_pol;

  return __saturatef(0.5f * (reflection_s_pol + reflection_p_pol));
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void process_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

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

    const vec3 refraction_dir = refract_vector(task.ray, normal, ior_in / ior_out);

    const float reflection_coefficient = ocean_reflection_coefficient(normal, task.ray, refraction_dir, ior_in, ior_out);

    vec3 bounce_ray;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY, task.index) < reflection_coefficient) {
      task.position = add_vector(task.position, scale_vector(task.ray, -2.0f * eps * (1.0f + get_length(task.position))));
      bounce_ray    = reflect_vector(task.ray, normal);
    }
    else {
      bounce_ray    = refraction_dir;
      task.position = add_vector(task.position, scale_vector(bounce_ray, 2.0f * eps * (1.0f + get_length(task.position))));

      const IORStackMethod ior_stack_method = (inside_water) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(ior_in, pixel, ior_stack_method);
    }

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    TraceTask new_task;
    new_task.origin = task.position;
    new_task.ray    = bounce_ray;
    new_task.index  = task.index;

    state_consume(pixel, STATE_FLAG_BOUNCE_LIGHTING);

    store_RGBF(device.ptrs.records + pixel, record);
    store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), new_task);
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void process_debug_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

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
