#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bsdf.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_TOY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    const GBufferData data = toy_generate_g_buffer(task, pixel);

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    const bool include_emission = state_peek(pixel, STATE_FLAG_BOUNCE_LIGHTING);

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    const float ior_to_store              = data.ior_out;
    const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    const float shift = (bounce_info.is_transparent_pass) ? -eps : eps;
    task.position     = add_vector(task.position, scale_vector(data.V, shift * get_length(task.position)));

    bool use_light_rays = false;
    if (bounce_info.is_transparent_pass) {
      use_light_rays |= data.ior_in != data.ior_out && data.roughness > 0.05f;
    }
    else {
      use_light_rays |= !include_emission;
      use_light_rays |= !bounce_info.is_microfacet_based || data.roughness > 0.05f;
    }

    TraceTask bounce_task;
    bounce_task.origin = task.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records + pixel, bounce_record);
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);
    }

    if (use_light_rays) {
      state_release(pixel, STATE_FLAG_BOUNCE_LIGHTING);
    }

    RGBF light_color = get_color(0.0f, 0.0f, 0.0f);

    if (use_light_rays) {
      uint32_t light_ray_index = 0;
      GBufferData light_data;

      if (device.restir.num_light_rays) {
        for (int j = 0; j < device.restir.num_light_rays; j++) {
          light_data = toy_generate_g_buffer(task, pixel);
          light_data.flags |= (bounce_info.is_transparent_pass) ? G_BUFFER_IS_TRANSPARENT_PASS : 0;
          light_color =
            add_color(light_color, optix_compute_light_ray(light_data, task.index, LIGHT_RAY_TARGET_GEOMETRY, light_ray_index++));
        }

        light_color = scale_color(light_color, 1.0f / device.restir.num_light_rays);
      }

      light_data = toy_generate_g_buffer(task, pixel);
      light_data.flags |= (bounce_info.is_transparent_pass) ? G_BUFFER_IS_TRANSPARENT_PASS : 0;
      light_color = add_color(light_color, optix_compute_light_ray(light_data, task.index, LIGHT_RAY_TARGET_SUN, light_ray_index++));

      light_data = toy_generate_g_buffer(task, pixel);
      light_data.flags |= (bounce_info.is_transparent_pass) ? G_BUFFER_IS_TRANSPARENT_PASS : 0;
      light_color = add_color(light_color, optix_compute_light_ray(light_data, task.index, LIGHT_RAY_TARGET_TOY, light_ray_index++));

      const float side_prob =
        (bounce_info.is_transparent_pass) ? bounce_info.transparent_pass_prob : (1.0f - bounce_info.transparent_pass_prob);

      light_color = scale_color(light_color, 1.0f / side_prob);
    }

    if (include_emission)
      light_color = add_color(light_color, data.emission);

    light_color = mul_color(light_color, record);

    write_beauty_buffer(light_color, pixel);

    if (bounce_info.is_transparent_pass) {
      ior_stack_interact(ior_to_store, pixel, ior_stack_method);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
