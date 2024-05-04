#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bsdf.cuh"
#include "directives.cuh"
#include "geometry_utils.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = geometry_generate_g_buffer(task, pixel);

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    RGBF accumulated_light   = get_color(0.0f, 0.0f, 0.0f);
    uint32_t light_ray_index = 0;

    if (device.restir.num_light_rays) {
      for (int j = 0; j < device.restir.num_light_rays; j++) {
        accumulated_light =
          add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_GEOMETRY, light_ray_index++));
      }

      accumulated_light = scale_color(accumulated_light, 1.0f / device.restir.num_light_rays);
    }

    accumulated_light = add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_SUN, light_ray_index++));
    accumulated_light = add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_TOY, light_ray_index++));

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    if (state_peek(pixel, STATE_FLAG_BOUNCE_LIGHTING))
      accumulated_light = add_color(accumulated_light, data.emission);

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    const float shift = (bounce_info.is_transparent_pass) ? -eps : eps;
    data.position     = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_out, pixel, ior_stack_method);
    }

    TraceTask bounce_task;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records + pixel, bounce_record);
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);

      state_release(pixel, STATE_FLAG_BOUNCE_LIGHTING);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
