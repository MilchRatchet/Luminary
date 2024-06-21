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
#include "toy_utils.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel        = task.index.y * device.width + task.index.x;

    GBufferData data;
    if (task.hit_id == HIT_TYPE_TOY) {
      data = toy_generate_g_buffer(task, pixel);
    }
    else {
      data = geometry_generate_g_buffer(task, pixel);
    }

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    const bool include_emission = state_peek(pixel, STATE_FLAG_BOUNCE_LIGHTING);

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    BSDFSampleInfo bounce_info;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info);

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    bool use_light_rays        = false;
    bool allow_bounce_lighting = false;
    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_out, pixel, ior_stack_method);

      const float refraction_scale = (data.ior_in > data.ior_out) ? data.ior_in / data.ior_out : data.ior_out / data.ior_in;
      use_light_rays |= data.roughness * (refraction_scale - 1.0f) > 0.05f;
      allow_bounce_lighting = !use_light_rays;
    }
    else {
      use_light_rays |= ((data.metallic < 1.0f && data.albedo.a > 0.0f) || data.roughness > 0.05f);
      allow_bounce_lighting |= bounce_info.is_microfacet_based && data.roughness <= 0.05f;
    }

    if (include_emission) {
      const RGBF emission = mul_color(data.emission, record);

      write_beauty_buffer(emission, pixel);
    }

    RGBF accumulated_light = get_color(0.0f, 0.0f, 0.0f);

    if (use_light_rays) {
      if (device.ris_settings.num_light_rays) {
        for (int j = 0; j < device.ris_settings.num_light_rays; j++) {
          accumulated_light = add_color(accumulated_light, optix_compute_light_ray_geometry(data, task.index, j));
        }

        accumulated_light = scale_color(accumulated_light, 1.0f / device.ris_settings.num_light_rays);
      }

      accumulated_light = add_color(accumulated_light, optix_compute_light_ray_sun(data, task.index));
      accumulated_light = add_color(accumulated_light, optix_compute_light_ray_toy(data, task.index));
    }

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel);

    const float shift = (bounce_info.is_transparent_pass) ? -eps : eps;
    data.position     = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

    TraceTask bounce_task;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    // This must be done after the trace rays due to some optimization in the compiler.
    // The compiler reloads these values at some point for some reason and if we overwrite
    // the values we will get garbage. I am not sure if this is a compiler bug or some undefined
    // behaviour on my side.
    if (validate_trace_task(bounce_task, bounce_record)) {
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);
      store_RGBF(device.ptrs.records + pixel, bounce_record);

      if (!allow_bounce_lighting) {
        state_release(pixel, STATE_FLAG_BOUNCE_LIGHTING);
        state_release(pixel, STATE_FLAG_DELTA_PATH);
      }
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
