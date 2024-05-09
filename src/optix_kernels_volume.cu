#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL
#define VOLUME_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bsdf.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "particle_utils.cuh"
#include "shading_kernel.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel  = task.index.y * device.width + task.index.x;

    const VolumeType volume_type  = VOLUME_HIT_TYPE(task.hit_id);
    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    GBufferData data;
    if (volume_type == VOLUME_TYPE_PARTICLE) {
      data = particle_generate_g_buffer(task, pixel);
      write_albedo_buffer(opaque_color(data.albedo), pixel);
    }
    else {
      data = volume_generate_g_buffer(task, pixel, volume);
      write_albedo_buffer(get_color(0.0f, 0.0f, 0.0f), pixel);
    }

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
    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    const vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    RGBF bounce_record = record;

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
