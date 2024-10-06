// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL
#define PHASE_KERNEL
#define PARTICLE_KERNEL

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
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const ShadingTask task            = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const ShadingTaskAuxData aux_data = load_shading_task_aux_data(device.ptrs.aux_data + get_task_address(task_offset + i));
    const int pixel                   = get_pixel_id(task.index);

    const VolumeType volume_type  = VOLUME_HIT_TYPE(task.instance_id);
    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    GBufferData data = particle_generate_g_buffer(task, aux_data, pixel);

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    // Bounce Ray Sampling
    BSDFSampleInfo bounce_info;
    const vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info);

    TraceTask bounce_task;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    RGBF bounce_record = record;

    // Light Ray Sampling
    RGBF accumulated_light = get_color(0.0f, 0.0f, 0.0f);

    accumulated_light = add_color(accumulated_light, optix_compute_light_ray_geo(data, task.index));
    accumulated_light = add_color(accumulated_light, optix_compute_light_ray_sun(data, task.index));
    accumulated_light = add_color(accumulated_light, optix_compute_light_ray_toy(data, task.index));
    accumulated_light = add_color(
      accumulated_light,
      optix_compute_light_ray_ambient_sky(data, bounce_ray, bounce_info.weight, bounce_info.is_transparent_pass, task.index));

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel);

    // This must be done after the trace rays due to some optimization in the compiler.
    // The compiler reloads these values at some point for some reason and if we overwrite
    // the values we will get garbage. I am not sure if this is a compiler bug or some undefined
    // behaviour on my side.
    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records + pixel, bounce_record);
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);

      state_release(pixel, STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
