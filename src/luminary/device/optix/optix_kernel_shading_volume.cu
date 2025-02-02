// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL
#define PHASE_KERNEL
#define VOLUME_KERNEL

#include "bsdf.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset       = get_task_address(task_offset + i);
    DeviceTask task             = task_load(offset);
    const TriangleHandle handle = triangle_handle_load(offset);
    const float depth           = trace_depth_load(offset);
    const uint32_t pixel        = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    const VolumeType volume_type  = VOLUME_HIT_TYPE(handle.instance_id);
    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    GBufferData data = volume_generate_g_buffer(task, handle.instance_id, pixel, volume);

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    // Bounce Ray Sampling
    BSDFSampleInfo bounce_info;
    const vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info);

    uint8_t new_state = task.state & ~(STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION);

    if (volume_type == VOLUME_TYPE_OCEAN) {
      new_state &= ~STATE_FLAG_OCEAN_SCATTERED;
    }

    DeviceTask bounce_task;
    bounce_task.state  = new_state;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    RGBF bounce_record = record;

    // Light Ray Sampling
    RGBF accumulated_light = get_color(0.0f, 0.0f, 0.0f);

    accumulated_light = add_color(accumulated_light, optix_compute_light_ray_sun(data, task.index));
    accumulated_light = add_color(
      accumulated_light,
      optix_compute_light_ray_ambient_sky(data, bounce_ray, bounce_info.weight, bounce_info.is_transparent_pass, task.index));

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel, task.state);

    if (((task.state & STATE_FLAG_DELTA_PATH) != 0) && (device.ocean.triangle_light_contribution || volume_type != VOLUME_TYPE_OCEAN)) {
      RGBF bridge_color = bridges_sample(task, volume);
      bridge_color      = mul_color(bridge_color, record);

      write_beauty_buffer_indirect(bridge_color, pixel);
    }

    // This must be done after the trace rays due to some optimization in the compiler.
    // The compiler reloads these values at some point for some reason and if we overwrite
    // the values we will get garbage. I am not sure if this is a compiler bug or some undefined
    // behaviour on my side.
    if (task_russian_roulette(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records, pixel, bounce_record);
      task_store(bounce_task, get_task_address(trace_count++));
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
