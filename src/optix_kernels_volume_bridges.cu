#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL
#define PHASE_KERNEL
#define VOLUME_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bridges.cuh"
#include "bsdf.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);

    const int pixel = task.index.y * device.width + task.index.x;

    if (!state_peek(pixel, STATE_FLAG_DELTA_PATH))
      continue;

    float start            = FLT_MAX;
    VolumeType volume_type = VOLUME_TYPE_NONE;

    if (device.scene.fog.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.y > 0.0f && path.x < start) {
        volume_type = VOLUME_TYPE_FOG;
        start       = path.x;
      }
    }

    if (device.scene.ocean.active && device.scene.ocean.triangle_light_contribution) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.y > 0.0f && path.x < start) {
        volume_type = VOLUME_TYPE_OCEAN;
        start       = path.x;
      }
    }

    if (volume_type == VOLUME_TYPE_NONE)
      continue;

    task.origin = add_vector(task.origin, scale_vector(task.ray, start * (1.0f - 8.0f * eps)));

    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    const float ior = ior_stack_interact(1.0f, pixel, IOR_STACK_METHOD_PEEK_CURRENT);

    RGBF light_color  = bridges_sample(task, volume, depth, ior);
    const RGBF record = load_RGBF(device.ptrs.records + pixel);
    light_color       = mul_color(light_color, record);

    write_beauty_buffer_indirect(light_color, pixel);
  }
}
