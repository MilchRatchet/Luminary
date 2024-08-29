#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL
#define PHASE_KERNEL
#define VOLUME_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bsdf.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
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

    if (state_peek(pixel, STATE_FLAG_SKIP_BRIDGE_SAMPLING))
      continue;

    const VolumeType volume_type = volume_get_type_at_position(task.origin);

    if (volume_type == VOLUME_TYPE_NONE)
      continue;

    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    const float ior = ior_stack_interact(1.0f, pixel, IOR_STACK_METHOD_PEEK_CURRENT);

    RGBF light_color  = optix_compute_light_ray_geo(task, volume, depth, ior);
    const RGBF record = load_RGBF(device.ptrs.records + pixel);
    light_color       = mul_color(light_color, record);

    write_beauty_buffer(light_color, pixel);
  }
}
