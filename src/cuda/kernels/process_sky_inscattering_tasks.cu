#include "memory.cuh"
#include "sky.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 6) void process_sky_inscattering_tasks() {
  const int task_count = device.trace_count[THREAD_ID];

  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);

    if (hit_id == HIT_TYPE_SKY) {
      continue;
    }

    const int pixel = task.index.y * device.width + task.index.x;

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float inscattering_limit = world_to_sky_scale(depth);

    RGBF record = load_RGBF(device.records + pixel);

    const RGBF inscattering = sky_trace_inscattering(sky_origin, task.ray, inscattering_limit, record, seed);

    store_RGBF(device.records + pixel, record);
    write_beauty_buffer(inscattering, pixel);
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}
