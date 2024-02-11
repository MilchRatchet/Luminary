#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void postprocess_trace_tasks() {
  const int task_count         = device.trace_count[THREAD_ID];
  uint16_t geometry_task_count = 0;
  uint16_t particle_task_count = 0;
  uint16_t sky_task_count      = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t toy_task_count      = 0;
  uint16_t volume_task_count   = 0;

  // count data
  for (int i = 0; i < task_count; i++) {
    const int offset      = get_task_address(i);
    const uint32_t hit_id = __ldca((uint32_t*) (device.ptrs.trace_results + offset) + 1);

    if (hit_id == HIT_TYPE_SKY) {
      sky_task_count++;
    }
    else if (hit_id == HIT_TYPE_OCEAN) {
      ocean_task_count++;
    }
    else if (hit_id == HIT_TYPE_TOY) {
      toy_task_count++;
    }
    else if (VOLUME_HIT_CHECK(hit_id)) {
      volume_task_count++;
    }
    else if (hit_id <= HIT_TYPE_PARTICLE_MAX && hit_id >= HIT_TYPE_PARTICLE_MIN) {
      particle_task_count++;
    }
    else if (hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      geometry_task_count++;
    }
  }

  int geometry_offset = 0;
  int particle_offset = geometry_offset + geometry_task_count;
  int ocean_offset    = particle_offset + particle_task_count;
  int sky_offset      = ocean_offset + ocean_task_count;
  int toy_offset      = sky_offset + sky_task_count;
  int volume_offset   = toy_offset + toy_task_count;
  int rejects_offset  = volume_offset + volume_task_count;
  int k               = 0;

  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE] = particle_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN]    = ocean_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY]      = sky_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_TOY]      = toy_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]   = volume_offset;

  const int num_tasks               = rejects_offset;
  const int initial_geometry_offset = geometry_offset;
  const int initial_particle_offset = particle_offset;
  const int initial_ocean_offset    = ocean_offset;
  const int initial_sky_offset      = sky_offset;
  const int initial_toy_offset      = toy_offset;
  const int initial_volume_offset   = volume_offset;
  const int initial_rejects_offset  = rejects_offset;

  // order data
  while (k < task_count) {
    const int offset      = get_task_address(k);
    const uint32_t hit_id = __ldca((uint32_t*) (device.ptrs.trace_results + offset) + 1);

    int index;
    int needs_swapping;

    if (hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      index          = geometry_offset;
      needs_swapping = (k < initial_geometry_offset) || (k >= geometry_task_count + initial_geometry_offset);
      if (needs_swapping || k >= geometry_offset) {
        geometry_offset++;
      }
    }
    else if (hit_id <= HIT_TYPE_PARTICLE_MAX) {
      index          = particle_offset;
      needs_swapping = (k < initial_particle_offset) || (k >= particle_task_count + initial_particle_offset);
      if (needs_swapping || k >= particle_offset) {
        particle_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_OCEAN) {
      index          = ocean_offset;
      needs_swapping = (k < initial_ocean_offset) || (k >= ocean_task_count + initial_ocean_offset);
      if (needs_swapping || k >= ocean_offset) {
        ocean_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_SKY) {
      index          = sky_offset;
      needs_swapping = (k < initial_sky_offset) || (k >= sky_task_count + initial_sky_offset);
      if (needs_swapping || k >= sky_offset) {
        sky_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_TOY) {
      index          = toy_offset;
      needs_swapping = (k < initial_toy_offset) || (k >= toy_task_count + initial_toy_offset);
      if (needs_swapping || k >= toy_offset) {
        toy_offset++;
      }
    }
    else if (VOLUME_HIT_CHECK(hit_id)) {
      index          = volume_offset;
      needs_swapping = (k < initial_volume_offset) || (k >= volume_task_count + initial_volume_offset);
      if (needs_swapping || k >= volume_offset) {
        volume_offset++;
      }
    }
    else {
      index          = rejects_offset;
      needs_swapping = (k < initial_rejects_offset);
      if (needs_swapping || k >= rejects_offset) {
        rejects_offset++;
      }
    }

    if (needs_swapping) {
      swap_trace_data(k, index);
    }
    else {
      k++;
    }
  }

  if (device.iteration_type == TYPE_LIGHT) {
    for (int i = 0; i < task_count; i++) {
      const int offset     = get_task_address(i);
      TraceTask task       = load_trace_task(device.trace_tasks + offset);
      const uint32_t pixel = get_pixel_id(task.index.x, task.index.y);
      state_release(pixel, STATE_FLAG_LIGHT_OCCUPIED);
    }
  }

  // process data
  for (int i = 0; i < num_tasks; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);
    const uint32_t pixel  = get_pixel_id(task.index.x, task.index.y);

    if (is_first_ray()) {
      device.ptrs.raydir_buffer[pixel] = task.ray;

      TraceResult trace_result;
      trace_result.depth  = depth;
      trace_result.hit_id = hit_id;

      device.ptrs.trace_result_buffer[pixel] = trace_result;
    }

    if (hit_id != HIT_TYPE_SKY)
      task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    float4* ptr = (float4*) (device.trace_tasks + offset);
    float4 data0;
    float4 data1;

    data0.x = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
    data0.y = task.origin.x;
    data0.z = task.origin.y;
    data0.w = task.origin.z;

    __stcs(ptr, data0);

    data1.x = task.ray.x;
    data1.y = task.ray.y;
    data1.z = task.ray.z;
    data1.w = __uint_as_float(hit_id);

    __stcs(ptr + 1, data1);
  }

  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY]   = geometry_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE]   = particle_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN]      = ocean_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY]        = sky_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOY]        = toy_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]     = volume_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOTALCOUNT] = num_tasks;

  device.trace_count[THREAD_ID] = 0;
}
