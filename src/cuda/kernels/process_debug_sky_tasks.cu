#include "math.cuh"
#include "memory.cuh"
#include "sky.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_debug_sky_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY];

  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      RGBF sky;
      if (device.scene.sky.hdri_active) {
        sky = sky_hdri_sample(task.ray, device.scene.sky.hdri_mip_bias);
      }
      else {
        sky = sky_get_color(world_to_sky_transform(task.origin), task.ray, FLT_MAX, true, device.scene.sky.steps, seed);
      }
      write_beauty_buffer(sky, pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float value = __saturatef((1.0f / device.scene.camera.far_clip_distance) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      write_beauty_buffer(get_color(0.0f, 0.63f, 1.0f), pixel, true);
    }
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}
