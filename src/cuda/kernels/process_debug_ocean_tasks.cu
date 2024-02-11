#include "math.cuh"
#include "memory.cuh"
#include "ocean.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = ocean_get_normal(task.position);

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      write_beauty_buffer(get_color(0.0f, 0.0f, 1.0f), pixel, true);
    }
  }
}
