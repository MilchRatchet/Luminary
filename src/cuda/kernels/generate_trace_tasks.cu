#include "camera.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void generate_trace_tasks() {
  int offset       = 0;
  const int amount = device.width * device.height;

  for (int pixel = THREAD_ID; pixel < amount; pixel += blockDim.x * gridDim.x) {
    TraceTask task;

    task.index.x = (uint16_t) (pixel % device.width);
    task.index.y = (uint16_t) (pixel / device.width);

    task = camera_get_ray(task, pixel);

    device.ptrs.light_records[pixel]  = get_color(1.0f, 1.0f, 1.0f);
    device.ptrs.bounce_records[pixel] = get_color(1.0f, 1.0f, 1.0f);
    device.ptrs.frame_buffer[pixel]   = get_color(0.0f, 0.0f, 0.0f);
    device.ptrs.mis_buffer[pixel]     = 1.0f;
    device.ptrs.state_buffer[pixel]   = 0;

    if ((device.denoiser || device.aov_mode) && !device.temporal_frames) {
      device.ptrs.albedo_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
      device.ptrs.normal_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }

    if (device.aov_mode) {
      device.ptrs.frame_direct_buffer[pixel]   = get_color(0.0f, 0.0f, 0.0f);
      device.ptrs.frame_indirect_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }

    store_trace_task(device.ptrs.bounce_trace + get_task_address(offset++), task);
  }

  device.ptrs.light_trace_count[THREAD_ID]  = 0;
  device.ptrs.bounce_trace_count[THREAD_ID] = offset;
}
