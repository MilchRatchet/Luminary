#include "cloud.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 5) void clouds_render_tasks() {
  const int task_count = device.trace_count[THREAD_ID];

  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset         = get_task_address(i);
    TraceTask task           = load_trace_task(device.trace_tasks + offset);
    const float depth        = __ldcs((float*) (device.ptrs.trace_results + offset));
    vec3 sky_origin          = world_to_sky_transform(task.origin);
    const float sky_max_dist = (depth == device.scene.camera.far_clip_distance) ? FLT_MAX : world_to_sky_scale(depth);
    const int pixel          = task.index.y * device.width + task.index.x;

    RGBF record = load_RGBF(device.records + pixel);
    RGBF color  = get_color(0.0f, 0.0f, 0.0f);

    const float cloud_offset = clouds_render(sky_origin, task.ray, sky_max_dist, pixel, color, record, seed);

    if (device.iteration_type != TYPE_LIGHT && device.scene.sky.cloud.atmosphere_scattering) {
      if (cloud_offset != FLT_MAX && cloud_offset > 0.0f) {
        const float cloud_world_offset = sky_to_world_scale(cloud_offset);

        task.origin = add_vector(task.origin, scale_vector(task.ray, cloud_world_offset));
        store_trace_task(device.trace_tasks + offset, task);

        if (depth != device.scene.camera.far_clip_distance) {
          __stcs((float*) (device.ptrs.trace_results + offset), depth - cloud_world_offset);
        }
      }
    }

    store_RGBF(device.records + pixel, record);
    write_beauty_buffer(color, pixel);
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}
