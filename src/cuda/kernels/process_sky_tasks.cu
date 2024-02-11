#include "math.cuh"
#include "memory.cuh"
#include "sky.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_sky_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY];

  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    const RGBF record    = load_RGBF(device.records + pixel);
    const uint32_t light = device.ptrs.light_sample_history[pixel];

    RGBF sky;

    if (device.scene.sky.hdri_active) {
      const float mip_bias = (device.iteration_type == TYPE_CAMERA) ? 0.0f : 1.0f;

      sky = mul_color(sky_hdri_sample(task.ray, mip_bias), record);
    }
    else {
      const vec3 origin = world_to_sky_transform(task.origin);

      const bool sample_sun = proper_light_sample(light, LIGHT_ID_SUN);

      RGBF sky_color;
      if (device.iteration_type == TYPE_LIGHT && sample_sun) {
        sky_color = sky_get_sun_color(origin, task.ray);
      }
      else if (device.iteration_type != TYPE_LIGHT) {
        sky_color = sky_get_color(origin, task.ray, FLT_MAX, true, device.scene.sky.steps, seed);
        if (sky_ray_hits_sun(origin, task.ray)) {
          sky_color = scale_color(sky_color, device.ptrs.mis_buffer[pixel]);
        }
      }
      else {
        continue;
      }

      sky = mul_color(sky_color, record);
    }

    write_beauty_buffer(sky, pixel);
    write_albedo_buffer(sky, pixel);
    write_normal_buffer(get_vector(0.0f, 0.0f, 0.0f), pixel);
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}
