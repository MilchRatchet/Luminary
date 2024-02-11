#include "math.cuh"
#include "memory.cuh"
#include "ocean.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_ocean_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 normal = ocean_get_normal(task.position);

    const float ambient_index_of_refraction = ocean_get_ambient_index_of_refraction(task.position);

    float index_in, index_out;
    if (dot_product(task.ray, normal) > 0.0f) {
      normal    = scale_vector(normal, -1.0f);
      index_in  = device.scene.ocean.refractive_index;
      index_out = ambient_index_of_refraction;
    }
    else {
      index_in  = ambient_index_of_refraction;
      index_out = device.scene.ocean.refractive_index;
    }

    write_normal_buffer(normal, pixel);

    const float refraction_index_ratio = index_in / index_out;
    const vec3 refraction_dir          = refract_ray(task.ray, normal, refraction_index_ratio);

    const float reflection_coefficient = ocean_reflection_coefficient(normal, task.ray, refraction_dir, index_in, index_out);
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY, pixel) < reflection_coefficient) {
      task.position = add_vector(task.position, scale_vector(task.ray, -2.0f * eps * (1.0f + get_length(task.position))));
      task.ray      = reflect_vector(task.ray, normal);
    }
    else {
      task.ray      = refraction_dir;
      task.position = add_vector(task.position, scale_vector(task.ray, 2.0f * eps * (1.0f + get_length(task.position))));
    }

    RGBF record = load_RGBF(device.records + pixel);

    TraceTask new_task;
    new_task.origin = task.position;
    new_task.ray    = task.ray;
    new_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 1.0f;
    store_RGBF(device.ptrs.bounce_records + pixel, record);
    store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}
