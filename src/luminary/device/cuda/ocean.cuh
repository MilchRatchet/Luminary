#ifndef CU_LUMINARY_OCEAN_H
#define CU_LUMINARY_OCEAN_H

#include "bsdf.cuh"
#include "directives.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "utils.cuh"

LUMINARY_KERNEL void ocean_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_OCEAN];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t offset = get_task_address(task_offset + i);
    DeviceTask task       = task_load(offset);
    const float depth     = trace_depth_load(offset);
    const uint32_t pixel  = get_pixel_id(task.index);
    RGBF record           = load_RGBF(device.ptrs.records + pixel);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    MaterialContextGeometry ctx = ocean_get_context(task, pixel);

    const BSDFSampleInfo<MATERIAL_GEOMETRY> bounce_info = bsdf_sample(ctx, task.index);

    record = mul_color(record, bounce_info.weight);

    const bool is_pass_through = bsdf_is_pass_through_ray(ctx, bounce_info.is_transparent_pass);

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method =
        (ctx.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(ctx.ior_out, pixel, ior_stack_method);
    }

    ctx.position = shift_origin_vector(ctx.position, ctx.V, bounce_info.ray, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state;

    if (!is_pass_through) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    }

    DeviceTask bounce_task;
    bounce_task.state  = new_state;
    bounce_task.origin = ctx.position;
    bounce_task.ray    = bounce_info.ray;
    bounce_task.index  = task.index;

    if (task_russian_roulette(bounce_task, task.state, record)) {
      task_store(bounce_task, get_task_address(trace_count++));
      store_RGBF(device.ptrs.records, pixel, record);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void ocean_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_OCEAN];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t offset = get_task_address(task_offset + i);
    DeviceTask task       = task_load(offset);
    const float depth     = trace_depth_load(offset);
    const uint32_t pixel  = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / depth) * 2.0f)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        vec3 normal = ocean_get_normal(task.origin);

        normal.x = 0.5f * normal.x + 0.5f;
        normal.y = 0.5f * normal.y + 0.5f;
        normal.z = 0.5f * normal.z + 0.5f;

        write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION: {
        write_beauty_buffer_forced(get_color(0.0f, 0.0f, 1.0f), pixel);
      } break;
      default:
        break;
    }
  }
}

#endif /* CU_LUMINARY_OCEAN_H */
