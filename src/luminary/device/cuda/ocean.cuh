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

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    DeviceIORStack ior_stack    = trace.ior_stack;
    MaterialContextGeometry ctx = ocean_get_context(task, ior_stack);

    const BSDFSampleInfo<MATERIAL_GEOMETRY> bounce_info = bsdf_sample<MaterialContextGeometry::RANDOM_GI>(ctx, task.index);

    RGBF record = record_unpack(throughput.record);
    record      = mul_color(record, bounce_info.weight);

    uint16_t volume_id = task.volume_id;

    if (bounce_info.is_transparent_pass) {
      const bool refraction_is_inside       = ctx.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE;
      const IORStackMethod ior_stack_method = (refraction_is_inside) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(ior_stack, device.ocean.refractive_index, ior_stack_method);

      const VolumeType above_water_volume = (device.fog.active) ? VOLUME_TYPE_FOG : VOLUME_TYPE_NONE;

      volume_id = (refraction_is_inside) ? above_water_volume : VOLUME_TYPE_OCEAN;
    }

    const float shift_length = 2.0f * eps * (1.0f + device.ocean.amplitude) * (1.0f + trace.depth);
    ctx.position             = shift_origin_vector(ctx.position, ctx.V, bounce_info.ray, bounce_info.is_transparent_pass, shift_length);

    uint16_t new_state = task.state;

    new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    new_state &= ~STATE_FLAG_MIS_EMISSION;
    new_state &= ~STATE_FLAG_USE_IGNORE_HANDLE;

    DeviceTask bounce_task;
    bounce_task.state     = new_state;
    bounce_task.origin    = ctx.position;
    bounce_task.ray       = bounce_info.ray;
    bounce_task.index     = task.index;
    bounce_task.volume_id = volume_id;

    if (task_russian_roulette(bounce_task, task.state, record)) {
      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      task_store(dst_task_base_address, bounce_task);
      task_trace_ior_stack_store(dst_task_base_address, ior_stack);
      task_throughput_record_store(dst_task_base_address, record_pack(record));
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

    const uint32_t task_base_address = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                  = task_load(task_base_address);
    const DeviceTaskTrace trace      = task_trace_load(task_base_address);

    const uint32_t pixel = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / trace.depth) * 2.0f)), pixel);
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
