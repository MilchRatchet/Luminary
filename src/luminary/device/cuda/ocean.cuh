#ifndef CU_LUMINARY_OCEAN_H
#define CU_LUMINARY_OCEAN_H

#include "bsdf.cuh"
#include "directives.cuh"
#include "math.cuh"
#include "medium_stack.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "utils.cuh"

LUMINARY_KERNEL void ocean_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_OCEAN];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);
    DeviceTaskMediumStack medium          = task_medium_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    const MaterialContextGeometry ctx = ocean_get_context(task, medium);

    const uint32_t task_direct_lighting_base_address =
      task_get_base_address<DeviceTaskDirectLight>(task_offset + i, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting BSDF
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_bsdf_is_allowed(task, trace)) {
      const DeviceTaskDirectLightBSDF direct_light_bsdf_task = direct_lighting_bsdf_create_task(ctx, task.path_id, 0.0f);

      task_direct_light_bsdf_store(task_direct_lighting_base_address, direct_light_bsdf_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Sun
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_sun_is_allowed(task, trace)) {
      const DeviceTaskDirectLightSun direct_light_sun_task = direct_lighting_sun_create_task(ctx, medium, task.path_id);

      task_direct_light_sun_store(task_direct_lighting_base_address, direct_light_sun_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    const BSDFSampleInfo<MATERIAL_GEOMETRY> bounce_info = bsdf_sample<MaterialContextGeometry::RANDOM_GI>(ctx, task.path_id);

    RGBF record = record_unpack(throughput.record);
    record      = mul_color(record, bounce_info.weight);

    const vec3 bounce_pos = ocean_shift_vector(ctx, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state;

    new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    new_state &= ~STATE_FLAG_ALLOW_EMISSION;
    new_state &= ~STATE_FLAG_USE_IGNORE_HANDLE;

    DeviceTask bounce_task;
    bounce_task.state   = new_state;
    bounce_task.origin  = bounce_pos;
    bounce_task.ray     = bounce_info.ray;
    bounce_task.path_id = task.path_id;

    if (task_russian_roulette(bounce_task, task.state, record)) {
      // Apply medium transition.
      if (bounce_info.is_transparent_pass) {
        const bool refraction_is_inside = ctx.params.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE;

        medium_stack_ior_modify(medium, device.ocean.refractive_index, refraction_is_inside == false);
        medium_stack_volume_modify(medium, VOLUME_TYPE_OCEAN, refraction_is_inside == false);
      }

      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      task_store(dst_task_base_address, bounce_task);
      task_throughput_record_store(dst_task_base_address, record_pack(record));
      task_medium_store(dst_task_base_address, medium);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void ocean_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_OCEAN];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    RGBF result;
    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_DEPTH: {
        result = splat_color(__saturatef((1.0f / trace.depth) * 2.0f));
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        vec3 normal = ocean_get_normal(task.origin);

        normal.x = 0.5f * normal.x + 0.5f;
        normal.y = 0.5f * normal.y + 0.5f;
        normal.z = 0.5f * normal.z + 0.5f;

        result = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
      } break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION: {
        result = get_color(0.0f, 0.0f, 1.0f);
      } break;
      default:
        result = splat_color(0.0f);
        break;
    }

    write_beauty_buffer(result, throughput.results_index);
  }
}

#endif /* CU_LUMINARY_OCEAN_H */
