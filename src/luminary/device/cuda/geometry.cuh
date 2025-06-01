#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "bsdf.cuh"
#include "directives.cuh"
#include "geometry_utils.cuh"
#include "math.cuh"
#include "memory.cuh"

LUMINARY_KERNEL void geometry_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t offset          = get_task_address(task_offset + i);
    DeviceTask task                = task_load(offset);
    TriangleHandle triangle_handle = triangle_handle_load(offset);
    const float depth              = trace_depth_load(offset);
    const uint32_t pixel           = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    MaterialContextGeometry ctx = geometry_get_context(task, triangle_handle, pixel);

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    const BSDFSampleInfo<MATERIAL_GEOMETRY> bounce_info = bsdf_sample(ctx, task.index);

    ////////////////////////////////////////////////////////////////////
    // Update delta path state
    ////////////////////////////////////////////////////////////////////

    bool is_delta_distribution;
    if (bounce_info.is_transparent_pass) {
      const float refraction_scale = (ctx.ior_in > ctx.ior_out) ? ctx.ior_in / ctx.ior_out : ctx.ior_out / ctx.ior_in;
      is_delta_distribution        = ctx.roughness * fminf(refraction_scale - 1.0f, 1.0f) <= GEOMETRY_DELTA_PATH_CUTOFF;
    }
    else {
      is_delta_distribution = bounce_info.is_microfacet_based && (ctx.roughness <= GEOMETRY_DELTA_PATH_CUTOFF);
    }

    const bool is_pass_through = bsdf_is_pass_through_ray(ctx, bounce_info.is_transparent_pass);

    ////////////////////////////////////////////////////////////////////
    // Emission and record
    ////////////////////////////////////////////////////////////////////

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    if (color_any(ctx.emission)) {
      write_beauty_buffer(mul_color(ctx.emission, record), pixel, task.state);
    }

    record = mul_color(record, bounce_info.weight);

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method =
        (ctx.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(ctx.ior_out, pixel, ior_stack_method);
    }

    ctx.position = shift_origin_vector(ctx.position, ctx.V, bounce_info.ray, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state & ~STATE_FLAG_ALLOW_EMISSION;

    if (!is_delta_distribution) {
      new_state &= ~STATE_FLAG_DELTA_PATH;
    }

    if (!is_pass_through) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;

      // We want to keep the old payload around if we are passthrough.
      new_state |= STATE_FLAG_MIS_EMISSION;
      store_mis_payload(pixel, mis_get_payload(ctx, bounce_info.ray));
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

LUMINARY_KERNEL void geometry_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t offset          = get_task_address(i);
    DeviceTask task                = task_load(offset);
    TriangleHandle triangle_handle = triangle_handle_load(offset);
    const float depth              = trace_depth_load(offset);
    const uint32_t pixel           = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_ALBEDO: {
        const MaterialContextGeometry ctx = geometry_get_context(task, triangle_handle, pixel);

        write_beauty_buffer_forced(add_color(opaque_color(ctx.albedo), ctx.emission), pixel);
      } break;
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / depth) * 2.0f)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        const MaterialContextGeometry ctx = geometry_get_context(task, triangle_handle, pixel);

        const vec3 normal = ctx.normal;

        write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION: {
        const uint32_t v = random_uint32_t_base(0x55555555, (triangle_handle.instance_id << 16) | triangle_handle.tri_id);

        const uint16_t r = v & 0x7ff;
        const uint16_t g = (v >> 10) & 0x7ff;
        const uint16_t b = (v >> 20) & 0x7ff;

        const float cr = ((float) r) / 0x7ff;
        const float cg = ((float) g) / 0x7ff;
        const float cb = ((float) b) / 0x7ff;

        const RGBF color = get_color(cr, cg, cb);

        write_beauty_buffer_forced(color, pixel);
      } break;
      case LUMINARY_SHADING_MODE_LIGHTS: {
        const MaterialContextGeometry ctx = geometry_get_context(task, triangle_handle, pixel);
        const RGBF color                  = add_color(scale_color(opaque_color(ctx.albedo), 0.025f), ctx.emission);

        write_beauty_buffer_forced(color, pixel);
      } break;
      default:
        break;
    }
  }
}

#endif /* CU_GEOMETRY_H */
