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

    const MaterialContext ctx = geometry_generate_g_buffer(task, triangle_handle, pixel, false);

    GBufferData data = ctx.data;

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    BSDFSampleInfo bounce_info;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info);

    ////////////////////////////////////////////////////////////////////
    // Update delta path state
    ////////////////////////////////////////////////////////////////////

    bool is_delta_distribution;
    if (bounce_info.is_transparent_pass) {
      const float refraction_scale = (data.ior_in > data.ior_out) ? data.ior_in / data.ior_out : data.ior_out / data.ior_in;
      is_delta_distribution        = data.roughness * fminf(refraction_scale - 1.0f, 1.0f) <= GEOMETRY_DELTA_PATH_CUTOFF;
    }
    else {
      is_delta_distribution = bounce_info.is_microfacet_based && (data.roughness <= GEOMETRY_DELTA_PATH_CUTOFF);
    }

    const bool is_pass_through = bsdf_is_pass_through_ray(bounce_info.is_transparent_pass, data.ior_in, data.ior_out);

    ////////////////////////////////////////////////////////////////////
    // Emission and record
    ////////////////////////////////////////////////////////////////////

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    if (color_any(data.emission)) {
      write_beauty_buffer(mul_color(data.emission, record), pixel, task.state);
    }

    record = mul_color(record, bounce_info.weight);

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method =
        (data.flags & G_BUFFER_FLAG_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_out, pixel, ior_stack_method);
    }

    data.position = shift_origin_vector(data.position, data.V, bounce_ray, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state & ~STATE_FLAG_ALLOW_EMISSION;

    if (!is_delta_distribution) {
      new_state &= ~STATE_FLAG_DELTA_PATH;
    }

    if (!is_pass_through) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    }

    DeviceTask bounce_task;
    bounce_task.state  = new_state;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
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
        const GBufferData data = geometry_generate_g_buffer(task, triangle_handle, pixel, false).data;

        write_beauty_buffer_forced(add_color(opaque_color(data.albedo), data.emission), pixel);
      } break;
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / depth) * 2.0f)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        const GBufferData data = geometry_generate_g_buffer(task, triangle_handle, pixel, false).data;

        const vec3 normal = data.normal;

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
        const GBufferData data = geometry_generate_g_buffer(task, triangle_handle, pixel, false).data;
        const RGBF color       = add_color(scale_color(opaque_color(data.albedo), 0.025f), data.emission);

        write_beauty_buffer_forced(color, pixel);
      } break;
      default:
        break;
    }
  }
}

#endif /* CU_GEOMETRY_H */
