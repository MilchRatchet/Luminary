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

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    DeviceIORStack ior_stack     = trace.ior_stack;
    PackedMISPayload mis_payload = throughput.payload;

    GeometryContextCreationInfo ctx_creation_info;
    ctx_creation_info.task               = task;
    ctx_creation_info.handle             = trace.handle;
    ctx_creation_info.ior_stack          = trace.ior_stack;
    ctx_creation_info.packed_mis_payload = throughput.payload;
    ctx_creation_info.hints              = GEOMETRY_CONTEXT_CREATION_HINT_NONE;

    MaterialContextGeometry ctx = geometry_get_context(ctx_creation_info);

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    const BSDFSampleInfo<MATERIAL_GEOMETRY> bounce_info = bsdf_sample(ctx, task.index);

    ////////////////////////////////////////////////////////////////////
    // Update delta path state
    ////////////////////////////////////////////////////////////////////

    const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx);

    bool is_delta_distribution;
    if (bounce_info.is_transparent_pass) {
      const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx);

      const float refraction_scale = (ior >= 1.0f) ? ior : 1.0f / ior;
      is_delta_distribution        = roughness * fminf(refraction_scale - 1.0f, 1.0f) <= GEOMETRY_DELTA_PATH_CUTOFF;
    }
    else {
      is_delta_distribution = bounce_info.is_microfacet_based && (roughness <= GEOMETRY_DELTA_PATH_CUTOFF);
    }

    const bool is_pass_through = bsdf_is_pass_through_ray(ctx, bounce_info);

    ////////////////////////////////////////////////////////////////////
    // Emission and record
    ////////////////////////////////////////////////////////////////////

    RGBF record = record_unpack(throughput.record);

    const RGBF emission = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx);

    if (color_any(emission)) {
      const uint32_t pixel = get_pixel_id(task.index);
      write_beauty_buffer(mul_color(emission, record), pixel, task.state);
    }

    record = mul_color(record, bounce_info.weight);

    if (bounce_info.is_transparent_pass) {
      const bool refraction_is_inside = ctx.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE;

      const IORStackMethod ior_get_method = (refraction_is_inside) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
      const float ray_ior                 = ior_stack_interact(ior_stack, 1.0f, ior_get_method);

      const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx);

      const float new_ior = (refraction_is_inside) ? 1.0f : ray_ior / ior;

      const IORStackMethod ior_set_method = (refraction_is_inside) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(ior_stack, new_ior, ior_set_method);
    }

    ctx.position = shift_origin_vector(ctx.position, ctx.V, bounce_info.ray, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state;

    if (!is_delta_distribution) {
      new_state &= ~STATE_FLAG_DELTA_PATH;
    }

    if (!is_pass_through) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
      new_state &= ~STATE_FLAG_ALLOW_EMISSION;

      // We want to keep the old payload around if we are passthrough.
      new_state |= STATE_FLAG_MIS_EMISSION;
      mis_payload = mis_payload_pack(mis_get_payload(ctx, bounce_info.ray, bounce_info.is_transparent_pass));
    }

    DeviceTask bounce_task;
    bounce_task.state     = new_state;
    bounce_task.origin    = ctx.position;
    bounce_task.ray       = bounce_info.ray;
    bounce_task.index     = task.index;
    bounce_task.volume_id = task.volume_id;

    if (task_russian_roulette(bounce_task, task.state, record)) {
      DeviceTaskThroughput bounce_throughput;
      bounce_throughput.record  = record_pack(record);
      bounce_throughput.payload = mis_payload;

      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      task_store(dst_task_base_address, bounce_task);
      task_trace_ior_stack_store(dst_task_base_address, ior_stack);
      task_throughput_store(dst_task_base_address, bounce_throughput);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void geometry_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);
    const uint32_t pixel                  = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_ALBEDO: {
        GeometryContextCreationInfo ctx_creation_info;
        ctx_creation_info.task               = task;
        ctx_creation_info.handle             = trace.handle;
        ctx_creation_info.ior_stack          = trace.ior_stack;
        ctx_creation_info.packed_mis_payload = throughput.payload;
        ctx_creation_info.hints              = GEOMETRY_CONTEXT_CREATION_HINT_NONE;

        const MaterialContextGeometry ctx = geometry_get_context(ctx_creation_info);
        const RGBF albedo                 = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx);
        const RGBF emission               = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx);

        write_beauty_buffer_forced(add_color(albedo, emission), pixel);
      } break;
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / trace.depth) * 2.0f)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        GeometryContextCreationInfo ctx_creation_info;
        ctx_creation_info.task               = task;
        ctx_creation_info.handle             = trace.handle;
        ctx_creation_info.ior_stack          = trace.ior_stack;
        ctx_creation_info.packed_mis_payload = throughput.payload;
        ctx_creation_info.hints              = GEOMETRY_CONTEXT_CREATION_HINT_NONE;

        const MaterialContextGeometry ctx = geometry_get_context(ctx_creation_info);

        const vec3 normal = ctx.normal;

        write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
      } break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION: {
        const uint32_t v = random_uint32_t_base(0x55555555, (trace.handle.instance_id << 16) | trace.handle.tri_id);

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
        GeometryContextCreationInfo ctx_creation_info;
        ctx_creation_info.task               = task;
        ctx_creation_info.handle             = trace.handle;
        ctx_creation_info.ior_stack          = trace.ior_stack;
        ctx_creation_info.packed_mis_payload = throughput.payload;
        ctx_creation_info.hints              = GEOMETRY_CONTEXT_CREATION_HINT_NONE;

        const MaterialContextGeometry ctx = geometry_get_context(ctx_creation_info);
        const RGBF albedo                 = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx);
        const RGBF emission               = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx);
        const RGBF color                  = add_color(scale_color(albedo, 0.025f), emission);

        write_beauty_buffer_forced(color, pixel);
      } break;
      default:
        break;
    }
  }
}

#endif /* CU_GEOMETRY_H */
