#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "directives.cuh"
#include "geometry_utils.cuh"
#include "math.cuh"
#include "memory.cuh"

LUMINARY_KERNEL void geometry_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                  = task_load(task_base_address);
    const DeviceTaskTrace trace      = task_trace_load(task_base_address);
    DeviceTaskMediumStack medium     = task_medium_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    const MaterialContextGeometry ctx = geometry_get_context(task, trace, medium);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Geometry
    ////////////////////////////////////////////////////////////////////

    const uint32_t task_direct_lighting_base_address =
      task_get_base_address<DeviceTaskDirectLight>(task_offset + i, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

    float light_tree_root_sum = 0.0f;
    if (direct_lighting_geometry_is_allowed(task, trace)) {
      const DeviceTaskDirectLightGeo direct_light_geo_task = direct_lighting_geometry_create_task(ctx, task.path_id, light_tree_root_sum);

      task_direct_light_geo_store(task_direct_lighting_base_address, direct_light_geo_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting BSDF
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_bsdf_is_allowed(task, trace)) {
      const DeviceTaskDirectLightBSDF direct_light_bsdf_task = direct_lighting_bsdf_create_task(ctx, task.path_id, light_tree_root_sum);

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

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Ambient
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_ambient_is_allowed(task, trace)) {
      const DeviceTaskDirectLightAmbient direct_light_ambient_task = direct_lighting_ambient_create_task(ctx, bounce_info, task.path_id);

      task_direct_light_ambient_store(task_direct_lighting_base_address, direct_light_ambient_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Update delta path state
    ////////////////////////////////////////////////////////////////////

    const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params);

    bool is_delta_distribution;
    if (bounce_info.is_transparent_pass) {
      const float ior = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params);

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

    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    RGBF record = record_unpack(throughput.record);

    const RGBF emission = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx.params);

    if (color_any(emission)) {
      const uint32_t index = path_id_get_pixel_index(task.path_id);
      write_beauty_buffer(mul_color(emission, record), index, task.state);
    }

    record = mul_color(record, bounce_info.weight);

    uint16_t new_state = task.state | STATE_FLAG_USE_IGNORE_HANDLE;

    if (device.sky.mode != LUMINARY_SKY_MODE_DEFAULT && is_pass_through == false)
      new_state &= ~STATE_FLAG_ALLOW_AMBIENT;
    else
      new_state |= STATE_FLAG_ALLOW_AMBIENT;

    if (is_delta_distribution == false) {
      new_state &= ~STATE_FLAG_DELTA_PATH;
    }

    if (is_pass_through == false) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
      new_state &= ~STATE_FLAG_ALLOW_EMISSION;
    }

    DeviceTask bounce_task;
    bounce_task.state   = new_state;
    bounce_task.origin  = ctx.position;
    bounce_task.ray     = bounce_info.ray;
    bounce_task.path_id = task.path_id;

    if (task_russian_roulette(bounce_task, task.state, record)) {
      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      task_store(dst_task_base_address, bounce_task);

      // We need to store a trace result so the raytracing kernels knows which triangle to ignore.
      DeviceTaskTrace bounce_trace;
      bounce_trace.handle = triangle_handle_get(ctx.instance_id, ctx.tri_id);
      bounce_trace.depth  = FLT_MAX;

      task_trace_store(dst_task_base_address, bounce_trace);

      DeviceTaskThroughput bounce_throughput;
      bounce_throughput.record = record_pack(record);

      task_throughput_store(dst_task_base_address, bounce_throughput);

      // Apply medium transition.
      if (bounce_info.is_transparent_pass) {
        const bool refraction_is_inside = ctx.params.flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE;

        float new_ior = 1.0f;
        if (refraction_is_inside == false) {
          // The material only stores the IOR ratio, we mirror what we do during material context construction and compute the materials
          // actual IOR.
          const float ray_ior   = medium_stack_ior_peek(medium, refraction_is_inside);
          const float ior_ratio = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params);
          new_ior               = ray_ior / ior_ratio;
        }

        medium_stack_ior_modify(medium, new_ior, refraction_is_inside == false);
      }

      task_medium_store(dst_task_base_address, medium);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void geometry_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address   = task_get_base_address(i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                    = task_load(task_base_address);
    const DeviceTaskTrace trace        = task_trace_load(task_base_address);
    const DeviceTaskMediumStack medium = task_medium_load(task_base_address);
    const uint32_t index               = path_id_get_pixel_index(task.path_id);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_ALBEDO: {
        const MaterialContextGeometry ctx = geometry_get_context(task, trace, medium);
        const RGBF albedo                 = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx.params);
        const RGBF emission               = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx.params);

        write_beauty_buffer_forced(add_color(albedo, emission), index);
      } break;
      case LUMINARY_SHADING_MODE_DEPTH: {
        write_beauty_buffer_forced(splat_color(__saturatef((1.0f / trace.depth) * 2.0f)), index);
      } break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        const MaterialContextGeometry ctx = geometry_get_context(task, trace, medium);

        const vec3 normal = ctx.normal;

        write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), index);
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

        write_beauty_buffer_forced(color, index);
      } break;
      case LUMINARY_SHADING_MODE_LIGHTS: {
        const MaterialContextGeometry ctx = geometry_get_context(task, trace, medium);
        const RGBF albedo                 = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx.params);
        const RGBF emission               = material_get_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx.params);
        const RGBF color                  = add_color(scale_color(albedo, 0.025f), emission);

        write_beauty_buffer_forced(color, index);
      } break;
      default:
        break;
    }
  }
}

#endif /* CU_GEOMETRY_H */
