
#include "math.cuh"
#include "particle_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

LUMINARY_KERNEL void particle_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_PARTICLE];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);
    const TaskFlags flags                 = throughput.flags;
    const DeviceTaskMediumStack medium    = task_medium_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    const MaterialContextParticle ctx = particle_get_context(task, flags, medium, trace.handle.instance_id);
    const bool split_replay           = path_id_is_split_replay(task.path_id);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Geometry
    ////////////////////////////////////////////////////////////////////

    const uint32_t task_direct_lighting_base_address =
      task_get_base_address<DeviceTaskDirectLight>(task_offset + i, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

    if (split_replay == false && direct_lighting_geometry_is_allowed(flags, trace)) {
      float light_tree_root_sum;
      const DeviceTaskDirectLightGeo direct_light_geo_task = direct_lighting_geometry_create_task(ctx, task.path_id, light_tree_root_sum);

      task_direct_light_geo_store(task_direct_lighting_base_address, direct_light_geo_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Sun
    ////////////////////////////////////////////////////////////////////

    if (split_replay == false && direct_lighting_sun_is_allowed(task, trace)) {
      const DeviceTaskDirectLightSun direct_light_sun_task = direct_lighting_sun_create_task(ctx, medium, task.path_id);

      task_direct_light_sun_store(task_direct_lighting_base_address, direct_light_sun_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    BSDFSampleInfo<MATERIAL_PARTICLE> bounce_info = bsdf_sample<MaterialContextParticle::RANDOM_GI>(ctx, task.path_id);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Ambient
    ////////////////////////////////////////////////////////////////////

    if (split_replay == false && direct_lighting_ambient_is_allowed(task, trace)) {
      const DeviceTaskDirectLightAmbient direct_light_ambient_task = direct_lighting_ambient_create_task(ctx, bounce_info, task.path_id);

      task_direct_light_ambient_store(task_direct_lighting_base_address, direct_light_ambient_task);
    }

    uint32_t new_state = flags;

    new_state &= ~STATE_FLAG_DELTA_PATH;

    if (device.particles.type == LUMINARY_PARTICLE_TYPE_RAIN) {
      new_state |= STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_AMBIENT;
    }
    else {
      new_state &= ~(STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_AMBIENT);
    }

    if (path_id_get_depth(task.path_id) < device.state.max_depth && russian_roulette_apply(task.path_id, flags, bounce_info.weight)) {
      RGBF record = record_unpack(throughput.record);
      record      = mul_color(record, bounce_info.weight);

      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      DeviceTask bounce_task;
      bounce_task.origin  = ctx.position;
      bounce_task.ray     = bounce_info.ray;
      bounce_task.path_id = task.path_id;
      path_id_increment_depth(bounce_task.path_id);

      task_store(dst_task_base_address, bounce_task);

      DeviceTaskThroughput bounce_throughput;
      bounce_throughput.record        = record_pack(scale_color(record, path_id_get_bounce_scale(task.path_id)));
      bounce_throughput.results_index = throughput.results_index;
      task_result_set_live(throughput.results_index, 1);
      bounce_throughput.flags = new_state;

      task_throughput_store(dst_task_base_address, bounce_throughput);

      task_medium_store(dst_task_base_address, medium);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

LUMINARY_KERNEL void particle_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const uint16_t task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE];
  const uint16_t task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_PARTICLE];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);
    const DeviceTaskMediumStack medium    = task_medium_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    RGBF result;
    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_ALBEDO:
        result = splat_color(1.0f);
        break;
      case LUMINARY_SHADING_MODE_DEPTH:
        result = splat_color(__saturatef((1.0f / trace.depth) * 2.0f));
        break;
      case LUMINARY_SHADING_MODE_NORMAL: {
        const MaterialContextParticle data = particle_get_context(task, throughput.flags, medium, trace.handle.instance_id);

        const vec3 normal = data.normal;

        result = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
      } break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION: {
        const uint32_t v = random_uint32_t_base(0x55555555, trace.handle.instance_id);

        const uint16_t r = v & 0x7ff;
        const uint16_t g = (v >> 10) & 0x7ff;
        const uint16_t b = (v >> 20) & 0x7ff;

        const float cr = ((float) r) / 0x7ff;
        const float cg = ((float) g) / 0x7ff;
        const float cb = ((float) b) / 0x7ff;

        result = get_color(cr, cg, cb);
      } break;
      default:
        result = splat_color(0.0f);
        break;
    }

    write_beauty_buffer(result, throughput.results_index);
  }
}

LUMINARY_KERNEL void particle_generate(const KernelArgsParticleGenerate args) {
  uint32_t id = THREAD_ID;

  while (id < args.count) {
    const float x = white_noise_offset(args.seed + id * 4 + 0);
    const float y = white_noise_offset(args.seed + id * 4 + 1);
    const float z = white_noise_offset(args.seed + id * 4 + 2);

    const float random_size = 2.0f * white_noise_offset(args.seed + id * 4 + 3) - 1.0f;
    const float radius      = fmaxf(args.radius + random_size * args.radius_variation, 0.0f);

    __stcs(args.vertex_buffer + id, make_float4(x, y, z, radius));

    id += blockDim.x * gridDim.x;
  }
}
