
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
    const DeviceTaskMediumStack medium    = task_medium_load(task_base_address);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    const MaterialContextParticle ctx = particle_get_context(task, medium, trace.handle.instance_id);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Geometry
    ////////////////////////////////////////////////////////////////////

    const uint32_t task_direct_lighting_base_address =
      task_get_base_address<DeviceTaskDirectLight>(task_offset + i, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

    if (direct_lighting_geometry_is_allowed(task)) {
      float light_tree_root_sum;
      const DeviceTaskDirectLightGeo direct_light_geo_task = direct_lighting_geometry_create_task(ctx, task.path_id, light_tree_root_sum);

      task_direct_light_geo_store(task_direct_lighting_base_address, direct_light_geo_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Sun
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_sun_is_allowed(task)) {
      const DeviceTaskDirectLightSun direct_light_sun_task = direct_lighting_sun_create_task(ctx, medium, task.path_id);

      task_direct_light_sun_store(task_direct_lighting_base_address, direct_light_sun_task);
    }

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    const BSDFSampleInfo<MATERIAL_PARTICLE> bounce_info = bsdf_sample<MaterialContextParticle::RANDOM_GI>(ctx, task.path_id);

    ////////////////////////////////////////////////////////////////////
    // Direct Lighting Ambient
    ////////////////////////////////////////////////////////////////////

    if (direct_lighting_ambient_is_allowed(task)) {
      const DeviceTaskDirectLightAmbient direct_light_ambient_task = direct_lighting_ambient_create_task(ctx, bounce_info, task.path_id);

      task_direct_light_ambient_store(task_direct_lighting_base_address, direct_light_ambient_task);
    }

    uint16_t new_state = task.state;

    new_state &= ~STATE_FLAG_DELTA_PATH;
    new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    new_state &= ~STATE_FLAG_ALLOW_EMISSION;
    new_state &= ~STATE_FLAG_USE_IGNORE_HANDLE;

    if (device.sky.mode != LUMINARY_SKY_MODE_DEFAULT)
      new_state &= ~STATE_FLAG_ALLOW_AMBIENT;
    else
      new_state |= STATE_FLAG_ALLOW_AMBIENT;

    DeviceTask bounce_task;
    bounce_task.state   = new_state;
    bounce_task.origin  = ctx.position;
    bounce_task.ray     = bounce_info.ray;
    bounce_task.path_id = task.path_id;

    RGBF record = record_unpack(throughput.record);
    record      = mul_color(record, bounce_info.weight);

    if (task_russian_roulette(bounce_task, task.state, record)) {
      const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

      task_store(dst_task_base_address, bounce_task);
      task_throughput_record_store(dst_task_base_address, record_pack(record));

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

    const uint32_t task_base_address   = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                    = task_load(task_base_address);
    const DeviceTaskTrace trace        = task_trace_load(task_base_address);
    const DeviceTaskMediumStack medium = task_medium_load(task_base_address);
    const uint32_t index               = path_id_get_pixel_index(task.path_id);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    if (device.settings.shading_mode == LUMINARY_SHADING_MODE_ALBEDO) {
      write_beauty_buffer_forced(device.particles.albedo, index);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_DEPTH) {
      write_beauty_buffer_forced(splat_color(__saturatef((1.0f / trace.depth) * 2.0f)), index);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_NORMAL) {
      const MaterialContextParticle data = particle_get_context(task, medium, trace.handle.instance_id);

      const vec3 normal = data.normal;

      write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), index);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0x55555555, trace.handle.instance_id);

      const uint16_t r = v & 0x7ff;
      const uint16_t g = (v >> 10) & 0x7ff;
      const uint16_t b = (v >> 20) & 0x7ff;

      const float cr = ((float) r) / 0x7ff;
      const float cg = ((float) g) / 0x7ff;
      const float cb = ((float) b) / 0x7ff;

      const RGBF color = get_color(cr, cg, cb);

      write_beauty_buffer_forced(color, index);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_LIGHTS) {
      write_beauty_buffer_forced(device.particles.albedo, index);
    }
  }
}

LUMINARY_KERNEL void particle_generate(const KernelArgsParticleGenerate args) {
  uint32_t id = THREAD_ID;

  while (id < args.count) {
    const float x = white_noise_offset(args.seed + id * 6 + 0);
    const float y = white_noise_offset(args.seed + id * 6 + 1);
    const float z = white_noise_offset(args.seed + id * 6 + 2);

    const vec3 p = get_vector(x, y, z);

    const float r1 = 2.0f * white_noise_offset(args.seed + id * 6 + 3) - 1.0f;
    const float r2 = white_noise_offset(args.seed + id * 6 + 4);

    const vec3 normal      = sample_ray_sphere(r1, r2);
    const Mat3x3 transform = create_basis(normal);

    const float random_size = 2.0f * white_noise_offset(args.seed + id * 6 + 5) - 1.0f;
    const float s           = args.size * (1.0f + random_size * args.size_variation);

    const vec3 a00 = add_vector(p, transform_vec3(transform, get_vector(s, s, 0.0f)));
    const vec3 a01 = add_vector(p, transform_vec3(transform, get_vector(s, -s, 0.0f)));
    const vec3 a10 = add_vector(p, transform_vec3(transform, get_vector(-s, s, 0.0f)));
    const vec3 a11 = add_vector(p, transform_vec3(transform, get_vector(-s, -s, 0.0f)));

    const float4 f00 = make_float4(a00.x, a00.y, a00.z, 1.0f);
    const float4 f01 = make_float4(a01.x, a01.y, a01.z, 1.0f);
    const float4 f10 = make_float4(a10.x, a10.y, a10.z, 1.0f);
    const float4 f11 = make_float4(a11.x, a11.y, a11.z, 1.0f);

    Quad quad;
    quad.vertex = a00;
    quad.edge1  = sub_vector(a01, a00);
    quad.edge2  = sub_vector(a10, a00);
    quad.normal = normalize_vector(cross_product(quad.edge1, quad.edge2));

    args.quads[id] = quad;

    __stcs(args.vertex_buffer + 6 * id + 0, f00);
    __stcs(args.vertex_buffer + 6 * id + 1, f01);
    __stcs(args.vertex_buffer + 6 * id + 2, f10);
    __stcs(args.vertex_buffer + 6 * id + 3, f11);
    __stcs(args.vertex_buffer + 6 * id + 4, f01);
    __stcs(args.vertex_buffer + 6 * id + 5, f10);

    id += blockDim.x * gridDim.x;
  }
}
