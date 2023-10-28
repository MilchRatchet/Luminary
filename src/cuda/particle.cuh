#include "bench.h"
#include "buffer.h"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

__device__ vec3 particle_transform_relative(vec3 p) {
  return sub_vector(p, device.scene.camera.pos);
}

__global__ void particle_generate_g_buffer() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];

  for (int i = 0; i < task_count; i++) {
    ParticleTask task = load_particle_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    Quad q   = load_quad(device.particle_quads, task.hit_id & HIT_TYPE_PARTICLE_MASK);
    q.vertex = particle_transform_relative(q.vertex);
    q.edge1  = particle_transform_relative(q.edge1);
    q.edge2  = particle_transform_relative(q.edge2);

    vec3 normal = (dot_product(task.ray, q.normal) < 0.0f) ? q.normal : scale_vector(q.normal, -1.0f);

    RGBAF albedo;
    albedo.r = device.scene.particles.albedo.r;
    albedo.g = device.scene.particles.albedo.g;
    albedo.b = device.scene.particles.albedo.b;
    albedo.a = 1.0f;

    // Particles BSDF is emulated using volume BSDFs
    uint32_t flags = (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) ? G_BUFFER_REQUIRES_SAMPLING : 0;
    flags |= G_BUFFER_VOLUME_HIT;

    GBufferData data;
    data.hit_id    = task.hit_id;
    data.albedo    = albedo;
    data.emission  = get_color(0.0f, 0.0f, 0.0f);
    data.normal    = normal;
    data.position  = task.position;
    data.V         = scale_vector(task.ray, -1.0f);
    data.roughness = 0.0f;
    data.metallic  = 0.0f;
    data.flags     = flags;

    store_g_buffer_data(data, pixel);
  }
}

__global__ void particle_process_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ParticleTask task = load_particle_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    const GBufferData data = load_g_buffer_data(pixel);

    RGBF record = load_RGBF(device.records + pixel);

    write_normal_buffer(data.normal, pixel);

    const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE, pixel);
    const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BOUNCE_DIR, pixel);

    const vec3 bounce_ray = jendersie_eon_phase_sample(task.ray, device.scene.particles.phase_diameter, random_dir, random_choice);

    VolumeType volume_type = VOLUME_TYPE_PARTICLE;

    record = mul_color(record, opaque_color(data.albedo));

    RGBF bounce_record = record;

    TraceTask bounce_task;
    bounce_task.origin = task.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 0.0f;
    if (validate_trace_task(bounce_task, pixel, bounce_record)) {
      store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      if (light.weight > 0.0f) {
        BRDFInstance brdf = brdf_get_instance_scattering(scale_vector(task.ray, -1.0f));
        const BRDFInstance brdf_sample =
          brdf_apply_sample_weight_scattering(brdf_apply_sample(brdf, light, task.position, pixel), volume_type);

        const RGBF light_record = mul_color(record, brdf_sample.term);

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
          store_RGBF(device.ptrs.light_records + pixel, light_record);
          light_history_buffer_entry = light.id;
          store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
        }
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void particle_process_debug_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];

  for (int i = 0; i < task_count; i++) {
    ParticleTask task = load_particle_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      const GBufferData data = load_g_buffer_data(pixel);

      write_beauty_buffer(add_color(opaque_color(data.albedo), data.emission), pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = load_g_buffer_data(pixel);

      const vec3 normal = data.normal;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_HEAT) {
      const float cost  = device.ptrs.trace_result_buffer[pixel].depth;
      const float value = 0.1f * cost;
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      write_beauty_buffer(get_color(red, green, blue), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0, task.hit_id);

      const uint16_t r = v & 0x7ff;
      const uint16_t g = (v >> 10) & 0x7ff;
      const uint16_t b = (v >> 20) & 0x7ff;

      const float cr = ((float) r) / 0x7ff;
      const float cg = ((float) g) / 0x7ff;
      const float cb = ((float) b) / 0x7ff;

      const RGBF color = get_color(cr, cg, cb);

      write_beauty_buffer(color, pixel, true);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      const GBufferData data = load_g_buffer_data(pixel);

      write_beauty_buffer(add_color(opaque_color(data.albedo), data.emission), pixel, true);
    }
  }
}

__global__ void particle_kernel_generate(
  const uint32_t count, float size, const float size_variation, float4* vertex_buffer, uint32_t* index_buffer, Quad* quads) {
  uint32_t id   = THREAD_ID;
  uint32_t seed = device.scene.particles.seed;

  size *= 0.001f;

  while (id < count) {
    const float x = white_noise_offset(seed + id * 6 + 0);
    const float y = white_noise_offset(seed + id * 6 + 1);
    const float z = white_noise_offset(seed + id * 6 + 2);

    const vec3 p = get_vector(x, y, z);

    const float r1 = 2.0f * white_noise_offset(seed + id * 6 + 3) - 1.0f;
    const float r2 = white_noise_offset(seed + id * 6 + 4);

    const vec3 normal      = sample_ray_sphere(r1, r2);
    const Mat3x3 transform = create_basis(normal);

    const float random_size = 2.0f * white_noise_offset(seed + id * 6 + 5) - 1.0f;
    const float s           = size * (1.0f + random_size * size_variation);

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

    quads[id] = quad;

    __stcs(vertex_buffer + 4 * id + 0, f00);
    __stcs(vertex_buffer + 4 * id + 1, f01);
    __stcs(vertex_buffer + 4 * id + 2, f10);
    __stcs(vertex_buffer + 4 * id + 3, f11);

    index_buffer[id * 6 + 0] = id * 4 + 0;
    index_buffer[id * 6 + 1] = id * 4 + 1;
    index_buffer[id * 6 + 2] = id * 4 + 2;
    index_buffer[id * 6 + 3] = id * 4 + 3;
    index_buffer[id * 6 + 4] = id * 4 + 2;
    index_buffer[id * 6 + 5] = id * 4 + 1;

    id += blockDim.x * gridDim.x;
  }
}

void device_particle_generate(RaytraceInstance* instance) {
  bench_tic((const char*) "Particles Generation");

  ParticlesInstance particles = instance->particles_instance;

  if (particles.vertex_buffer)
    device_buffer_destroy(&particles.vertex_buffer);
  if (particles.index_buffer)
    device_buffer_destroy(&particles.index_buffer);
  if (particles.quad_buffer)
    device_buffer_destroy(&particles.quad_buffer);

  const uint32_t count     = instance->scene.particles.count;
  particles.triangle_count = 2 * count;
  particles.vertex_count   = 4 * count;
  particles.index_count    = 6 * count;

  device_buffer_init(&particles.vertex_buffer);
  device_buffer_init(&particles.index_buffer);
  device_buffer_init(&particles.quad_buffer);

  device_buffer_malloc(particles.vertex_buffer, 4 * sizeof(float4), count);
  device_buffer_malloc(particles.index_buffer, 6 * sizeof(uint32_t), count);
  device_buffer_malloc(particles.quad_buffer, sizeof(Quad), count);

  void* quads = device_buffer_get_pointer(particles.quad_buffer);
  device_update_symbol(particle_quads, quads);

  particle_kernel_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    count, instance->scene.particles.size, instance->scene.particles.size_variation,
    (float4*) device_buffer_get_pointer(particles.vertex_buffer), (uint32_t*) device_buffer_get_pointer(particles.index_buffer),
    (Quad*) device_buffer_get_pointer(particles.quad_buffer));
  gpuErrchk(cudaDeviceSynchronize());

  instance->particles_instance = particles;

  bench_toc();
}
