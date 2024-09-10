#include "bench.h"
#include "buffer.h"
#include "math.cuh"
#include "particle_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

LUMINARY_KERNEL void particle_process_debug_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];

  for (int i = 0; i < task_count; i++) {
    ShadingTask task     = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const uint32_t pixel = get_pixel_id(task.index);

    if (VOLUME_HIT_CHECK(task.hit_id))
      continue;

    if (device.shading_mode == SHADING_ALBEDO) {
      write_beauty_buffer(device.scene.particles.albedo, pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = particle_generate_g_buffer(task, pixel);

      const vec3 normal = data.normal;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0x55555555, task.hit_id);

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
      write_beauty_buffer(device.scene.particles.albedo, pixel, true);
    }
  }
}

LUMINARY_KERNEL void particle_kernel_generate(
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
