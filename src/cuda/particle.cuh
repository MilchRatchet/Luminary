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

    const float2 coords = get_coordinates_in_triangle(q.vertex, q.edge1, q.edge2, task.position);

    const vec3 normal = (dot_product(task.ray, q.normal) < 0.0f) ? q.normal : scale_vector(q.normal, -1.0f);

    RGBAF albedo;
    albedo.r = 0.0f;
    albedo.g = 1.0f;
    albedo.b = 0.0f;
    albedo.a = 1.0f;

    float roughness = 1.0f;
    float metallic  = 0.0f;

    uint32_t flags = 0;

    const QuasiRandomTarget random_target =
      (device.iteration_type == TYPE_LIGHT) ? QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY : QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY;

    if (albedo.a < 1.0f && quasirandom_sequence_1D(random_target, pixel) > albedo.a) {
      flags |= G_BUFFER_TRANSPARENT_PASS;
    }

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      flags |= G_BUFFER_REQUIRES_SAMPLING;
    }

    if (flags & G_BUFFER_TRANSPARENT_PASS) {
      task.position = add_vector(task.position, scale_vector(task.ray, eps * get_length(task.position)));
    }
    else {
      task.position = add_vector(task.position, scale_vector(task.ray, -eps * get_length(task.position)));
    }

    GBufferData data;
    data.hit_id    = task.hit_id;
    data.albedo    = albedo;
    data.emission  = get_color(0.0f, 0.0f, 0.0f);
    data.normal    = normal;
    data.position  = task.position;
    data.V         = scale_vector(task.ray, -1.0f);
    data.roughness = roughness;
    data.metallic  = metallic;
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

    if (data.flags & G_BUFFER_TRANSPARENT_PASS) {
      if (device.scene.material.colored_transparency) {
        record = mul_color(record, opaque_color(data.albedo));
      }

      TraceTask new_task;
      new_task.origin = data.position;
      new_task.ray    = task.ray;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBF(device.ptrs.bounce_records + pixel, record);
          store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          device.ptrs.light_transparency_weight_buffer[pixel] *= 2.0f;
          if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY_ROULETTE, pixel) > 0.5f) {
            if (state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
              store_RGBF(device.ptrs.light_records + pixel, record);
              store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
            }
          }
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      if (!material_is_mirror(data.roughness, data.metallic))
        write_albedo_buffer(opaque_color(data.albedo), pixel);

      BRDFInstance brdf = brdf_get_instance(data.albedo, data.V, data.normal, data.roughness, data.metallic);

      bool bounce_is_specular;
      BRDFInstance bounce_brdf = brdf_sample_ray(brdf, pixel, bounce_is_specular);

      float bounce_mis_weight = 1.0f;

      if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
        uint32_t light_history_buffer_entry = LIGHT_ID_ANY;
        LightSample light                   = load_light_sample(device.ptrs.light_samples, pixel);

        if (light.weight > 0.0f) {
          const BRDFInstance brdf_sample = brdf_apply_sample_weight(brdf_apply_sample(brdf, light, data.position, pixel));

          const RGBF light_record = mul_color(record, brdf_sample.term);

          TraceTask light_task;
          light_task.origin = data.position;
          light_task.ray    = brdf_sample.L;
          light_task.index  = task.index;

          if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
            const float light_mis_weight = (bounce_is_specular) ? data.roughness * data.roughness : 1.0f;
            bounce_mis_weight            = 1.0f - light_mis_weight;

            store_RGBF(device.ptrs.light_records + pixel, scale_color(light_record, light_mis_weight));
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }

        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
      }

      RGBF bounce_record = mul_color(record, bounce_brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = data.position;
      bounce_task.ray    = bounce_brdf.L;
      bounce_task.index  = task.index;

      if (validate_trace_task(bounce_task, pixel, bounce_record)) {
        device.ptrs.mis_buffer[pixel] = bounce_mis_weight;
        store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
        store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
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

      store_RGBF(device.ptrs.frame_buffer + pixel, add_color(opaque_color(data.albedo), data.emission));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(value, value, value));
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = load_g_buffer_data(pixel);

      const vec3 normal = data.normal;

      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device.shading_mode == SHADING_HEAT) {
      const float cost  = device.ptrs.trace_result_buffer[pixel].depth;
      const float value = 0.1f * cost;
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(red, green, blue));
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

      store_RGBF(device.ptrs.frame_buffer + pixel, color);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      const GBufferData data = load_g_buffer_data(pixel);

      store_RGBF(device.ptrs.frame_buffer + pixel, add_color(opaque_color(data.albedo), data.emission));
    }
  }
}

__global__ void particle_kernel_generate(const uint32_t count, float4* vertex_buffer, uint32_t* index_buffer, Quad* quads) {
  uint32_t id   = THREAD_ID;
  uint32_t seed = device.scene.particles.seed;

  const float scale = 0.01f;

  while (id < count) {
    const float x = (white_noise_offset(seed + id * 3 + 0) * 20.0f) - 10.0f;
    const float y = (white_noise_offset(seed + id * 3 + 1) * 20.0f) - 10.0f;
    const float z = (white_noise_offset(seed + id * 3 + 2) * 20.0f) - 10.0f;

    const vec3 p = get_vector(x, y, z);

    const vec3 a00 = add_vector(p, get_vector(scale, scale, 0.0f));
    const vec3 a01 = add_vector(p, get_vector(scale, -scale, 0.0f));
    const vec3 a10 = add_vector(p, get_vector(-scale, scale, 0.0f));
    const vec3 a11 = add_vector(p, get_vector(-scale, -scale, 0.0f));

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
    index_buffer[id * 6 + 3] = id * 4 + 1;
    index_buffer[id * 6 + 4] = id * 4 + 3;
    index_buffer[id * 6 + 5] = id * 4 + 2;

    id += blockDim.x * gridDim.x;
  }
}

void device_particle_generate(RaytraceInstance* instance) {
  bench_tic((const char*) "Particles Generation");

  const uint32_t count = 4096;

  ParticlesInstance particles = instance->particles_instance;

  particles.triangle_count = 2 * count;
  particles.vertex_count   = 4 * count;
  particles.index_count    = 6 * count;

  device_buffer_init(&particles.vertex_buffer);
  device_buffer_init(&particles.index_buffer);

  device_buffer_malloc(particles.vertex_buffer, 4 * sizeof(float4), count);
  device_buffer_malloc(particles.index_buffer, 6 * sizeof(uint32_t), count);

  Quad* quads;
  device_malloc(&quads, sizeof(Quad) * count);
  device_update_symbol(particle_quads, quads);

  particle_kernel_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    count, (float4*) device_buffer_get_pointer(particles.vertex_buffer), (uint32_t*) device_buffer_get_pointer(particles.index_buffer),
    quads);
  gpuErrchk(cudaDeviceSynchronize());

  instance->particles_instance = particles;

  bench_toc();
}
