
#include "math.cuh"
#include "particle_utils.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

LUMINARY_KERNEL void particle_process_tasks_debug() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset       = get_task_address(task_offset + i);
    const DeviceTask task       = task_load(offset);
    const TriangleHandle handle = triangle_handle_load(offset);
    const uint32_t pixel        = get_pixel_id(task.index);

    if (VOLUME_HIT_CHECK(handle.instance_id))
      continue;

    if (device.settings.shading_mode == LUMINARY_SHADING_MODE_ALBEDO) {
      write_beauty_buffer_forced(device.particles.albedo, pixel);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_DEPTH) {
      const float dist  = get_length(sub_vector(device.camera.pos, task.origin));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer_forced(get_color(value, value, value), pixel);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_NORMAL) {
      const GBufferData data = particle_generate_g_buffer(task, handle.instance_id, pixel);

      const vec3 normal = data.normal;

      write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0x55555555, handle.instance_id);

      const uint16_t r = v & 0x7ff;
      const uint16_t g = (v >> 10) & 0x7ff;
      const uint16_t b = (v >> 20) & 0x7ff;

      const float cr = ((float) r) / 0x7ff;
      const float cg = ((float) g) / 0x7ff;
      const float cb = ((float) b) / 0x7ff;

      const RGBF color = get_color(cr, cg, cb);

      write_beauty_buffer_forced(color, pixel);
    }
    else if (device.settings.shading_mode == LUMINARY_SHADING_MODE_LIGHTS) {
      write_beauty_buffer_forced(device.particles.albedo, pixel);
    }
  }
}

LUMINARY_KERNEL void particle_kernel_generate(
  const uint32_t count, const uint32_t seed, const float size, const float size_variation, float4* vertex_buffer, uint32_t* index_buffer,
  Quad* quads) {
  uint32_t id = THREAD_ID;

  // TODO: Make sure to do this multiplication when passing this arg to the kernel.
  // size *= 0.001f;

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
