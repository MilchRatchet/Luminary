#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"

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
