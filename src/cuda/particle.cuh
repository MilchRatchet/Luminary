#include "bench.h"
#include "buffer.h"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

__global__ void particle_kernel_generate(const uint32_t count, float4* vertex_buffer, uint32_t* index_buffer) {
  uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  const float scale = 1.0f;

  while (id < count) {
    const float x = white_noise_offset(seed++);
    const float y = white_noise_offset(seed++);
    const float z = white_noise_offset(seed++);

    const vec3 p = get_vector(x, y, z);

    const vec3 a00 = add_vector(p, get_vector(scale, scale, 0.0f));
    const vec3 a01 = add_vector(p, get_vector(scale, -scale, 0.0f));
    const vec3 a10 = add_vector(p, get_vector(-scale, scale, 0.0f));
    const vec3 a11 = add_vector(p, get_vector(-scale, -scale, 0.0f));

    const float4 f00 = make_float4(a00.x, a00.y, a00.z, 1.0f);
    const float4 f01 = make_float4(a01.x, a01.y, a01.z, 1.0f);
    const float4 f10 = make_float4(a10.x, a10.y, a10.z, 1.0f);
    const float4 f11 = make_float4(a11.x, a11.y, a11.z, 1.0f);

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

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

void device_particle_generate(RaytraceInstance* instance) {
  bench_tic();

  const uint32_t count = 4096;

  ParticlesInstance particles;
  memset(&particles, 0, sizeof(ParticlesInstance));

  particles.triangle_count = 2 * count;
  particles.vertex_count   = 4 * count;
  particles.index_count    = 6 * count;

  device_buffer_init(&particles.vertex_buffer);
  device_buffer_init(&particles.index_buffer);

  device_buffer_malloc(particles.vertex_buffer, 4 * sizeof(float4), count);
  device_buffer_malloc(particles.index_buffer, 6 * sizeof(uint32_t), count);

  particle_kernel_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    count, (float4*) device_buffer_get_pointer(particles.vertex_buffer), (uint32_t*) device_buffer_get_pointer(particles.index_buffer));
  gpuErrchk(cudaDeviceSynchronize());

  instance->particles_instance = particles;

  bench_toc((char*) "Particles Generation");
}
