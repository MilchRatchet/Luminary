#include "device_particle.h"

#include "device.h"
#include "internal_error.h"

DEVICE_CTX_FUNC LuminaryResult device_particle_create(ParticleHandle** particle, Particles particles, Device* device) {
  __CHECK_NULL_ARGUMENT(particle);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(particle, sizeof(ParticleHandle)));

  const uint32_t count = particles.count;

  __FAILURE_HANDLE(device_malloc(&(*particle)->vertex_buffer, 4 * 4 * sizeof(float) * count));
  __FAILURE_HANDLE(device_malloc(&(*particle)->index_buffer, 6 * sizeof(uint16_t) * count));
  __FAILURE_HANDLE(device_malloc(&(*particle)->quad_buffer, sizeof(Quad) * count));

  // TODO: Add kernel arguments

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[CUDA_KERNEL_TYPE_PARTICLE_GENERATE], device->stream_main));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_particle_destroy(ParticleHandle** particle, Device* device) {
  __CHECK_NULL_ARGUMENT(particle);
  __CHECK_NULL_ARGUMENT(*particle);

  __FAILURE_HANDLE(device_free(&(*particle)->vertex_buffer));
  __FAILURE_HANDLE(device_free(&(*particle)->index_buffer));
  __FAILURE_HANDLE(device_free(&(*particle)->quad_buffer));

  __FAILURE_HANDLE(host_free(particle));

  return LUMINARY_SUCCESS;
}
