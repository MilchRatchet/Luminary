#include "device_particle.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "optix_bvh.h"

LuminaryResult device_particles_handle_create(DeviceParticlesHandle** particles_handle) {
  __CHECK_NULL_ARGUMENT(particles_handle);

  __FAILURE_HANDLE(host_malloc(particles_handle, sizeof(DeviceParticlesHandle)));
  memset(*particles_handle, 0, sizeof(DeviceParticlesHandle));

  __FAILURE_HANDLE(optix_bvh_create(&(*particles_handle)->bvh));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_particles_handle_generate(DeviceParticlesHandle* particles_handle, const Particles* particles, Device* device) {
  __CHECK_NULL_ARGUMENT(particles_handle);
  __CHECK_NULL_ARGUMENT(particles);
  __CHECK_NULL_ARGUMENT(device);

  if (particles->count == 0 || particles->active == false)
    return LUMINARY_SUCCESS;

  if (particles->count > particles_handle->allocated_count) {
    if (particles_handle->vertex_buffer) {
      __FAILURE_HANDLE(device_free(&particles_handle->vertex_buffer));
    }

    if (particles_handle->index_buffer) {
      __FAILURE_HANDLE(device_free(&particles_handle->index_buffer));
    }

    if (particles_handle->quad_buffer) {
      __FAILURE_HANDLE(device_free(&particles_handle->quad_buffer));
    }

    __FAILURE_HANDLE(device_malloc(&particles_handle->vertex_buffer, 4 * 4 * sizeof(float) * particles->count));
    __FAILURE_HANDLE(device_malloc(&particles_handle->index_buffer, 6 * sizeof(uint16_t) * particles->count));
    __FAILURE_HANDLE(device_malloc(&particles_handle->quad_buffer, sizeof(Quad) * particles->count));

    particles_handle->allocated_count = particles->count;
  }

  particles_handle->count = particles->count;

  KernelArgsParticleGenerate args;

  args.count          = particles->count;
  args.seed           = particles->seed;
  args.size           = particles->size * 0.001f;
  args.size_variation = particles->size_variation;
  args.vertex_buffer  = DEVICE_PTR(particles_handle->vertex_buffer);
  args.index_buffer   = DEVICE_PTR(particles_handle->index_buffer);
  args.quads          = DEVICE_PTR(particles_handle->quad_buffer);

  __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_PARTICLE_GENERATE], &args, device->stream_main));

  __FAILURE_HANDLE(optix_bvh_particles_build(particles_handle->bvh, device, particles_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_particles_handle_destroy(DeviceParticlesHandle** particles_handle) {
  __CHECK_NULL_ARGUMENT(particles_handle);
  __CHECK_NULL_ARGUMENT(*particles_handle);

  __FAILURE_HANDLE(optix_bvh_destroy(&(*particles_handle)->bvh));

  if ((*particles_handle)->vertex_buffer) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->vertex_buffer));
  }

  if ((*particles_handle)->index_buffer) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->index_buffer));
  }

  if ((*particles_handle)->quad_buffer) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->quad_buffer));
  }

  __FAILURE_HANDLE(host_free(particles_handle));

  return LUMINARY_SUCCESS;
}
