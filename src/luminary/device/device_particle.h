#ifndef LUMINARY_DEVICE_PARTICLE_H
#define LUMINARY_DEVICE_PARTICLE_H

#include "device_utils.h"

struct Device typedef Device;
struct OptixBVH typedef OptixBVH;

struct DeviceParticlesHandle {
  uint32_t allocated_count;
  uint32_t count;
  DEVICE float* vertex_buffer;
  DEVICE uint16_t* index_buffer;
  DEVICE Quad* quad_buffer;
  OptixBVH* bvh;
} typedef DeviceParticlesHandle;

DEVICE_CTX_FUNC LuminaryResult device_particles_handle_create(DeviceParticlesHandle** particles_handle);
DEVICE_CTX_FUNC LuminaryResult
  device_particles_handle_generate(DeviceParticlesHandle* particles_handle, const Particles* particles, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_particles_handle_destroy(DeviceParticlesHandle** particles_handle);

#endif /* LUMINARY_DEVICE_PARTICLE_H */
