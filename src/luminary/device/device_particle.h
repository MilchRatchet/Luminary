#ifndef LUMINARY_DEVICE_PARTICLE_H
#define LUMINARY_DEVICE_PARTICLE_H

#include "device_utils.h"

struct Device typedef Device;

struct ParticleHandle {
  DEVICE float* vertex_buffer;
  DEVICE uint16_t* index_buffer;
  DEVICE Quad* quad_buffer;
} typedef ParticleHandle;

DEVICE_CTX_FUNC LuminaryResult device_particle_create(ParticleHandle** particle, Particles particles, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_particle_destroy(ParticleHandle** particle, Device* device);

#endif /* LUMINARY_DEVICE_PARTICLE_H */
