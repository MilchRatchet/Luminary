#ifndef LUMINARY_DEVICE_PARTICLE_H
#define LUMINARY_DEVICE_PARTICLE_H

#include "device_utils.h"

struct Device typedef Device;
struct OptixBVH typedef OptixBVH;

struct DeviceParticlesHandlePtrs {
  CUdeviceptr quads;
  OptixTraversableHandle bvh;
} typedef DeviceParticlesHandlePtrs;

struct DeviceParticlesHandle {
  uint32_t allocated_count;
  DEVICE float* vertex_buffer;
  DEVICE Quad* quad_buffer;
  DEVICE OptixInstance* optix_instances;
  uint32_t num_instances;
  OptixBVH* geometry_bvh;
  OptixBVH* instance_bvh;
} typedef DeviceParticlesHandle;

DEVICE_CTX_FUNC LuminaryResult device_particles_handle_create(DeviceParticlesHandle** particles_handle);
DEVICE_CTX_FUNC LuminaryResult device_particles_handle_update(
  DeviceParticlesHandle* particles_handle, Device* device, const Particles* shared_particles, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_particles_handle_get_ptrs(DeviceParticlesHandle* particles_handle, DeviceParticlesHandlePtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_particles_handle_destroy(DeviceParticlesHandle** particles_handle);

#endif /* LUMINARY_DEVICE_PARTICLE_H */
