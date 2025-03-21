#include "device_particle.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "optix_bvh.h"

#define PARTICLES_BLOCK_DIM 25

LuminaryResult device_particles_handle_create(DeviceParticlesHandle** particles_handle) {
  __CHECK_NULL_ARGUMENT(particles_handle);

  __FAILURE_HANDLE(host_malloc(particles_handle, sizeof(DeviceParticlesHandle)));
  memset(*particles_handle, 0, sizeof(DeviceParticlesHandle));

  __FAILURE_HANDLE(optix_bvh_create(&(*particles_handle)->geometry_bvh));
  __FAILURE_HANDLE(optix_bvh_create(&(*particles_handle)->instance_bvh));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_particles_instances_create(
  DeviceParticlesHandle* particles_handle, Device* device, OptixTraversableHandle gas_handle) {
  __CHECK_NULL_ARGUMENT(particles_handle);
  __CHECK_NULL_ARGUMENT(device);

  particles_handle->num_instances = PARTICLES_BLOCK_DIM * PARTICLES_BLOCK_DIM * PARTICLES_BLOCK_DIM;

  if (particles_handle->optix_instances == (OptixInstance*) 0) {
    __FAILURE_HANDLE(device_malloc(&particles_handle->optix_instances, sizeof(OptixInstance) * particles_handle->num_instances));
  }

  OptixInstance* instances;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, particles_handle->optix_instances, 0, sizeof(OptixInstance) * particles_handle->num_instances,
    (void**) &instances));

  uint32_t instance_id = 0;

  for (int32_t xi = 0; xi < PARTICLES_BLOCK_DIM; xi++) {
    const int32_t x = xi - (PARTICLES_BLOCK_DIM >> 1);
    for (int32_t yi = 0; yi < PARTICLES_BLOCK_DIM; yi++) {
      const int32_t y = yi - (PARTICLES_BLOCK_DIM >> 1);
      for (int32_t zi = 0; zi < PARTICLES_BLOCK_DIM; zi++) {
        const int32_t z = zi - (PARTICLES_BLOCK_DIM >> 1);

        OptixInstance instance;
        memset(&instance, 0, sizeof(OptixInstance));

        instance.instanceId        = instance_id;
        instance.sbtOffset         = 0;
        instance.visibilityMask    = ((1u << (device->optix_properties.num_bits_instance_visibility_mask)) - 1);
        instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        instance.traversableHandle = gas_handle;

        instance.transform[0]  = 1.0f;
        instance.transform[1]  = 0.0f;
        instance.transform[2]  = 0.0f;
        instance.transform[3]  = x;
        instance.transform[4]  = 0.0f;
        instance.transform[5]  = 1.0f;
        instance.transform[6]  = 0.0f;
        instance.transform[7]  = y;
        instance.transform[8]  = 0.0f;
        instance.transform[9]  = 0.0f;
        instance.transform[10] = 1.0f;
        instance.transform[11] = z;

        instances[instance_id++] = instance;
      }
    }
  }

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

    if (particles_handle->quad_buffer) {
      __FAILURE_HANDLE(device_free(&particles_handle->quad_buffer));
    }

    __FAILURE_HANDLE(device_malloc(&particles_handle->vertex_buffer, 6 * 4 * sizeof(float) * particles->count));
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
  args.quads          = DEVICE_PTR(particles_handle->quad_buffer);

  __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_PARTICLE_GENERATE], &args, device->stream_main));

  __FAILURE_HANDLE(optix_bvh_particles_gas_build(particles_handle->geometry_bvh, device, particles_handle));

  __FAILURE_HANDLE(
    _device_particles_instances_create(particles_handle, device, particles_handle->geometry_bvh->traversable[OPTIX_BVH_TYPE_DEFAULT]));

  __FAILURE_HANDLE(optix_bvh_particles_ias_build(particles_handle->instance_bvh, device, particles_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_particles_handle_destroy(DeviceParticlesHandle** particles_handle) {
  __CHECK_NULL_ARGUMENT(particles_handle);
  __CHECK_NULL_ARGUMENT(*particles_handle);

  __FAILURE_HANDLE(optix_bvh_destroy(&(*particles_handle)->geometry_bvh));
  __FAILURE_HANDLE(optix_bvh_destroy(&(*particles_handle)->instance_bvh));

  if ((*particles_handle)->vertex_buffer) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->vertex_buffer));
  }

  if ((*particles_handle)->quad_buffer) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->quad_buffer));
  }

  if ((*particles_handle)->optix_instances) {
    __FAILURE_HANDLE(device_free(&(*particles_handle)->optix_instances));
  }

  __FAILURE_HANDLE(host_free(particles_handle));

  return LUMINARY_SUCCESS;
}
