#include "device_constant_memory_manager.h"

#include "device.h"
#include "internal_error.h"

#define DEVICE_CONSTANT_MEMORY_MANAGER_STAGING_BUFFER_ID_MASK (DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS - 1)

// #define DEVICE_CONSTANT_MEMORY_VALIDATE_BUFFER_COUNT

static const DeviceConstantMemoryMember _device_scene_entity_to_const_memory_member[SCENE_ENTITY_GLOBAL_COUNT] = {
  DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS,   // SCENE_ENTITY_SETTINGS
  DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA,     // SCENE_ENTITY_CAMERA
  DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN,      // SCENE_ENTITY_OCEAN
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY,        // SCENE_ENTITY_SKY
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD,      // SCENE_ENTITY_CLOUD
  DEVICE_CONSTANT_MEMORY_MEMBER_FOG,        // SCENE_ENTITY_FOG
  DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES,  // SCENE_ENTITY_PARTICLES
};

static const size_t _device_cuda_const_memory_offsets[DEVICE_CONSTANT_MEMORY_MEMBER_COUNT + 1] = {
  offsetof(DeviceConstantMemory, ptrs),                          // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  offsetof(DeviceConstantMemory, settings),                      // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  offsetof(DeviceConstantMemory, camera),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  offsetof(DeviceConstantMemory, ocean),                         // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  offsetof(DeviceConstantMemory, sky),                           // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  offsetof(DeviceConstantMemory, cloud),                         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  offsetof(DeviceConstantMemory, fog),                           // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  offsetof(DeviceConstantMemory, particles),                     // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  offsetof(DeviceConstantMemory, optix_bvh),                     // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  offsetof(DeviceConstantMemory, moon_albedo_tex),               // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  offsetof(DeviceConstantMemory, sky_lut_transmission_low_tex),  // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX
  offsetof(DeviceConstantMemory, sky_hdri_color_tex),            // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX
  offsetof(DeviceConstantMemory, bsdf_lut_conductor),            // DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX
  offsetof(DeviceConstantMemory, cloud_noise_shape_tex),         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX
  offsetof(DeviceConstantMemory, spectral_xy_lut_tex),           // DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX
  offsetof(DeviceConstantMemory, config),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG
  offsetof(DeviceConstantMemory, state),                         // DEVICE_CONSTANT_MEMORY_MEMBER_STATE
  sizeof(DeviceConstantMemory)                                   // DEVICE_CONSTANT_MEMORY_MEMBER_COUNT
};

static const size_t _device_cuda_const_memory_sizes[DEVICE_CONSTANT_MEMORY_MEMBER_COUNT] = {
  sizeof(DevicePointers),                // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  sizeof(DeviceRendererSettings),        // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  sizeof(DeviceCamera),                  // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  sizeof(DeviceOcean),                   // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  sizeof(DeviceSky),                     // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  sizeof(DeviceCloud),                   // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  sizeof(DeviceFog),                     // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  sizeof(DeviceParticles),               // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  sizeof(OptixTraversableHandle) * 4,    // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  sizeof(DeviceTextureObject) * 4,       // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX
  sizeof(DeviceTextureObject) * 4,       // DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX
  sizeof(DeviceTextureObject) * 3,       // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX
  sizeof(DeviceExecutionConfiguration),  // DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG
  sizeof(DeviceExecutionState)           // DEVICE_CONSTANT_MEMORY_MEMBER_STATE
};

LuminaryResult device_constant_memory_manager_create(DeviceConstantMemoryManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(DeviceConstantMemoryManager)));
  memset(*manager, 0, sizeof(DeviceConstantMemoryManager));

  for (uint32_t staging_buffer_id = 0; staging_buffer_id < DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS; staging_buffer_id++) {
    __FAILURE_HANDLE(device_malloc_staging(
      &(*manager)->staging_buffers[staging_buffer_id], sizeof(DeviceConstantMemory), DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY));
    memset((*manager)->staging_buffers[staging_buffer_id], 0, sizeof(DeviceConstantMemory));

    CUDA_FAILURE_HANDLE(cuEventCreate(&(*manager)->sync_finished_event[staging_buffer_id], CU_EVENT_DISABLE_TIMING));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_constant_memory_manager_get_host_buffer(
  DeviceConstantMemoryManager* manager, const DeviceConstantMemory** host_buffer) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(host_buffer);

  *host_buffer = &manager->data;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_constant_memory_manager_set_data(
  DeviceConstantMemoryManager* manager, size_t offset, size_t size, const void* value) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(value);

  memcpy(((uint8_t*) &manager->data) + offset, value, size);
  memcpy(((STAGING uint8_t*) manager->staging_buffers[manager->current_staging_buffer_id]) + offset, value, size);

  const size_t max_address = offset + size;

  if (manager->is_dirty) {
    manager->min_address_dirty = (offset < manager->min_address_dirty) ? offset : manager->min_address_dirty;
    manager->max_address_dirty = (max_address > manager->max_address_dirty) ? max_address : manager->max_address_dirty;
  }
  else {
    manager->min_address_dirty = offset;
    manager->max_address_dirty = max_address;
    manager->is_dirty          = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_constant_memory_manager_set_scene_entity(
  DeviceConstantMemoryManager* manager, SceneEntity entity, const void* value) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(value);

  const DeviceConstantMemoryMember member = _device_scene_entity_to_const_memory_member[entity];
  const size_t member_offset              = _device_cuda_const_memory_offsets[member];
  const size_t member_size                = _device_cuda_const_memory_sizes[member];

  __FAILURE_HANDLE(device_constant_memory_manager_set_data(manager, member_offset, member_size, value));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_constant_memory_manager_ensure_synced(DeviceConstantMemoryManager* manager, Device* device, CUstream stream) {
  __CHECK_NULL_ARGUMENT(manager);

  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(stream, manager->sync_finished_event[manager->current_staging_buffer_id], CU_EVENT_WAIT_DEFAULT));

  if (manager->is_dirty) {
    const size_t offset = manager->min_address_dirty;
    const size_t size   = manager->max_address_dirty - offset;

    const void* src = ((uint8_t*) manager->staging_buffers[manager->current_staging_buffer_id]) + offset;

#ifdef DEVICE_CONSTANT_MEMORY_VALIDATE_BUFFER_COUNT
    if (cuEventQuery(manager->sync_finished_event[manager->current_staging_buffer_id]) != CUDA_SUCCESS)
      warn_message("Constant memory manager stalled due to lack of staging buffers.");
#endif /* DEVICE_CONSTANT_MEMORY_VALIDATE_BUFFER_COUNT*/

    CUDA_FAILURE_HANDLE(cuEventSynchronize(manager->sync_finished_event[manager->current_staging_buffer_id]));
    CUDA_FAILURE_HANDLE(cuMemcpyHtoDAsync(device->cuda_device_const_memory + offset, src, size, stream));

    uint32_t next_staging_buffer_id = (manager->current_staging_buffer_id + 1) & DEVICE_CONSTANT_MEMORY_MANAGER_STAGING_BUFFER_ID_MASK;

    // Update the next staging buffer using the host data (which is always in the most recent state)
    memcpy(manager->staging_buffers[next_staging_buffer_id], &manager->data, sizeof(DeviceConstantMemory));

    CUDA_FAILURE_HANDLE(cuEventRecord(manager->sync_finished_event[next_staging_buffer_id], stream));

    manager->current_staging_buffer_id = next_staging_buffer_id;
    manager->is_dirty                  = false;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_constant_memory_manager_destroy(DeviceConstantMemoryManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  for (uint32_t staging_buffer_id = 0; staging_buffer_id < DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS; staging_buffer_id++) {
    __FAILURE_HANDLE(device_free_staging(&(*manager)->staging_buffers[staging_buffer_id]));
    CUDA_FAILURE_HANDLE(cuEventDestroy((*manager)->sync_finished_event[staging_buffer_id]));
  }

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}
