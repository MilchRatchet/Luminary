#ifndef LUMINARY_DEVICE_CONSTANT_MEMORY_MANAGER_H
#define LUMINARY_DEVICE_CONSTANT_MEMORY_MANAGER_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

#define DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS (8)

struct DeviceConstantMemoryManager {
  DeviceConstantMemory data;
  uint32_t current_staging_buffer_id;
  STAGING DeviceConstantMemory* staging_buffers[DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS];
  CUevent sync_finished_event[DEVICE_CONSTANT_MEMORY_MANAGER_NUM_STAGING_BUFFERS];
  bool is_dirty;
  size_t min_address_dirty;
  size_t max_address_dirty;
} typedef DeviceConstantMemoryManager;

LuminaryResult device_constant_memory_manager_create(DeviceConstantMemoryManager** manager);
LuminaryResult device_constant_memory_manager_get_host_buffer(
  DeviceConstantMemoryManager* manager, const DeviceConstantMemory** host_buffer);
LuminaryResult device_constant_memory_manager_set_data(DeviceConstantMemoryManager* manager, size_t offset, size_t size, const void* value);
LuminaryResult device_constant_memory_manager_set_scene_entity(DeviceConstantMemoryManager* manager, SceneEntity entity, const void* value);
LuminaryResult device_constant_memory_manager_ensure_synced(DeviceConstantMemoryManager* manager, Device* device, CUstream stream);
LuminaryResult device_constant_memory_manager_destroy(DeviceConstantMemoryManager** manager);

#endif /* LUMINARY_DEVICE_CONSTANT_MEMORY_MANAGER_H */
