#ifndef LUMINARY_DEVICE_STAGING_MANAGER_H
#define LUMINARY_DEVICE_STAGING_MANAGER_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;
struct StagingEntry;

struct DeviceStagingManager {
  Device* device;
  STAGING void* buffer;
  size_t buffer_write_offset;
  size_t buffer_size_in_use;
  struct StagingEntry* entries;
  uint32_t entries_read_offset;
  uint32_t entries_write_offset;
  uint32_t entries_count_in_use;
} typedef DeviceStagingManager;

DEVICE_CTX_FUNC LuminaryResult device_staging_manager_create(DeviceStagingManager** staging_manager, Device* device);
DEVICE_CTX_FUNC LuminaryResult
  device_staging_manager_register(DeviceStagingManager* staging_manager, void const* src, DEVICE void* dst, size_t dst_offset, size_t size);

/*
 * Registers memory for staging and returns a ptr which points to a memory section of the registered size that will be staged.
 */
DEVICE_CTX_FUNC LuminaryResult device_staging_manager_register_direct_access(
  DeviceStagingManager* staging_manager, DEVICE void* dst, size_t dst_offset, size_t size, void** buffer);
DEVICE_CTX_FUNC LuminaryResult device_staging_manager_execute(DeviceStagingManager* staging_manager);
DEVICE_CTX_FUNC LuminaryResult device_staging_manager_destroy(DeviceStagingManager** staging_manager);

#endif /* LUMINARY_DEVICE_STAGING_MANAGER_H */
