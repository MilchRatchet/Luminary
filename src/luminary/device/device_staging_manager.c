#include "device_staging_manager.h"

#include "device.h"
#include "device_memory.h"
#include "internal_error.h"

#define STAGING_BUFFER_SIZE (1024u * 1024u * 1024u)  // 1 GB
#define STAGING_ENTRIES_COUNT 0x10000                // 65k entries
#define STAGING_ENTRIES_OFFSET_MASK (STAGING_ENTRIES_COUNT - 1)

struct StagingEntry {
  size_t buffer_offset;
  size_t size;
  size_t used_memory;
  DEVICE void* dst;
  size_t dst_offset;
} typedef StagingEntry;

LuminaryResult device_staging_manager_create(DeviceStagingManager** staging_manager, Device* device) {
  __CHECK_NULL_ARGUMENT(staging_manager);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(staging_manager, sizeof(DeviceStagingManager)));

  (*staging_manager)->device = device;

  __FAILURE_HANDLE(device_malloc_staging(&(*staging_manager)->buffer, STAGING_BUFFER_SIZE, true));
  (*staging_manager)->buffer_write_offset = 0;
  (*staging_manager)->buffer_size_in_use  = 0;

  __FAILURE_HANDLE(host_malloc(&(*staging_manager)->entries, sizeof(StagingEntry) * STAGING_ENTRIES_COUNT));
  (*staging_manager)->entries_read_offset  = 0;
  (*staging_manager)->entries_write_offset = 0;
  (*staging_manager)->entries_count_in_use = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_staging_manager_register_direct_access(
  DeviceStagingManager* staging_manager, DEVICE void* dst, size_t dst_offset, size_t size, void** buffer) {
  __CHECK_NULL_ARGUMENT(staging_manager);
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(buffer);

  if (size > STAGING_BUFFER_SIZE) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Staging with direct access does not support entries larger than the staging buffer size.");
  }

  size_t buffer_offset = staging_manager->buffer_write_offset;
  size_t used_memory   = size;

  if (buffer_offset + size > STAGING_BUFFER_SIZE) {
    buffer_offset = 0;
    used_memory += STAGING_BUFFER_SIZE - buffer_offset;
  }

  bool emergency_staging_required = false;

  emergency_staging_required |= (staging_manager->buffer_size_in_use + used_memory > STAGING_BUFFER_SIZE);
  emergency_staging_required |= (staging_manager->entries_count_in_use == STAGING_ENTRIES_COUNT);

  if (emergency_staging_required) {
    log_message("Staging buffer ran out of memory, performing emergency staging.");
    __FAILURE_HANDLE(device_staging_manager_execute(staging_manager));
    CUDA_FAILURE_HANDLE(cuStreamSynchronize(staging_manager->device->stream_main));
  }

  StagingEntry entry;

  entry.buffer_offset = buffer_offset;
  entry.dst           = dst;
  entry.dst_offset    = dst_offset;
  entry.size          = size;
  entry.used_memory   = used_memory;

  *buffer = (void*) (((uint8_t*) staging_manager->buffer) + buffer_offset);

  staging_manager->entries[staging_manager->entries_write_offset] = entry;

  staging_manager->buffer_write_offset = buffer_offset + size;
  staging_manager->buffer_size_in_use += used_memory;
  staging_manager->entries_write_offset = (staging_manager->entries_write_offset + 1) & STAGING_ENTRIES_OFFSET_MASK;
  staging_manager->entries_count_in_use++;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_staging_manager_register(
  DeviceStagingManager* staging_manager, void const* src, DEVICE void* dst, size_t dst_offset, size_t size) {
  __CHECK_NULL_ARGUMENT(staging_manager);
  __CHECK_NULL_ARGUMENT(src);
  __CHECK_NULL_ARGUMENT(dst);

  void* direct_access_buffer;

  // Very large data needs to be staged in chunks.
  while (size > STAGING_BUFFER_SIZE) {
    __FAILURE_HANDLE(
      device_staging_manager_register_direct_access(staging_manager, dst, dst_offset, STAGING_BUFFER_SIZE, &direct_access_buffer));

    memcpy(direct_access_buffer, src, size);

    src = (void const*) (((uint8_t const*) src) + STAGING_BUFFER_SIZE);
    dst_offset += STAGING_BUFFER_SIZE;
    size -= STAGING_BUFFER_SIZE;
  }

  __FAILURE_HANDLE(device_staging_manager_register_direct_access(staging_manager, dst, dst_offset, size, &direct_access_buffer));
  memcpy(direct_access_buffer, src, size);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_staging_manager_execute(DeviceStagingManager* staging_manager) {
  __CHECK_NULL_ARGUMENT(staging_manager);

  while (staging_manager->entries_read_offset != staging_manager->entries_write_offset) {
    const StagingEntry entry = staging_manager->entries[staging_manager->entries_read_offset];

    if (entry.used_memory > staging_manager->buffer_size_in_use) {
      __RETURN_ERROR(
        LUMINARY_ERROR_MEMORY_LEAK, "Staging entry claims to use %llu bytes but only %llu bytes are in use in total.", entry.used_memory,
        staging_manager->buffer_size_in_use);
    }

    __FAILURE_HANDLE(device_upload(
      entry.dst, ((uint8_t*) staging_manager->buffer) + entry.buffer_offset, entry.dst_offset, entry.size,
      staging_manager->device->stream_main));

    staging_manager->buffer_size_in_use -= entry.used_memory;
    staging_manager->entries_read_offset = (staging_manager->entries_read_offset + 1) & STAGING_ENTRIES_OFFSET_MASK;
    staging_manager->entries_count_in_use--;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_staging_manager_destroy(DeviceStagingManager** staging_manager) {
  __CHECK_NULL_ARGUMENT(staging_manager);
  __CHECK_NULL_ARGUMENT(*staging_manager);

  __FAILURE_HANDLE(device_free_staging(&(*staging_manager)->buffer));
  __FAILURE_HANDLE(host_free(&(*staging_manager)->entries));

  __FAILURE_HANDLE(host_free(staging_manager));

  return LUMINARY_SUCCESS;
}
