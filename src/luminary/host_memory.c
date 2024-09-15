#include <stdint.h>
#include <stdlib.h>

// Include stdatomic later, otherwise I run into issues.
#include <stdatomic.h>

#include "internal_error.h"
#include "utils.h"

struct HostMemoryHeader {
  uint64_t magic;
  uint64_t size;
  uint64_t padding[6];
};
LUM_STATIC_SIZE_ASSERT(struct HostMemoryHeader, 64);

// LUMHOSTM
#define HOST_MEMORY_HEADER_MAGIC (0x4D54534F484D554Cull)

// TODO: Do I need to mark this as atomic????
static _Atomic uint64_t _host_memory_total_allocation;

void _host_memory_init(void) {
  atomic_store(&_host_memory_total_allocation, 0);
}

LuminaryResult _host_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  if (!ptr) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Ptr is NULL.");
  }

  struct HostMemoryHeader* header = (struct HostMemoryHeader*) malloc((uint64_t) size + sizeof(struct HostMemoryHeader));

  header->magic = HOST_MEMORY_HEADER_MAGIC;
  header->size  = size;

  atomic_fetch_add(&_host_memory_total_allocation, header->size);

  *ptr = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_realloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  if (!ptr) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Ptr is NULL.");
  }

  if (!(*ptr)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Ptr is NULL.");
  }

  struct HostMemoryHeader* header = ((struct HostMemoryHeader*) (*ptr)) - 1;

  if (header->magic != HOST_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Memory was not allocated through a call to host_memory.");
  }

  if (header->size > atomic_load(&_host_memory_total_allocation)) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Memory allocation is larger than total allocated memory.");
  }

  atomic_fetch_sub(&_host_memory_total_allocation, header->size);

  realloc(header, (uint64_t) size + sizeof(struct HostMemoryHeader));

  header->size = size;

  atomic_fetch_add(&_host_memory_total_allocation, header->size);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_free(void** ptr, const char* buf_name, const char* func, uint32_t line) {
  if (!ptr) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Ptr is NULL.");
  }

  if (!(*ptr)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Ptr is NULL.");
  }

  struct HostMemoryHeader* header = ((struct HostMemoryHeader*) (*ptr)) - 1;

  if (header->magic != HOST_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Memory was not allocated through a call to host_memory.");
  }

  if (header->size > atomic_load(&_host_memory_total_allocation)) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Memory allocation is larger than total allocated memory.");
  }

  atomic_fetch_sub(&_host_memory_total_allocation, header->size);

  free(header);

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}
