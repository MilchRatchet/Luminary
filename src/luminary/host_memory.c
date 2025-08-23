#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

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
#define HOST_MEMORY_HEADER_FREED_MAGIC (1337ull)

// TODO: Do I need to mark this as atomic????
static _Atomic uint64_t _host_memory_total_allocation;

#ifdef LUMINARY_MEMORY_DEBUG
static mtx_t _memory_debug_mutex;

struct MemoryDebugAllocation {
  const char* name;
  const void* ptr;
  size_t size;
} typedef MemoryDebugAllocation;

static MemoryDebugAllocation* _debug_memory_allocations;
static uint64_t _debug_memory_allocations_count;
static uint64_t _debug_memory_allocations_allocated;

static MemoryDebugAllocation* _debug_memory_allocation_find(const void* ptr) {
  for (uint64_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++) {
    if (_debug_memory_allocations[allocation].ptr == ptr && _debug_memory_allocations[allocation].size != SIZE_MAX)
      return _debug_memory_allocations + allocation;
  }

  return (MemoryDebugAllocation*) 0;
}

static const char* _debug_memory_allocation_get_name(const char* buf_name, const char* func, uint32_t line) {
  char* allocation_name          = malloc(4096);
  const int allocation_name_size = sprintf(allocation_name, "[%s:%u]: %s", func, line, buf_name);
  allocation_name                = realloc(allocation_name, allocation_name_size + 1);

  return allocation_name;
}

static LuminaryResult _debug_memory_allocation_add(
  const void* ptr, const char* buf_name, const char* func, uint32_t line, const size_t size) {
  mtx_lock(&_memory_debug_mutex);

  const char* allocation_name = _debug_memory_allocation_get_name(buf_name, func, line);

  const MemoryDebugAllocation* duplicate_allocation = _debug_memory_allocation_find(ptr);

  if (duplicate_allocation) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "LUMINARY_MEMORY_DEBUG Allocation %s already exists as %s.", allocation_name, duplicate_allocation->name);
  }

  MemoryDebugAllocation allocation;
  allocation.name = allocation_name;
  allocation.size = size;
  allocation.ptr  = ptr;

  if (_debug_memory_allocations_count == _debug_memory_allocations_allocated) {
    _debug_memory_allocations_allocated *= 2;
    _debug_memory_allocations = realloc(_debug_memory_allocations, sizeof(MemoryDebugAllocation) * _debug_memory_allocations_allocated);
  }

  _debug_memory_allocations[_debug_memory_allocations_count++] = allocation;

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _debug_memory_allocation_resize(
  const void* old_ptr, const void* new_ptr, const char* buf_name, const char* func, uint32_t line, const size_t size) {
  mtx_lock(&_memory_debug_mutex);

  const char* allocation_name = _debug_memory_allocation_get_name(buf_name, func, line);

  MemoryDebugAllocation* allocation = _debug_memory_allocation_find(old_ptr);

  if (!allocation) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "LUMINARY_MEMORY_DEBUG Allocation %s does not exist.", allocation_name);
  }

  allocation->size = size;
  allocation->ptr  = new_ptr;

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _debug_memory_allocation_remove(
  const void* ptr, const char* buf_name, const char* func, uint32_t line, const size_t size) {
  mtx_lock(&_memory_debug_mutex);

  const char* allocation_name = _debug_memory_allocation_get_name(buf_name, func, line);

  MemoryDebugAllocation* allocation = _debug_memory_allocation_find(ptr);

  if (!allocation) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "LUMINARY_MEMORY_DEBUG Allocation %s does not exist.", allocation_name);
  }

  if (allocation->size != size) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "LUMINARY_MEMORY_DEBUG Allocation %s has a corrupted size %llu != %llu.", allocation_name, size,
      allocation->size);
  }

  free((char*) allocation->name);
  free((char*) allocation_name);
  allocation->size = SIZE_MAX;

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}

static void _debug_memory_allocation_check_for_leaks() {
  for (uint64_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++) {
    if (_debug_memory_allocations[allocation].size != SIZE_MAX) {
      luminary_print_error("LUMINARY_MEMORY_DEBUG Allocation %s was not freed.", _debug_memory_allocations[allocation].name);
    }
  }
}
#endif /* LUMINARY_MEMORY_DEBUG */

void _host_memory_init(void) {
  atomic_store(&_host_memory_total_allocation, 0);

#ifdef LUMINARY_MEMORY_DEBUG
  mtx_init(&_memory_debug_mutex, mtx_plain);

  _debug_memory_allocations_allocated = 4096;
  _debug_memory_allocations_count     = 0;
  _debug_memory_allocations           = malloc(sizeof(MemoryDebugAllocation) * _debug_memory_allocations_allocated);
#endif /* LUMINARY_MEMORY_DEBUG */
}

void _host_memory_shutdown(void) {
#ifdef LUMINARY_MEMORY_DEBUG
  _debug_memory_allocation_check_for_leaks();
  free(_debug_memory_allocations);

  mtx_destroy(&_memory_debug_mutex);
#endif /* LUMINARY_MEMORY_DEBUG */

  uint64_t leaked_memory = atomic_load(&_host_memory_total_allocation);

  if (leaked_memory > 0) {
    luminary_print_error("Luminary leaked %llu bytes.", leaked_memory);
  }
}

LuminaryResult _host_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(buf_name);
  __CHECK_NULL_ARGUMENT(func);

  struct HostMemoryHeader* header = (struct HostMemoryHeader*) malloc((uint64_t) size + sizeof(struct HostMemoryHeader));

  memset(header, 0, sizeof(struct HostMemoryHeader));

  header->magic = HOST_MEMORY_HEADER_MAGIC;
  header->size  = size;

  const uint64_t prev_total = atomic_fetch_add(&_host_memory_total_allocation, header->size);
  LUM_UNUSED(prev_total);

#ifdef LUMINARY_MEMORY_DEBUG
  _debug_memory_allocation_add((const void*) (header + 1), buf_name, func, line, size);
  luminary_print_log("Malloc  %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total + size, func, line, buf_name);
#endif /* LUMINARY_MEMORY_DEBUG */

  *ptr = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_realloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);
  __CHECK_NULL_ARGUMENT(buf_name);
  __CHECK_NULL_ARGUMENT(func);

  struct HostMemoryHeader* header = ((struct HostMemoryHeader*) (*ptr)) - 1;

  if (header->magic != HOST_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Memory was not allocated through a call to host_memory.");
  }

  if (header->size > atomic_load(&_host_memory_total_allocation)) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Memory allocation is larger than total allocated memory.");
  }

  atomic_fetch_sub(&_host_memory_total_allocation, header->size);

  header = realloc(header, (uint64_t) size + sizeof(struct HostMemoryHeader));

  header->size = size;

  const uint64_t prev_total = atomic_fetch_add(&_host_memory_total_allocation, header->size);
  LUM_UNUSED(prev_total);

#ifdef LUMINARY_MEMORY_DEBUG
  _debug_memory_allocation_resize(*ptr, (const void*) (header + 1), buf_name, func, line, size);
  luminary_print_log("Realloc %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total + size, func, line, buf_name);
#endif /* LUMINARY_MEMORY_DEBUG */

  *ptr = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_free(void** ptr, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);
  __CHECK_NULL_ARGUMENT(buf_name);
  __CHECK_NULL_ARGUMENT(func);

  struct HostMemoryHeader* header = ((struct HostMemoryHeader*) (*ptr)) - 1;

  if (header->magic != HOST_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Memory was not allocated through a call to host_memory.");
  }

  if (header->size > atomic_load(&_host_memory_total_allocation)) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Memory allocation is larger than total allocated memory.");
  }

  header->magic = HOST_MEMORY_HEADER_FREED_MAGIC;

  const uint64_t size = header->size;
  LUM_UNUSED(size);

  const uint64_t prev_total = atomic_fetch_sub(&_host_memory_total_allocation, header->size);
  LUM_UNUSED(prev_total);

#ifdef LUMINARY_MEMORY_DEBUG
  _debug_memory_allocation_remove(*ptr, buf_name, func, line, size);
  luminary_print_log("Free    %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total - size, func, line, buf_name);
#endif /* LUMINARY_MEMORY_DEBUG */

  free(header);

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}
