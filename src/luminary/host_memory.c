#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

// Include stdatomic later, otherwise I run into issues.
#include <luminary/debug.h>
#include <luminary/stack_trace.h>
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

#if defined(LUMINARY_MEMORY_DEBUG) || defined(LUMINARY_ENABLE_DEBUG_API)
#define HOST_MEMORY_TRACKING_ENABLED

static mtx_t _memory_debug_mutex;

struct MemoryDebugAllocation {
  char* name;
#ifdef LUMINARY_ENABLE_DEBUG_API
  char* stack_trace;
#endif
  const void* ptr;
  size_t size;
} typedef MemoryDebugAllocation;

static MemoryDebugAllocation* _debug_memory_allocations;
static size_t _debug_memory_allocations_count;
static size_t _debug_memory_allocations_allocated;

static MemoryDebugAllocation* _debug_memory_allocation_find(const void* ptr) {
  for (size_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++) {
    if (_debug_memory_allocations[allocation].ptr == ptr)
      return _debug_memory_allocations + allocation;
  }

  return (MemoryDebugAllocation*) 0;
}

static char* _debug_memory_allocation_get_name(const char* buf_name, const char* func, uint32_t line) {
  const int allocation_name_size = snprintf((char*) 0, 0, "[%s:%u]: %s", func, line, buf_name);

  if (allocation_name_size < 0)
    return (char*) 0;

  char* allocation_name = malloc((size_t) allocation_name_size + 1);

  if (allocation_name == (char*) 0)
    return (char*) 0;

  if (snprintf(allocation_name, (size_t) allocation_name_size + 1, "[%s:%u]: %s", func, line, buf_name) < 0) {
    free(allocation_name);
    return (char*) 0;
  }

  return allocation_name;
}

#ifdef LUMINARY_ENABLE_DEBUG_API
static LuminaryResult _debug_memory_allocation_get_stack_trace(char** stack_trace) {
  __CHECK_NULL_ARGUMENT(stack_trace);

  LuminaryNativeStackTrace native_stack_trace = {0};
  __FAILURE_HANDLE(luminary_get_native_stack_trace(&native_stack_trace));

  *stack_trace              = native_stack_trace.text;
  native_stack_trace.text   = (char*) 0;
  native_stack_trace.length = 0;

  return LUMINARY_SUCCESS;
}
#endif

static LuminaryResult _debug_memory_allocation_add(
  const void* ptr, const char* buf_name, const char* func, uint32_t line, const size_t size, uint64_t* prev_total) {
  char* allocation_name = _debug_memory_allocation_get_name(buf_name, func, line);

  if (allocation_name == (char*) 0)
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to create host allocation tracking name.");

#ifdef LUMINARY_ENABLE_DEBUG_API
  char* allocation_stack_trace      = (char*) 0;
  LuminaryResult stack_trace_result = _debug_memory_allocation_get_stack_trace(&allocation_stack_trace);

  if (stack_trace_result != LUMINARY_SUCCESS) {
    free(allocation_name);
    return stack_trace_result;
  }
#endif

  mtx_lock(&_memory_debug_mutex);

  const MemoryDebugAllocation* duplicate_allocation = _debug_memory_allocation_find(ptr);

  if (duplicate_allocation) {
    mtx_unlock(&_memory_debug_mutex);
    free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
    free(allocation_stack_trace);
#endif
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Host allocation is already present in the tracking list.");
  }

  if (_debug_memory_allocations_count == _debug_memory_allocations_allocated) {
    size_t new_allocated = _debug_memory_allocations_allocated == 0 ? 256 : _debug_memory_allocations_allocated * 2;

    if (new_allocated < _debug_memory_allocations_allocated || new_allocated > SIZE_MAX / sizeof(MemoryDebugAllocation)) {
      mtx_unlock(&_memory_debug_mutex);
      free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
      free(allocation_stack_trace);
#endif
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Host allocation tracking list is too large.");
    }

    MemoryDebugAllocation* new_allocations = realloc(_debug_memory_allocations, sizeof(MemoryDebugAllocation) * new_allocated);

    if (new_allocations == (MemoryDebugAllocation*) 0) {
      mtx_unlock(&_memory_debug_mutex);
      free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
      free(allocation_stack_trace);
#endif
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to grow host allocation tracking list.");
    }

    _debug_memory_allocations           = new_allocations;
    _debug_memory_allocations_allocated = new_allocated;
  }

  MemoryDebugAllocation* allocation = _debug_memory_allocations + _debug_memory_allocations_count++;
  allocation->name                  = allocation_name;
#ifdef LUMINARY_ENABLE_DEBUG_API
  allocation->stack_trace = allocation_stack_trace;
#endif
  allocation->size = size;
  allocation->ptr  = ptr;

  *prev_total = atomic_fetch_add(&_host_memory_total_allocation, size);

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _debug_memory_allocation_remove(
  const void* ptr, const char* buf_name, const char* func, uint32_t line, const size_t size, uint64_t* prev_total) {
  mtx_lock(&_memory_debug_mutex);

  MemoryDebugAllocation* allocation = _debug_memory_allocation_find(ptr);

  if (allocation == (MemoryDebugAllocation*) 0) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Host allocation [%s:%u]: %s is not tracked.", func, line, buf_name);
  }

  if (allocation->size != size) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "Host allocation [%s:%u]: %s has a corrupted size %llu != %llu.", func, line, buf_name, size,
      allocation->size);
  }

  free(allocation->name);
#ifdef LUMINARY_ENABLE_DEBUG_API
  free(allocation->stack_trace);
#endif

  const size_t allocation_index = (size_t) (allocation - _debug_memory_allocations);
  _debug_memory_allocations_count--;

  if (allocation_index < _debug_memory_allocations_count)
    _debug_memory_allocations[allocation_index] = _debug_memory_allocations[_debug_memory_allocations_count];

  *prev_total = atomic_fetch_sub(&_host_memory_total_allocation, size);

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}

#ifdef LUMINARY_MEMORY_DEBUG
static void _debug_memory_allocation_check_for_leaks() {
  for (size_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++) {
    luminary_print_error("LUMINARY_MEMORY_DEBUG Allocation %s was not freed.", _debug_memory_allocations[allocation].name);
  }
}
#endif /* LUMINARY_MEMORY_DEBUG */

static LuminaryResult _debug_memory_allocation_realloc(
  void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line, uint64_t* prev_total) {
  char* allocation_name = _debug_memory_allocation_get_name(buf_name, func, line);

  if (allocation_name == (char*) 0)
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to create host allocation tracking name.");

#ifdef LUMINARY_ENABLE_DEBUG_API
  char* allocation_stack_trace      = (char*) 0;
  LuminaryResult stack_trace_result = _debug_memory_allocation_get_stack_trace(&allocation_stack_trace);

  if (stack_trace_result != LUMINARY_SUCCESS) {
    free(allocation_name);
    return stack_trace_result;
  }
#endif

  mtx_lock(&_memory_debug_mutex);

  struct HostMemoryHeader* header   = ((struct HostMemoryHeader*) (*ptr)) - 1;
  MemoryDebugAllocation* allocation = _debug_memory_allocation_find(*ptr);

  if (allocation == (MemoryDebugAllocation*) 0) {
    mtx_unlock(&_memory_debug_mutex);
    free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
    free(allocation_stack_trace);
#endif
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Host allocation is not tracked.");
  }

  if (allocation->size != header->size) {
    mtx_unlock(&_memory_debug_mutex);
    free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
    free(allocation_stack_trace);
#endif
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Host allocation tracking size does not match its header.");
  }

  const uint64_t old_size             = header->size;
  struct HostMemoryHeader* new_header = realloc(header, size + sizeof(struct HostMemoryHeader));

  if (new_header == (struct HostMemoryHeader*) 0) {
    mtx_unlock(&_memory_debug_mutex);
    free(allocation_name);
#ifdef LUMINARY_ENABLE_DEBUG_API
    free(allocation_stack_trace);
#endif
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to realloc host memory.");
  }

  new_header->size = size;

  atomic_fetch_sub(&_host_memory_total_allocation, old_size);
  *prev_total = atomic_fetch_add(&_host_memory_total_allocation, size);

  free(allocation->name);
#ifdef LUMINARY_ENABLE_DEBUG_API
  free(allocation->stack_trace);
  allocation->stack_trace = allocation_stack_trace;
#endif
  allocation->name = allocation_name;
  allocation->ptr  = (const void*) (new_header + 1);
  allocation->size = size;

  *ptr = (void*) (new_header + 1);

  mtx_unlock(&_memory_debug_mutex);

  return LUMINARY_SUCCESS;
}
#endif /* HOST_MEMORY_TRACKING_ENABLED */

void _host_memory_init(void) {
  atomic_store(&_host_memory_total_allocation, 0);

#ifdef HOST_MEMORY_TRACKING_ENABLED
  mtx_init(&_memory_debug_mutex, mtx_plain);

  _debug_memory_allocations_allocated = 0;
  _debug_memory_allocations_count     = 0;
  _debug_memory_allocations           = (MemoryDebugAllocation*) 0;
#endif /* HOST_MEMORY_TRACKING_ENABLED */
}

void _host_memory_shutdown(void) {
#ifdef LUMINARY_MEMORY_DEBUG
  _debug_memory_allocation_check_for_leaks();
#endif /* LUMINARY_MEMORY_DEBUG */

#ifdef HOST_MEMORY_TRACKING_ENABLED
  for (size_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++)
    free(_debug_memory_allocations[allocation].name);
#ifdef LUMINARY_ENABLE_DEBUG_API
  for (size_t allocation = 0; allocation < _debug_memory_allocations_count; allocation++)
    free(_debug_memory_allocations[allocation].stack_trace);
#endif

  free(_debug_memory_allocations);
  _debug_memory_allocations           = (MemoryDebugAllocation*) 0;
  _debug_memory_allocations_count     = 0;
  _debug_memory_allocations_allocated = 0;
  mtx_destroy(&_memory_debug_mutex);
#endif /* HOST_MEMORY_TRACKING_ENABLED */

  uint64_t leaked_memory = atomic_load(&_host_memory_total_allocation);

  if (leaked_memory > 0) {
    luminary_print_error("Luminary leaked %llu bytes.", leaked_memory);
  }
}

LuminaryResult _host_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(buf_name);
  __CHECK_NULL_ARGUMENT(func);

  if (size > SIZE_MAX - sizeof(struct HostMemoryHeader))
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Host allocation size is too large.");

  struct HostMemoryHeader* header = (struct HostMemoryHeader*) malloc(size + sizeof(struct HostMemoryHeader));

  if (header == (struct HostMemoryHeader*) 0)
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to realloc host memory.");

  memset(header, 0, sizeof(struct HostMemoryHeader));

  header->magic = HOST_MEMORY_HEADER_MAGIC;
  header->size  = size;

  uint64_t prev_total;

#ifdef HOST_MEMORY_TRACKING_ENABLED
  LuminaryResult tracking_result = _debug_memory_allocation_add((const void*) (header + 1), buf_name, func, line, size, &prev_total);

  if (tracking_result != LUMINARY_SUCCESS) {
    free(header);
    return tracking_result;
  }
#else
  prev_total = atomic_fetch_add(&_host_memory_total_allocation, header->size);
#endif /* HOST_MEMORY_TRACKING_ENABLED */

#ifdef LUMINARY_MEMORY_DEBUG
  luminary_print_log("Malloc  %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total + size, func, line, buf_name);
#else
  LUM_UNUSED(prev_total);
#endif /* LUMINARY_MEMORY_DEBUG */

  *ptr = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_realloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);
  __CHECK_NULL_ARGUMENT(buf_name);
  __CHECK_NULL_ARGUMENT(func);

  if (size > SIZE_MAX - sizeof(struct HostMemoryHeader))
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Host allocation size is too large.");

  struct HostMemoryHeader* header = ((struct HostMemoryHeader*) (*ptr)) - 1;

  if (header->magic != HOST_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Memory was not allocated through a call to host_memory.");
  }

  if (header->size > atomic_load(&_host_memory_total_allocation)) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Memory allocation is larger than total allocated memory.");
  }

  uint64_t prev_total;

#ifdef HOST_MEMORY_TRACKING_ENABLED
  LuminaryResult tracking_result = _debug_memory_allocation_realloc(ptr, size, buf_name, func, line, &prev_total);

  if (tracking_result != LUMINARY_SUCCESS)
    return tracking_result;
#else
  const uint64_t old_size             = header->size;
  struct HostMemoryHeader* new_header = realloc(header, size + sizeof(struct HostMemoryHeader));

  if (new_header == (struct HostMemoryHeader*) 0)
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to realloc host memory.");

  new_header->size = size;
  atomic_fetch_sub(&_host_memory_total_allocation, old_size);
  prev_total = atomic_fetch_add(&_host_memory_total_allocation, new_header->size);
  *ptr       = (void*) (new_header + 1);
#endif /* HOST_MEMORY_TRACKING_ENABLED */

#ifdef LUMINARY_MEMORY_DEBUG
  luminary_print_log("Realloc %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total + size, func, line, buf_name);
#else
  LUM_UNUSED(prev_total);
#endif /* LUMINARY_MEMORY_DEBUG */

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

  const uint64_t size = header->size;
  uint64_t prev_total;

#ifdef HOST_MEMORY_TRACKING_ENABLED
  LuminaryResult tracking_result = _debug_memory_allocation_remove(*ptr, buf_name, func, line, size, &prev_total);

  if (tracking_result != LUMINARY_SUCCESS)
    return tracking_result;
#else
  prev_total = atomic_fetch_sub(&_host_memory_total_allocation, header->size);
#endif /* HOST_MEMORY_TRACKING_ENABLED */

  header->magic = HOST_MEMORY_HEADER_FREED_MAGIC;

#ifdef LUMINARY_MEMORY_DEBUG
  luminary_print_log("Free    %012llu [Total: %012llu] [%s:%u]: %s", size, prev_total - size, func, line, buf_name);
#else
  LUM_UNUSED(prev_total);
#endif /* LUMINARY_MEMORY_DEBUG */

  free(header);

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_memory_get_allocations(LuminaryHostMemoryAllocationList* list) {
  __CHECK_NULL_ARGUMENT(list);

#ifdef LUMINARY_ENABLE_DEBUG_API

  mtx_lock(&_memory_debug_mutex);

  const size_t allocation_count = _debug_memory_allocations_count;

  if (allocation_count > SIZE_MAX / sizeof(LuminaryHostMemoryAllocation)) {
    mtx_unlock(&_memory_debug_mutex);
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Host allocation snapshot is too large.");
  }

  LuminaryHostMemoryAllocation* copied_allocations = (LuminaryHostMemoryAllocation*) 0;

  if (allocation_count > 0) {
    copied_allocations = calloc(allocation_count, sizeof(LuminaryHostMemoryAllocation));

    if (copied_allocations == (LuminaryHostMemoryAllocation*) 0) {
      mtx_unlock(&_memory_debug_mutex);
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to allocate host allocation snapshot.");
    }
  }

  for (size_t allocation = 0; allocation < allocation_count; allocation++) {
    copied_allocations[allocation].ptr  = _debug_memory_allocations[allocation].ptr;
    copied_allocations[allocation].size = _debug_memory_allocations[allocation].size;

    if (_debug_memory_allocations[allocation].name != (char*) 0) {
      const size_t name_size = strlen(_debug_memory_allocations[allocation].name) + 1;
      char* copied_name      = malloc(name_size);

      if (copied_name == (char*) 0) {
        for (size_t cleanup = 0; cleanup < allocation; cleanup++) {
          free((void*) copied_allocations[cleanup].name);
          free((void*) copied_allocations[cleanup].stack_trace);
        }

        free(copied_allocations);
        mtx_unlock(&_memory_debug_mutex);
        __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to copy host allocation snapshot name.");
      }

      memcpy(copied_name, _debug_memory_allocations[allocation].name, name_size);
      copied_allocations[allocation].name = copied_name;
    }

    if (_debug_memory_allocations[allocation].stack_trace != (char*) 0) {
      const size_t stack_trace_size = strlen(_debug_memory_allocations[allocation].stack_trace) + 1;
      char* copied_stack_trace      = (char*) malloc(stack_trace_size);

      if (copied_stack_trace == (char*) 0) {
        for (size_t cleanup = 0; cleanup <= allocation; cleanup++) {
          free((void*) copied_allocations[cleanup].name);
          free((void*) copied_allocations[cleanup].stack_trace);
        }

        free(copied_allocations);
        mtx_unlock(&_memory_debug_mutex);
        __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to copy host allocation snapshot stack trace.");
      }

      memcpy(copied_stack_trace, _debug_memory_allocations[allocation].stack_trace, stack_trace_size);
      copied_allocations[allocation].stack_trace = copied_stack_trace;
    }
  }

  list->allocations = copied_allocations;
  list->count       = allocation_count;

  mtx_unlock(&_memory_debug_mutex);

#endif /* LUMINARY_ENABLE_DEBUG_API */

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_memory_allocation_list_destroy(LuminaryHostMemoryAllocationList* list) {
  __CHECK_NULL_ARGUMENT(list);

#ifdef LUMINARY_ENABLE_DEBUG_API

  for (uint64_t allocation = 0; allocation < list->count; allocation++) {
    free((void*) list->allocations[allocation].name);
    free((void*) list->allocations[allocation].stack_trace);
  }

  free(list->allocations);
  list->allocations = (LuminaryHostMemoryAllocation*) 0;
  list->count       = 0;

#endif /* LUMINARY_ENABLE_DEBUG_API */

  return LUMINARY_SUCCESS;
}
