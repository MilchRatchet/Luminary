#include "host_local_memory.h"

#include <stdatomic.h>
#include <string.h>
#include <threads.h>

#include "internal_error.h"

#define LOCAL_STACK_ALLOCATOR_DEFAULT_SIZE (1024 * 1024 * 2)  // 2MB
#define LOCAL_STACK_ALLOCATOR_NOT_INITIALIZED ((size_t) -1)

#define LOCAL_ALLOCATOR_MAGIC_INITIALIZED (0xCAFEBABE)
#define LOCAL_ALLOCATOR_MAGIC_NOT_INITIALIZED (0xBAADF00D)
#define LOCAL_ALLOCATION_MAGIC (0xABCDEF01)
#define LOCAL_ALLOCATION_ID_INVALID (0xFFFFFFFF)
#define LOCAL_ALLOCATION_ALIGNMENT (0x100)

struct LocalMemoryAllocationHeader {
  union {
    struct {
      uint32_t allocation_id;
      uint32_t magic;
      uint32_t allocator_id;
    };

    uint8_t padding[LOCAL_ALLOCATION_ALIGNMENT];
  };
} typedef LocalMemoryAllocationHeader;
LUM_STATIC_SIZE_ASSERT(LocalMemoryAllocationHeader, LOCAL_ALLOCATION_ALIGNMENT);

struct LocalMemoryAllocation {
  size_t size;
  void* memory;
  bool is_available;
} typedef LocalMemoryAllocation;

struct LocalAllocator {
  ARRAY LocalMemoryAllocation* allocations;
  uint32_t magic;
  uint32_t id;
} typedef LocalAllocator;

static _Atomic uint32_t _thread_local_allocator_counter     = 0;
static _Thread_local LocalAllocator _thread_local_allocator = {.magic = LOCAL_ALLOCATOR_MAGIC_NOT_INITIALIZED};

LuminaryResult _host_memory_local_init() {
  memset(&_thread_local_allocator, 0, sizeof(LocalAllocator));

  __FAILURE_HANDLE(array_create(&_thread_local_allocator.allocations, sizeof(LocalMemoryAllocation), 16));

  _thread_local_allocator.id    = atomic_fetch_add(&_thread_local_allocator_counter, 1);
  _thread_local_allocator.magic = LOCAL_ALLOCATOR_MAGIC_INITIALIZED;

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_memory_local_shutdown() {
  uint32_t num_allocations;
  __FAILURE_HANDLE(array_get_num_elements(_thread_local_allocator.allocations, &num_allocations));

  for (uint32_t allocation_id = 0; allocation_id < num_allocations; allocation_id++) {
    LocalMemoryAllocation* allocation = _thread_local_allocator.allocations + allocation_id;

    if (allocation->is_available == false)
      __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Local memory has not been freed.");

    if (allocation->memory)
      __FAILURE_HANDLE(host_free(&allocation->memory));
  }

  __FAILURE_HANDLE(array_destroy(&_thread_local_allocator.allocations));

  _thread_local_allocator.magic = LOCAL_ALLOCATOR_MAGIC_NOT_INITIALIZED;

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_malloc_local(void** ptr, size_t size) {
  __CHECK_NULL_ARGUMENT(ptr);

  uint32_t num_allocations;
  __FAILURE_HANDLE(array_get_num_elements(_thread_local_allocator.allocations, &num_allocations));

  uint32_t selected_allocation = LOCAL_ALLOCATION_ID_INVALID;
  size_t selected_size         = 0;

  for (uint32_t allocation_id = 0; allocation_id < num_allocations; allocation_id++) {
    LocalMemoryAllocation* allocation = _thread_local_allocator.allocations + allocation_id;

    if (allocation->is_available == false)
      continue;

    if (allocation->size > selected_size) {
      selected_allocation = allocation_id;
      selected_size       = allocation->size;
    }

    if (selected_size >= size)
      break;
  }

  if (selected_allocation == LOCAL_ALLOCATION_ID_INVALID) {
    selected_size = size;

    LocalMemoryAllocation new_allocation;
    memset(&new_allocation, 0, sizeof(LocalMemoryAllocation));

    __FAILURE_HANDLE(host_malloc(&new_allocation.memory, size + sizeof(LocalMemoryAllocationHeader)));
    new_allocation.size = size;

    __FAILURE_HANDLE(array_get_num_elements(_thread_local_allocator.allocations, &selected_allocation));

    __FAILURE_HANDLE(array_push(&_thread_local_allocator.allocations, &new_allocation));

    LocalMemoryAllocationHeader* header = (LocalMemoryAllocationHeader*) new_allocation.memory;
    header->allocation_id               = selected_allocation;
    header->magic                       = LOCAL_ALLOCATION_MAGIC;
    header->allocator_id                = _thread_local_allocator.id;
  }

  _thread_local_allocator.allocations[selected_allocation].is_available = false;

  *ptr = (void*) (((LocalMemoryAllocationHeader*) (_thread_local_allocator.allocations[selected_allocation].memory)) + 1);

  if (selected_size < size)
    __FAILURE_HANDLE(host_realloc_local(ptr, size));

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_realloc_local(void** ptr, size_t size) {
  __CHECK_NULL_ARGUMENT(ptr);

  LocalMemoryAllocationHeader* header = (void*) (((LocalMemoryAllocationHeader*) (*ptr)) - 1);

  if (header->magic != LOCAL_ALLOCATION_MAGIC)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given memory is not valid local host memory.");

  if (header->allocation_id == LOCAL_ALLOCATION_ID_INVALID)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Local memory has been corrupted.");

  if (header->allocator_id != _thread_local_allocator.id)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Local memory has been allocated by another local memory allocator.");

  // Header will be invalid after the realloc.
  const uint32_t allocation_id = header->allocation_id;

  // Invalidate header already.
  header = (LocalMemoryAllocationHeader*) 0;

  LocalMemoryAllocation* allocation = _thread_local_allocator.allocations + allocation_id;

  if (allocation->is_available)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Local memory has not been acquired but is being reallocated.");

  // Since we want to avoid unnecessary reallocations, we forbid shrinking the memory. Since the caller does not know the actual size
  // of the buffer, it is likely for shrinking calls to appear. Most of these calls actually only have the intention to increase buffer size
  // and should never have the intention of reducing memory usage.
  if (allocation->size >= size)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(host_realloc(&allocation->memory, size + sizeof(LocalMemoryAllocationHeader)));
  allocation->size = size;

  *ptr = (void*) (((LocalMemoryAllocationHeader*) (_thread_local_allocator.allocations[allocation_id].memory)) + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _host_free_local(void** ptr) {
  __CHECK_NULL_ARGUMENT(ptr);

  LocalMemoryAllocationHeader* header = (void*) (((LocalMemoryAllocationHeader*) (*ptr)) - 1);

  if (header->magic != LOCAL_ALLOCATION_MAGIC)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given memory is not valid local host memory.");

  if (header->allocation_id == LOCAL_ALLOCATION_ID_INVALID)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Local memory has been corrupted.");

  if (header->allocator_id != _thread_local_allocator.id)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Local memory has been allocated by another local memory allocator.");

  LocalMemoryAllocation* allocation = _thread_local_allocator.allocations + header->allocation_id;

  allocation->is_available = true;

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}
